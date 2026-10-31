import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langsmith import tracing_context

from app.config import Settings
from app.graph.build import build_graph
from app.llm.providers import (
    LLMClient, LLMConfigurationError, LLMError, LLMRateLimitError, StructuredLLM,
)
from app.observability.logging import configure_logging
from app.observability.tracking import build_run_config, create_trace_client, trace_info
from app.schemas import AgentRequest, AgentResponse, DocumentArtifact


logger = logging.getLogger(__name__)
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def create_app(settings: Settings | None = None, llm: StructuredLLM | None = None) -> FastAPI:
    settings = settings or Settings()
    graph = build_graph(settings, llm if llm is not None else LLMClient(settings))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging()
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            application.state.trace_client = create_trace_client(settings)
        except Exception as exc:
            logger.warning("LangSmith initialization failed error_type=%s", type(exc).__name__)
            application.state.trace_client = None
        yield

    application = FastAPI(
        title="Autonomous Word Document Agent",
        version="1.0.0",
        description=(
            "A LangGraph agent that plans, executes, reviews, revises, and renders "
            "business documents. POST /agent returns the plan and a DOCX download URL, "
            "with optional Base64 document content."
        ),
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health():
        return {"status": "ok", "provider": settings.llm_provider, "model": settings.model_name}

    @application.post(
        "/agent",
        response_model=AgentResponse,
        responses={
            429: {"description": "Model provider rate limit reached."},
            502: {"description": "Language model execution failed."},
            503: {"description": "Language model credentials are not configured."},
            504: {"description": "Agent execution exceeded the time limit."},
        },
    )
    async def run_agent(payload: AgentRequest, request: Request) -> AgentResponse:
        run_id = uuid4()
        started_at = time.perf_counter()
        client = request.app.state.trace_client
        warnings = []
        if settings.langsmith_tracing and client is None:
            warnings.append("LangSmith is not configured; execution continued without tracing.")
        config = build_run_config(settings, run_id, client)
        initial_state = {
            "request": payload.request,
            "run_id": run_id.hex,
            "task_index": 0,
            "task_results": [],
            "revisions": 0,
            "warnings": warnings,
        }
        result = None

        try:
            with tracing_context(
                enabled=client is not None,
                project_name=settings.langsmith_project,
                client=client,
            ):
                async with asyncio.timeout(settings.agent_timeout_seconds):
                    result = await graph.ainvoke(initial_state, config=config)
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMRateLimitError as exc:
            logger.warning("Agent rate limited run_id=%s", run_id)
            headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
            raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
        except LLMError as exc:
            logger.warning("Agent model failure run_id=%s", run_id)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except TimeoutError as exc:
            logger.warning("Agent timeout run_id=%s", run_id)
            raise HTTPException(
                status_code=504,
                detail="Agent execution timed out. Try a smaller request or increase AGENT_TIMEOUT_SECONDS.",
            ) from exc
        except OSError as exc:
            logger.error("Document save failed run_id=%s error_type=%s", run_id, type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="The document could not be saved. Check OUTPUT_DIR write permissions.",
            ) from exc
        except Exception as exc:
            logger.error("Agent failed run_id=%s error_type=%s", run_id, type(exc).__name__)
            raise HTTPException(status_code=500, detail="Agent execution failed.") from exc
        finally:
            if result is None:
                try:
                    (settings.output_dir / f"{run_id.hex}.docx").unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to discard incomplete artifact run_id=%s", run_id)

        has_warnings = bool(result["warnings"]) or not result["review"].approved
        logger.info(
            "Agent completed run_id=%s tasks=%d revisions=%d approved=%s",
            run_id, len(result["task_results"]), result["revisions"], result["review"].approved,
        )
        encoded = None
        if payload.include_base64:
            doc_path = settings.output_dir / result["document_filename"]
            encoded = base64.b64encode(await asyncio.to_thread(doc_path.read_bytes)).decode("ascii")
        return AgentResponse(
            run_id=str(run_id),
            status="completed_with_warnings" if has_warnings else "completed",
            message=(
                f"Created '{result['document'].title}' after executing "
                f"{len(result['task_results'])} planned tasks and "
                f"{result['revisions']} revision(s)."
            ),
            summary=result["document"].summary,
            plan=result["plan"],
            tasks=result["task_results"],
            review=result["review"],
            revisions=result["revisions"],
            warnings=result["warnings"],
            document=DocumentArtifact(
                filename=result["document_filename"],
                download_url=str(request.url_for("download_document", document_id=run_id.hex)),
                base64=encoded,
            ),
            trace=trace_info(settings, run_id, client),
            elapsed_seconds=round(time.perf_counter() - started_at, 2),
        )

    @application.get("/documents/{document_id}", name="download_document")
    async def download_document(document_id: UUID):
        filename = f"{document_id.hex}.docx"
        path = settings.output_dir / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Document not found.")
        return FileResponse(path, media_type=DOCX_MEDIA_TYPE, filename=filename)

    return application


app = create_app()
