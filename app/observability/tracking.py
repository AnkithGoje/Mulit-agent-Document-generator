import logging
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from app.config import Settings
from app.schemas import TraceInfo


logger = logging.getLogger(__name__)


def create_trace_client(settings: Settings) -> Client | None:
    if not settings.langsmith_tracing:
        return None
    if not settings.langsmith_api_key.get_secret_value():
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is empty.")
        return None
    return Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
    )


def build_run_config(
    settings: Settings, run_id: UUID, client: Client | None
) -> RunnableConfig:
    config: RunnableConfig = {
        "run_id": run_id,
        "run_name": "autonomous-document-agent",
        "tags": ["docx", "langgraph", settings.llm_provider],
        "metadata": {
            "provider": settings.llm_provider,
            "model": settings.model_name,
            "engineering_improvement": "reflection-self-check",
        },
        "recursion_limit": 50,
        "callbacks": [],
    }
    if client is not None:
        config["callbacks"] = [
            LangChainTracer(project_name=settings.langsmith_project, client=client)
        ]
    return config


def trace_info(settings: Settings, run_id: UUID, client: Client | None) -> TraceInfo:
    return TraceInfo(
        enabled=client is not None,
        project=settings.langsmith_project if client is not None else None,
        run_id=str(run_id) if client is not None else None,
    )
