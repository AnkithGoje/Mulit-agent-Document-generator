import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4
from zipfile import ZipFile

import pytest
from docx import Document
from langchain_core.runnables import RunnableConfig
from langsmith import tracing_context
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.graph.build import build_graph
from app.graph.nodes import _compact_tasks
from app.llm.providers import LLMError
from app.rendering.docx_builder import render_docx
from app.schemas import (
    AgentRequest,
    DocumentContent,
    DocumentPlan,
    DocumentSection,
    DocumentTable,
    PlannedTask,
    RequirementCheck,
    ReviewResult,
    TaskResult,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
ROOT = Path(__file__).resolve().parents[1]
MOCK_ASSUMPTION = "All figures and dates are illustrative mock data; no external research was performed."
REVIEW_FEEDBACK = "Identify an accountable owner for the recommended next step."


class FakeLLM:
    """Deterministic structured responses shared by the offline graph and API tests."""

    def __init__(
        self,
        task_count: int = 3,
        reviews: tuple[bool | ReviewResult, ...] = (True,),
        fail_task: int | None = None,
    ):
        self.task_count = task_count
        self.reviews = reviews
        self.fail_task = fail_task
        self.calls: list[tuple[type[BaseModel], dict[str, Any]]] = []
        self.configs: list[RunnableConfig | None] = []
        self.task_calls = 0
        self.review_calls = 0
        self.documents: list[DocumentContent] = []

    async def generate(
        self,
        schema: type[ModelT],
        system_prompt: str,
        user_payload: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> ModelT:
        assert system_prompt.strip()
        self.calls.append((schema, user_payload))
        self.configs.append(config)
        result: BaseModel
        if schema is DocumentPlan:
            titles = [
                "Scope and assumptions",
                "Milestones and owners",
                "Illustrative budget",
                "Scenario comparison and risks",
            ]
            result = DocumentPlan(
                title="Illustrative Implementation Brief",
                document_type="Business brief",
                audience="Project sponsors",
                interpretation="Prepare a concise business brief using illustrative data.",
                assumptions=[MOCK_ASSUMPTION],
                tasks=[
                    PlannedTask(title=title, instructions=f"Develop {title.lower()} using mock data.")
                    for title in titles[: self.task_count]
                ],
            )
        elif schema is TaskResult:
            self.task_calls += 1
            if self.task_calls == self.fail_task:
                raise LLMError("The test provider could not produce a valid result.") from RuntimeError(
                    "private upstream response"
                )
            title = user_payload["current_task"]["title"]
            result = TaskResult(
                summary=f"Completed analysis for {title}.",
                findings=[f"Illustrative finding for {title}; validate before use."],
            )
        elif schema is DocumentContent:
            revised = "review_feedback" in user_payload
            sections = [
                DocumentSection(
                    heading="Recommendation",
                    paragraphs=[
                        "The operations lead owns the next step."
                        if revised
                        else "Confirm an owner before authorizing the pilot."
                    ],
                    bullets=["Validate the mock estimates before making a decision."],
                ),
                DocumentSection(
                    heading="Illustrative Budget",
                    paragraphs=["These amounts are illustrative mock data, not verified costs."],
                    table=DocumentTable(
                        caption="Mock budget in USD",
                        headers=["Phase", "Illustrative cost"],
                        rows=[["Preparation", "12000"], ["Pilot", "18000"], ["Total", "30000"]],
                    ),
                ),
            ]
            if self.task_count == 4:
                sections.extend(
                    [
                        DocumentSection(
                            heading="Scenario Comparison",
                            table=DocumentTable(
                                caption="Mock scenarios, not forecasts",
                                headers=["Scenario", "Illustrative outcome"],
                                rows=[["Base", "Phased rollout"], ["Stressed", "Delay one quarter"]],
                            ),
                        ),
                        DocumentSection(
                            heading="Risk Register",
                            table=DocumentTable(
                                caption="Illustrative risks requiring human validation",
                                headers=["Risk", "Owner", "Mitigation"],
                                rows=[["Capacity shortfall", "Operations lead", "Stage the rollout"]],
                            ),
                        ),
                    ]
                )
            result = DocumentContent(
                title=user_payload["plan"]["title"],
                summary="Revised illustrative draft with an accountable owner."
                if revised
                else "Initial illustrative draft for human review.",
                sections=sections,
            )
            self.documents.append(result)
        elif schema is ReviewResult:
            verdict = self.reviews[min(self.review_calls, len(self.reviews) - 1)]
            self.review_calls += 1
            result = verdict if isinstance(verdict, ReviewResult) else ReviewResult(
                approved=verdict,
                score=9 if verdict else 5,
                feedback=[] if verdict else [REVIEW_FEEDBACK],
                checklist=[RequirementCheck(requirement="Label mock data", met=True)],
            )
        else:
            raise AssertionError(f"Unexpected schema: {schema.__name__}")
        return cast(ModelT, result)


def example_payload(name: str) -> dict[str, str]:
    return json.loads((ROOT / "examples" / f"{name}_request.json").read_text(encoding="utf-8"))


def offline_settings(output_dir: Path, max_revisions: int = 1) -> Settings:
    return Settings(
        output_dir=output_dir,
        _env_file=None,
        llm_provider="groq",
        groq_api_key="",
        groq_model="openai/gpt-oss-20b",
        langsmith_tracing=False,
        langsmith_api_key="",
        max_revisions=max_revisions,
    )


def initial_state(request: str) -> dict[str, Any]:
    return {
        "request": request,
        "run_id": uuid4().hex,
        "task_index": 0,
        "task_results": [],
        "revisions": 0,
        "warnings": [],
    }


def invoke_graph(
    settings: Settings, llm: FakeLLM, request: str, config: RunnableConfig | None = None
) -> dict[str, Any]:
    # Suppress ambient LangSmith tracing even if the developer's shell enables it.
    with tracing_context(enabled=False):
        return asyncio.run(build_graph(settings, llm).ainvoke(initial_state(request), config=config))


def assert_valid_docx(data: bytes, content: DocumentContent, plan: DocumentPlan) -> str:
    with ZipFile(BytesIO(data)) as archive:
        assert archive.testzip() is None
        assert {"[Content_Types].xml", "word/document.xml", "docProps/core.xml"} <= set(
            archive.namelist()
        )
    document = Document(BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert document.core_properties.title == content.title
    assert content.title in text
    assert content.summary in text
    assert "Assumptions and Limitations" in text
    assert all(assumption in text for assumption in plan.assumptions)
    assert all(conflict in text for conflict in plan.conflicts)
    tables = [section.table for section in content.sections if section.table is not None]
    assert tables
    assert len(document.tables) == len(tables)
    for actual, expected in zip(document.tables, tables):
        assert [[cell.text for cell in row.cells] for row in actual.rows] == [
            expected.headers,
            *expected.rows,
        ]
    return text


def test_graph_executes_all_distinct_tasks_and_renders_docx(tmp_path):
    llm = FakeLLM(task_count=4)
    request = AgentRequest.model_validate(example_payload("complex")).request
    result = invoke_graph(offline_settings(tmp_path), llm, request)

    tasks = result["task_results"]
    assert result["task_index"] == len(tasks) == 4
    assert [task.title for task in tasks] == [task.title for task in result["plan"].tasks]
    assert len({task.title.casefold() for task in tasks}) == 4
    assert [task.number for task in tasks] == list(range(1, 5))
    assert all(task.status == "completed" and isinstance(task.result, TaskResult) for task in tasks)
    payloads = [payload for schema, payload in llm.calls if schema is TaskResult]
    assert [len(payload["earlier_results"]) for payload in payloads] == list(range(4))
    assert [payload["current_task"]["title"] for payload in payloads] == [task.title for task in tasks]
    assert [schema for schema, _ in llm.calls] == [
        DocumentPlan,
        *([TaskResult] * 4),
        DocumentContent,
        ReviewResult,
    ]
    assert result["review"].approved
    assert result["revisions"] == 0
    assert result["warnings"] == []
    filename = f"{result['run_id']}.docx"
    assert result["document_filename"] == filename
    assert [path.name for path in tmp_path.iterdir()] == [filename]
    assert_valid_docx((tmp_path / filename).read_bytes(), result["document"], result["plan"])


def test_reflection_revises_then_approves(tmp_path):
    llm = FakeLLM(reviews=(False, True))
    result = invoke_graph(offline_settings(tmp_path), llm, example_payload("standard")["request"])

    assert result["review"].approved
    assert result["revisions"] == 1
    assert result["warnings"] == []
    assert llm.task_calls == 3
    assert llm.review_calls == len(llm.documents) == 2
    revision = [payload for schema, payload in llm.calls if schema is DocumentContent][-1]
    assert revision["review_feedback"] == [REVIEW_FEEDBACK]
    assert revision["current_draft"] == llm.documents[0].model_dump(mode="json")
    assert len(revision["completed_tasks"]) == 3
    assert revision["completed_tasks"] == _compact_tasks(result["task_results"])
    assert result["document"].summary != llm.documents[0].summary
    assert [schema for schema, _ in llm.calls][-4:] == [
        DocumentContent,
        ReviewResult,
        DocumentContent,
        ReviewResult,
    ]


@pytest.mark.parametrize("budget", [0, 2])
def test_exhausted_revision_budget_preserves_rejection_and_warns(tmp_path, budget):
    llm = FakeLLM(reviews=(False,))
    result = invoke_graph(
        offline_settings(tmp_path, max_revisions=budget), llm, example_payload("standard")["request"]
    )

    assert result["revisions"] == budget
    assert not result["review"].approved
    assert result["review"].feedback == [REVIEW_FEEDBACK]
    assert llm.review_calls == len(llm.documents) == budget + 1
    assert result["warnings"]
    text = assert_valid_docx(
        (tmp_path / result["document_filename"]).read_bytes(), result["document"], result["plan"]
    )
    assert "REVIEW REQUIRED" in text
    assert REVIEW_FEEDBACK in text
    assert all(warning in text for warning in result["warnings"])


def test_failed_task_is_not_marked_completed_and_produces_no_artifact(tmp_path):
    llm = FakeLLM(fail_task=2)

    async def collect_updates():
        updates = []
        graph = build_graph(offline_settings(tmp_path), llm)
        with pytest.raises(LLMError):
            async for update in graph.astream(
                initial_state(example_payload("standard")["request"]), stream_mode="updates"
            ):
                updates.append(update)
        return updates

    with tracing_context(enabled=False):
        updates = asyncio.run(collect_updates())
    task_updates = [update["execute_task"] for update in updates if "execute_task" in update]
    assert len(task_updates) == 1
    assert task_updates[0]["task_index"] == 1
    assert len(task_updates[0]["task_results"]) == 1
    assert task_updates[0]["task_results"][0].number == 1
    assert llm.task_calls == 2
    assert llm.documents == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length", [15, 6000])
def test_request_schema_accepts_length_boundaries(length):
    assert AgentRequest(request="x" * length).request == "x" * length


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": ["A", "B"], "rows": [["too short"]]},
        {"headers": ["A"], "rows": [["too", "long"]]},
        {"headers": [], "rows": [["value"]]},
        {"headers": ["A"], "rows": []},
    ],
)
def test_table_schema_rejects_invalid_shape(payload):
    with pytest.raises(ValidationError):
        DocumentTable.model_validate(payload)


@pytest.mark.parametrize(
    "schema,payload",
    [
        (AgentRequest, {"request": "A valid-length request with a \x00 character."}),
        (TaskResult, {"summary": "Valid summary", "findings": ["Bad\x0btext"]}),
        (DocumentTable, {"headers": ["Item"], "rows": [["Bad\x00cell"]]}),
    ],
)
def test_schemas_reject_xml_unsafe_text(schema, payload):
    with pytest.raises(ValidationError, match="Word/XML"):
        schema.model_validate(payload)


def test_schemas_preserve_xml_safe_whitespace():
    text = "First\tsecond\nthird\rfourth"
    result = TaskResult(summary=text, findings=[text])
    table = DocumentTable(headers=["Item"], rows=[[text]])
    assert result.summary == text
    assert result.findings == [text]
    assert table.rows == [[text]]


@pytest.mark.parametrize(
    "review,expected_feedback",
    [
        (ReviewResult(approved=True, score=6), "Raise the draft quality"),
        (
            ReviewResult(
                approved=True,
                score=9,
                checklist=[RequirementCheck(requirement="Include a risk register", met=False)],
            ),
            "Requirement not met: Include a risk register",
        ),
    ],
)
def test_review_node_corrects_inconsistent_approval(tmp_path, review, expected_feedback):
    llm = FakeLLM(reviews=(review,))
    result = invoke_graph(
        offline_settings(tmp_path, max_revisions=0), llm, example_payload("standard")["request"]
    )
    assert not result["review"].approved
    assert any(expected_feedback in item for item in result["review"].feedback)
    assert result["review"].checklist == review.checklist
    assert review.approved
    assert llm.review_calls == 1


def test_plan_schema_rejects_more_than_four_tasks():
    with pytest.raises(ValidationError):
        DocumentPlan(
            title="Test plan",
            document_type="Business brief",
            audience="Sponsors",
            tasks=[PlannedTask(title=f"Task {i}", instructions="Analyze mock data.") for i in range(5)],
        )


def test_compact_payloads_timing_and_config_propagation(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="app.timing")
    llm = FakeLLM()
    result = invoke_graph(
        offline_settings(tmp_path),
        llm,
        example_payload("standard")["request"],
        config={"tags": ["timing-check"], "metadata": {"config_marker": "preserved"}},
    )
    assert all(config is not None for config in llm.configs)
    assert all(config["metadata"]["config_marker"] == "preserved" for config in llm.configs)
    assert all("timing-check" in config["tags"] for config in llm.configs)
    labels = {
        record.getMessage().split()[0].split("=", 1)[1]
        for record in caplog.records if record.name == "app.timing"
    }
    assert {"planner", "execute_task", "compose", "review", "render"} <= labels
    writer_payload = next(payload for schema, payload in llm.calls if schema is DocumentContent)
    assert writer_payload["completed_tasks"] == _compact_tasks(result["task_results"])
    assert all("instructions" not in item for item in writer_payload["completed_tasks"])
    review_payload = next(payload for schema, payload in llm.calls if schema is ReviewResult)
    assert "tasks" not in review_payload["plan"]
    assert review_payload["plan"]["interpretation"] == result["plan"].interpretation
    assert review_payload["plan"]["conflicts"] == result["plan"].conflicts


def test_compaction_truncates_findings_and_docx_renders_conflicts(tmp_path):
    result = invoke_graph(offline_settings(tmp_path), FakeLLM(), example_payload("standard")["request"])
    task = result["task_results"][0]
    task.result.findings = [f"Mock finding {i}" for i in range(12)]
    assert len(_compact_tasks([task])[0]["findings"]) == 8
    conflict = "The full rollout cannot fit the budget; use a smaller pilot first."
    plan = result["plan"].model_copy(update={"conflicts": [conflict]})
    data = render_docx(result["document"], plan, review=result["review"])
    text = assert_valid_docx(data, result["document"], plan)
    assert "Conflicts and Resolutions" in text
