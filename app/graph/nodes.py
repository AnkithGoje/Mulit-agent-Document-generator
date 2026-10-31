import asyncio
import logging
from uuid import UUID

from langchain_core.runnables import RunnableConfig

from app.config import Settings
from app.graph.state import AgentState
from app.llm.prompt import EXECUTOR_PROMPT, PLANNER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT
from app.llm.providers import StructuredLLM
from app.observability.logging import timed
from app.rendering.docx_builder import render_docx, save_docx
from app.schemas import CompletedTask, DocumentContent, DocumentPlan, ReviewResult, TaskResult


logger = logging.getLogger(__name__)


def _compact_tasks(results, max_findings: int = 8) -> list[dict]:
    return [
        {
            "number": r.number,
            "title": r.title,
            "summary": r.result.summary,
            "findings": r.result.findings[:max_findings],
        }
        for r in results
    ]


def _plan_brief(plan) -> dict:
    return {
        "title": plan.title,
        "document_type": plan.document_type,
        "audience": plan.audience,
        "interpretation": plan.interpretation,
        "assumptions": plan.assumptions,
        "conflicts": plan.conflicts,
    }


class AgentNodes:
    def __init__(self, settings: Settings, llm: StructuredLLM):
        self.settings = settings
        self.llm = llm

    @timed("planner")
    async def plan(self, state: AgentState, config: RunnableConfig) -> dict:
        logger.info("Planning run_id=%s", state["run_id"])
        plan = await self.llm.generate(
            DocumentPlan, PLANNER_PROMPT, {"request": state["request"]}, config=config
        )
        return {"plan": plan, "task_index": 0, "task_results": [], "revisions": 0}

    @timed("execute_task")
    async def execute_task(self, state: AgentState, config: RunnableConfig) -> dict:
        index = state["task_index"]
        task = state["plan"].tasks[index]
        logger.info("Executing run_id=%s task=%d", state["run_id"], index + 1)
        result = await self.llm.generate(
            TaskResult,
            EXECUTOR_PROMPT,
            {
                "request": state["request"],
                "plan": state["plan"].model_dump(mode="json"),
                "current_task": task.model_dump(mode="json"),
                "earlier_results": [item.model_dump(mode="json") for item in state["task_results"]],
            },
            config=config,
        )
        completed = CompletedTask(
            number=index + 1, title=task.title, instructions=task.instructions, result=result
        )
        return {
            "task_results": [*state["task_results"], completed],
            "task_index": index + 1,
        }

    @timed("compose")
    async def compose(self, state: AgentState, config: RunnableConfig) -> dict:
        logger.info("Composing run_id=%s", state["run_id"])
        document = await self.llm.generate(
            DocumentContent,
            WRITER_PROMPT,
            {
                "request": state["request"],
                "plan": state["plan"].model_dump(mode="json"),
                "completed_tasks": _compact_tasks(state["task_results"]),
            },
            config=config,
        )
        return {"document": document}

    @timed("review")
    async def review(self, state: AgentState, config: RunnableConfig) -> dict:
        logger.info("Reviewing run_id=%s revision=%d", state["run_id"], state["revisions"])
        review = await self.llm.generate(
            ReviewResult,
            REVIEWER_PROMPT,
            {
                "request": state["request"],
                "plan": _plan_brief(state["plan"]),
                "document": state["document"].model_dump(mode="json"),
            },
            config=config,
        )
        unmet = [c.requirement for c in review.checklist if not c.met]
        approved = review.approved and review.score >= 7 and not unmet
        feedback = list(review.feedback)
        for requirement in unmet:
            if not any(requirement in item for item in feedback):
                feedback.append(f"Requirement not met: {requirement}")
        if not approved and not any(item.strip() for item in feedback):
            feedback.append("Raise the draft quality: address coverage and consistency gaps.")
        return {"review": review.model_copy(update={"approved": approved, "feedback": feedback})}

    @timed("revise")
    async def revise(self, state: AgentState, config: RunnableConfig) -> dict:
        logger.info("Revising run_id=%s revision=%d", state["run_id"], state["revisions"] + 1)
        document = await self.llm.generate(
            DocumentContent,
            WRITER_PROMPT,
            {
                "request": state["request"],
                "plan": state["plan"].model_dump(mode="json"),
                "completed_tasks": _compact_tasks(state["task_results"]),
                "current_draft": state["document"].model_dump(mode="json"),
                "review_feedback": state["review"].feedback,
            },
            config=config,
        )
        return {"document": document, "revisions": state["revisions"] + 1}

    @timed("render")
    async def render(self, state: AgentState) -> dict:
        warnings = list(state.get("warnings", []))
        if not state["review"].approved:
            warnings.append(
                "The revision limit was reached. This document still has review issues "
                "and requires human review."
            )
        filename = f"{UUID(state['run_id']).hex}.docx"
        data = await asyncio.to_thread(
            render_docx,
            state["document"],
            state["plan"],
            warnings,
            state["review"],
        )
        save_docx(data, self.settings.output_dir / filename)
        logger.info("Rendered run_id=%s file=%s", state["run_id"], filename)
        return {"document_filename": filename, "warnings": warnings}
