from typing import TypedDict

from app.schemas import CompletedTask, DocumentContent, DocumentPlan, ReviewResult


class AgentState(TypedDict, total=False):
    request: str
    run_id: str
    plan: DocumentPlan
    task_index: int
    task_results: list[CompletedTask]
    document: DocumentContent
    review: ReviewResult
    revisions: int
    warnings: list[str]
    document_filename: str
