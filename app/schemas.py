import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="after")
    @classmethod
    def xml_safe_text(cls, value: Any) -> Any:
        if isinstance(value, str) and INVALID_XML_CHARACTERS.search(value):
            raise ValueError("Text contains a character not supported by Word/XML.")
        if isinstance(value, list):
            for item in value:
                cls.xml_safe_text(item)
        elif isinstance(value, dict):
            for item in value.values():
                cls.xml_safe_text(item)
        return value


class AgentRequest(StructuredModel):
    request: str = Field(min_length=15, max_length=6000)
    include_base64: bool = False


class PlannedTask(StructuredModel):
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1, max_length=2000)


class DocumentPlan(StructuredModel):
    title: str = Field(min_length=1, max_length=200)
    document_type: str = Field(min_length=1, max_length=80)
    audience: str = Field(min_length=1, max_length=200)
    interpretation: str = Field(default="", max_length=800)
    assumptions: list[str] = Field(default_factory=list, max_length=15)
    conflicts: list[str] = Field(default_factory=list, max_length=10)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def distinct_tasks(self) -> "DocumentPlan":
        titles = [task.title.casefold() for task in self.tasks]
        if len(set(titles)) != len(titles):
            raise ValueError("Each task must have a distinct title.")
        return self


class TaskResult(StructuredModel):
    summary: str = Field(min_length=1, max_length=1500)
    findings: list[str] = Field(min_length=1, max_length=20)


class CompletedTask(StructuredModel):
    number: int = Field(ge=1)
    title: str
    instructions: str
    status: Literal["completed"] = "completed"
    result: TaskResult


class DocumentTable(StructuredModel):
    caption: str = ""
    headers: list[str] = Field(min_length=1, max_length=8)
    rows: list[list[str]] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def consistent_columns(self) -> "DocumentTable":
        if any(len(row) != len(self.headers) for row in self.rows):
            raise ValueError("Every table row must match the number of headers.")
        return self


class DocumentSection(StructuredModel):
    heading: str = Field(min_length=1, max_length=200)
    paragraphs: list[str] = Field(default_factory=list, max_length=12)
    bullets: list[str] = Field(default_factory=list, max_length=15)
    table: DocumentTable | None = None

    @model_validator(mode="after")
    def nonempty_section(self) -> "DocumentSection":
        text = self.paragraphs + self.bullets
        if not any(item.strip() for item in text) and self.table is None:
            raise ValueError("A section must contain text, bullets, or a table.")
        return self


class DocumentContent(StructuredModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2500)
    sections: list[DocumentSection] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def distinct_headings(self) -> "DocumentContent":
        headings = [section.heading.casefold() for section in self.sections]
        if len(set(headings)) != len(headings):
            raise ValueError("Document section headings must be distinct.")
        return self


class RequirementCheck(StructuredModel):
    requirement: str = Field(min_length=1, max_length=300)
    met: bool
    note: str = Field(default="", max_length=400)


class ReviewResult(StructuredModel):
    approved: bool
    score: int = Field(ge=1, le=10)
    feedback: list[str] = Field(default_factory=list, max_length=15)
    checklist: list[RequirementCheck] = Field(default_factory=list, max_length=15)

    @model_validator(mode="after")
    def actionable_review(self) -> "ReviewResult":
        if not self.approved and not any(item.strip() for item in self.feedback):
            raise ValueError("A rejected draft requires actionable feedback.")
        return self


class DocumentArtifact(StructuredModel):
    filename: str
    download_url: str
    media_type: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    base64: str | None = None


class TraceInfo(StructuredModel):
    enabled: bool
    project: str | None = None
    run_id: str | None = None
    url: str | None = None


class AgentResponse(StructuredModel):
    run_id: str
    status: Literal["completed", "completed_with_warnings"]
    message: str
    summary: str
    plan: DocumentPlan
    tasks: list[CompletedTask]
    review: ReviewResult
    revisions: int = Field(ge=0)
    warnings: list[str]
    document: DocumentArtifact
    trace: TraceInfo
    elapsed_seconds: float
