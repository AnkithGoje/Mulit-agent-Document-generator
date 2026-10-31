from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from app.schemas import DocumentContent, DocumentPlan, DocumentTable, ReviewResult


NAVY = "17324D"
TEAL = "087E8B"


def _page_field(paragraph, instruction: str) -> None:
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), instruction)
    paragraph._p.append(field)


def _add_table(document, data: DocumentTable) -> None:
    if data.caption:
        caption = document.add_paragraph(data.caption)
        caption.paragraph_format.keep_with_next = True
        caption.runs[0].italic = True

    table = document.add_table(rows=1, cols=len(data.headers))
    table.style = "Light Shading Accent 1"
    table.autofit = True
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    header_properties.append(OxmlElement("w:tblHeader"))

    for cell, text in zip(table.rows[0].cells, data.headers):
        cell.text = text
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), NAVY)
        cell._tc.get_or_add_tcPr().append(shade)
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for values in data.rows:
        row = table.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for cell, value in zip(row.cells, values):
            cell.text = value

    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(4)
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    document.add_paragraph()


def render_docx(
    content: DocumentContent,
    plan: DocumentPlan,
    warnings: list[str] | None = None,
    review: ReviewResult | None = None,
) -> bytes:
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.8)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    for name, size in [("Title", 30), ("Heading 1", 17), ("Heading 2", 13)]:
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(TEAL if name == "Heading 2" else NAVY)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.text = "AUTONOMOUS DOCUMENT AGENT"
    header.runs[0].font.size = Pt(9)
    header.runs[0].font.color.rgb = RGBColor.from_string(TEAL)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Generated draft | Page ")
    _page_field(footer, "PAGE")
    footer.add_run(" of ")
    _page_field(footer, "NUMPAGES")
    for run in footer.runs:
        run.font.size = Pt(9)

    document.core_properties.title = content.title
    document.core_properties.subject = plan.document_type
    document.core_properties.author = "Autonomous Document Agent"
    document.core_properties.keywords = "LangGraph, business document, AI-generated draft"

    document.add_paragraph(plan.document_type.upper(), style="Subtitle")
    document.add_heading(content.title, level=0)
    document.add_paragraph(f"Prepared for: {plan.audience}")
    generated_at = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    document.add_paragraph(f"Generated: {generated_at}")

    if review is not None and not review.approved:
        notice = document.add_paragraph()
        run = notice.add_run("REVIEW REQUIRED: unresolved issues remain in this draft.")
        run.bold = True
        run.font.color.rgb = RGBColor.from_string("A33A21")

    document.add_heading("Executive Summary", level=1)
    document.add_paragraph(content.summary)
    document.add_heading("Contents", level=1)
    for item in content.sections:
        document.add_paragraph(item.heading, style="List Bullet")
    document.add_paragraph("Assumptions and Limitations", style="List Bullet")
    if warnings or (review is not None and not review.approved):
        document.add_paragraph("Review Notes", style="List Bullet")
    document.add_page_break()

    for item in content.sections:
        document.add_heading(item.heading, level=1)
        for text in item.paragraphs:
            if text.strip():
                document.add_paragraph(text)
        for text in item.bullets:
            if text.strip():
                document.add_paragraph(text, style="List Bullet")
        if item.table is not None:
            _add_table(document, item.table)

    document.add_heading("Assumptions and Limitations", level=1)
    for assumption in plan.assumptions:
        document.add_paragraph(assumption, style="List Bullet")
    if not plan.assumptions:
        document.add_paragraph("No additional assumptions were recorded during planning.")
    if plan.conflicts:
        document.add_heading("Conflicts and Resolutions", level=2)
        for conflict in plan.conflicts:
            document.add_paragraph(conflict, style="List Bullet")
    document.add_paragraph(
        "This is an AI-generated draft. Illustrative data and estimates are not verified "
        "facts. Validate figures, approvals, legal requirements, and business decisions "
        "before using this document. No external research was performed."
    )
    if review is not None:
        document.add_paragraph(
            f"Automated self-check: {review.score}/10. "
            "An LLM self-check does not replace professional or factual verification."
        )

    if warnings or (review is not None and not review.approved):
        document.add_heading("Review Notes", level=1)
        for warning in warnings or []:
            document.add_paragraph(warning, style="List Bullet")
        if review is not None and not review.approved:
            for feedback in review.feedback:
                document.add_paragraph(feedback, style="List Bullet")

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def save_docx(data: bytes, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    try:
        temporary_path.write_bytes(data)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def build_docx(
    content: DocumentContent,
    plan: DocumentPlan,
    output_path: Path,
    warnings: list[str] | None = None,
    review: ReviewResult | None = None,
) -> Path:
    return save_docx(render_docx(content, plan, warnings, review), output_path)
