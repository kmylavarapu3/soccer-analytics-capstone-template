"""
Render Reports/FINAL_REPORT_SOURCE.md into PDF using explicit FPDF layout.

This avoids fpdf.write_html (weak styling, inconsistent fonts) and matches a
clean academic report: off-white page, navy headings, body text in blue-gray
(not pure black) for readability.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "Reports"
SOURCE_PATH = REPORTS_DIR / "FINAL_REPORT_SOURCE.md"
PRIMARY_OUTPUT_PATH = REPORTS_DIR / "Final_Report_kmylavarapu3.pdf"
OUTPUT_PATH = REPORTS_DIR / "Final_Report_kmylavarapu3_refreshed.pdf"

APPENDIX_FIGURES = [
    ("Model Comparison", ROOT_DIR / "eda" / "output" / "05_model_comparison.png"),
    ("Temporal Stability", ROOT_DIR / "eda" / "output" / "06_temporal_stability.png"),
    ("Upset Tree Importance", ROOT_DIR / "eda" / "output" / "07_upset_tree_importance.png"),
]

# Typography — avoid pure black (#000) on white per readability preference
COLOR_TITLE = (15, 38, 92)
COLOR_H2 = (18, 45, 95)
COLOR_H3 = (35, 62, 110)
COLOR_BODY = (45, 55, 70)
COLOR_MUTED = (95, 105, 120)
COLOR_TABLE_GRID = (190, 198, 210)


def normalize_text(text: str) -> str:
    """Light cleanup; strip markdown emphasis to plain text for PDF."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2022": "-",
        "\u00a0": " ",
        "\u2192": "->",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
        "\u03c3": "sigma",
        "\u00b2": "^2",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="Letter")
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(18, 18, 18)
        self.alias_nb_pages()
        self.title_text = "Delivering Elite European Football Analytics"

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 6, self.title_text, 0, 1, "R")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    @property
    def epw(self) -> float:
        return self.w - self.l_margin - self.r_margin


def render_cover(pdf: ReportPDF, lines: list[str], start_idx: int) -> int:
    """Title and subtitle block until --- or ## section."""
    pdf.add_page()
    idx = start_idx
    if idx >= len(lines) or not lines[idx].startswith("# "):
        return idx

    title = normalize_text(lines[idx].replace("# ", "", 1))
    pdf.set_text_color(*COLOR_TITLE)
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(16)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 9, title, align="C")
    pdf.ln(3)
    idx += 1

    while idx < len(lines):
        line = lines[idx].rstrip()
        if line.strip() == "---":
            idx += 1
            break
        if line.startswith("## "):
            break
        if not line.strip():
            idx += 1
            continue
        text = normalize_text(line)
        if text:
            pdf.set_font("Helvetica", "", 11.5)
            pdf.set_text_color(*COLOR_BODY)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6.2, text, align="C")
        idx += 1

    pdf.ln(6)
    return idx


def render_heading(pdf: ReportPDF, line: str) -> None:
    line = line.rstrip()
    if line.startswith("## "):
        pdf.ln(3)
        pdf.set_text_color(*COLOR_H2)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7.5, normalize_text(line[3:]))
        pdf.ln(1)
        pdf.set_draw_color(*COLOR_TABLE_GRID)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2)
    elif line.startswith("### "):
        pdf.ln(2)
        pdf.set_text_color(*COLOR_H3)
        pdf.set_font("Helvetica", "B", 12.5)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6.8, normalize_text(line[4:]))
        pdf.ln(1)


def render_paragraph(pdf: ReportPDF, paragraph_lines: list[str]) -> None:
    if not paragraph_lines:
        return
    text = normalize_text(" ".join(line.strip() for line in paragraph_lines))
    if not text:
        return
    pdf.set_text_color(*COLOR_BODY)
    pdf.set_font("Times", "", 11.2)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6.1, text)
    pdf.ln(1.5)


def render_list_item(pdf: ReportPDF, line: str, bullet: bool) -> None:
    clean = normalize_text(line)
    if bullet and clean.startswith("- "):
        body = clean[2:].strip()
        prefix = "- "
    elif not bullet:
        match = re.match(r"^(\d+\.\s+)(.*)$", clean)
        prefix, body = (match.group(1), match.group(2)) if match else ("", clean)
    else:
        body = clean
        prefix = "- "

    pdf.set_text_color(*COLOR_BODY)
    pdf.set_font("Times", "", 11.2)
    indent = pdf.l_margin + 6
    pdf.set_x(indent)
    pdf.multi_cell(pdf.epw - 6, 6.0, prefix + body)
    pdf.ln(0.8)


def render_table(pdf: ReportPDF, table_lines: list[str]) -> None:
    rows: list[list[str]] = []
    for raw_line in table_lines:
        stripped = raw_line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [normalize_text(cell.strip()) for cell in stripped.strip("|").split("|")]
        if cells and all(set(cell) <= {":", "-"} for cell in cells):
            continue
        rows.append(cells)

    if not rows:
        return

    col_count = max(len(row) for row in rows)
    col_width = pdf.epw / col_count
    line_height = 6.8

    pdf.set_draw_color(*COLOR_TABLE_GRID)
    pdf.set_text_color(*COLOR_BODY)
    for row_idx, row in enumerate(rows):
        pdf.set_font("Helvetica", "B" if row_idx == 0 else "", 10.2)
        if row_idx == 0:
            pdf.set_fill_color(235, 239, 248)
        else:
            pdf.set_fill_color(252, 252, 254)
        for col_idx in range(col_count):
            text = row[col_idx] if col_idx < len(row) else ""
            pdf.cell(col_width, line_height, text[:52], border=1, align="L", fill=True)
        pdf.ln(line_height)
    pdf.ln(2)
    pdf.set_font("Times", "", 11.2)


def render_appendix_figures(pdf: ReportPDF) -> None:
    existing = [(title, path) for title, path in APPENDIX_FIGURES if path.exists()]
    if not existing:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*COLOR_H2)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, "Appendix: Generated Figures")
    pdf.ln(3)

    for title, image_path in existing:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*COLOR_H3)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, title)
        pdf.ln(1)

        usable_width = pdf.epw
        image_height = usable_width * 0.58
        if pdf.get_y() + image_height > pdf.h - pdf.b_margin - 12:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, title)
            pdf.ln(1)

        pdf.image(str(image_path), x=pdf.l_margin, y=pdf.get_y(), w=usable_width)
        pdf.ln(image_height + 6)


def build_pdf(source_path: Path, output_path: Path) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    pdf = ReportPDF()

    idx = 0
    idx = render_cover(pdf, lines, idx)

    paragraph_buf: list[str] = []
    table_buf: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_buf
        if paragraph_buf:
            render_paragraph(pdf, paragraph_buf)
        paragraph_buf = []

    while idx < len(lines):
        raw = lines[idx]
        line = raw.rstrip("\n")

        if line.strip() == "---":
            flush_paragraph()
            idx += 1
            continue

        if line.strip().startswith("|"):
            flush_paragraph()
            table_buf.append(line)
            idx += 1
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_buf.append(lines[idx])
                idx += 1
            render_table(pdf, table_buf)
            table_buf = []
            continue

        if line.startswith("## ") or line.startswith("### "):
            flush_paragraph()
            render_heading(pdf, line)
            idx += 1
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            flush_paragraph()
            render_list_item(pdf, stripped, bullet=True)
            idx += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            render_list_item(pdf, stripped, bullet=False)
            idx += 1
            continue

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        paragraph_buf.append(line)
        idx += 1

    flush_paragraph()
    render_appendix_figures(pdf)
    pdf.output(str(output_path))


if __name__ == "__main__":
    build_pdf(SOURCE_PATH, PRIMARY_OUTPUT_PATH)
    build_pdf(SOURCE_PATH, OUTPUT_PATH)
    print(PRIMARY_OUTPUT_PATH)
    print(OUTPUT_PATH)
