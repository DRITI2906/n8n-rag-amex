"""
Document parser: converts raw bytes → cleaned text + metadata.
Supports PDF, DOCX, PPTX, Markdown, plain text, HTML (email/Notion).
"""
from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime
from typing import Any

from langdetect import detect, LangDetectException


# ─── Public interface ────────────────────────────────────────────────────────

class ParsedDocument:
    def __init__(
        self,
        text: str,
        title: str | None,
        author: str | None,
        language: str,
        creation_date: datetime | None,
        modified_date: datetime | None,
        mime_type: str,
        file_hash: str,
        raw_size: int,
        metadata: dict[str, Any],
    ):
        self.text = text
        self.title = title
        self.author = author
        self.language = language
        self.creation_date = creation_date
        self.modified_date = modified_date
        self.mime_type = mime_type
        self.file_hash = file_hash
        self.raw_size = raw_size
        self.metadata = metadata


def parse_document(content: bytes, mime_type: str, filename: str = "") -> ParsedDocument:
    """Entry point: dispatch to the right parser by MIME type."""
    file_hash = hashlib.sha256(content).hexdigest()
    raw_size = len(content)

    if mime_type == "application/pdf":
        result = _parse_pdf(content)
    elif mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        result = _parse_docx(content)
    elif mime_type in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ):
        result = _parse_pptx(content)
    elif mime_type in ("text/html", "application/xhtml+xml"):
        result = _parse_html(content)
    elif mime_type in ("text/markdown", "text/x-markdown"):
        result = _parse_markdown(content)
    else:
        # Fallback: plain text
        result = {"text": content.decode("utf-8", errors="replace"), "metadata": {}}

    text = _clean_text(result["text"])
    language = _detect_language(text)
    title = result.get("title") or _infer_title(text, filename)

    return ParsedDocument(
        text=text,
        title=title,
        author=result.get("author"),
        language=language,
        creation_date=result.get("creation_date"),
        modified_date=result.get("modified_date"),
        mime_type=mime_type,
        file_hash=file_hash,
        raw_size=raw_size,
        metadata=result.get("metadata", {}),
    )


# ─── PDF ─────────────────────────────────────────────────────────────────────

def _parse_pdf(content: bytes) -> dict:
    import pdfplumber

    pages_text: list[str] = []
    headings: list[str] = []
    metadata: dict = {}

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if pdf.metadata:
            metadata = dict(pdf.metadata)

        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

            # Extract tables as markdown-style text
            for table in page.extract_tables():
                if not table:
                    continue
                rows = [" | ".join(str(c) if c else "" for c in row) for row in table]
                text += "\n\n" + "\n".join(rows)

            pages_text.append(text)

            # Heuristic: lines in ALL-CAPS or larger font size are headings
            for word in (page.chars or []):
                if word.get("size", 0) > 14 and word.get("text", "").strip():
                    headings.append(word["text"].strip())

    return {
        "text": "\n\n".join(pages_text),
        "title": metadata.get("Title") or metadata.get("title"),
        "author": metadata.get("Author") or metadata.get("author"),
        "creation_date": _parse_pdf_date(metadata.get("CreationDate")),
        "modified_date": _parse_pdf_date(metadata.get("ModDate")),
        "metadata": {"headings": list(dict.fromkeys(headings))},
    }


def _parse_pdf_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        raw = raw.replace("D:", "").replace("'", "").replace("Z", "+00:00")
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
    except Exception:
        return None


# ─── DOCX ────────────────────────────────────────────────────────────────────

def _parse_docx(content: bytes) -> dict:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(io.BytesIO(content))
    parts: list[str] = []
    headings: list[str] = []

    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name.startswith("Heading"):
            headings.append(para.text)
            parts.append(f"\n## {para.text}\n")
        else:
            parts.append(para.text)

    # Tables
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))

    core = doc.core_properties
    return {
        "text": "\n".join(parts),
        "title": core.title,
        "author": core.author,
        "creation_date": core.created,
        "modified_date": core.modified,
        "metadata": {"headings": headings},
    }


# ─── PPTX ────────────────────────────────────────────────────────────────────

def _parse_pptx(content: bytes) -> dict:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(content))
    slides_text: list[str] = []

    for i, slide in enumerate(prs.slides, 1):
        slide_parts = [f"## Slide {i}"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        slide_parts.append(text)
        slides_text.append("\n".join(slide_parts))

    core = prs.core_properties
    return {
        "text": "\n\n".join(slides_text),
        "title": core.title,
        "author": core.author,
        "creation_date": core.created,
        "modified_date": core.modified,
        "metadata": {"slide_count": len(prs.slides)},
    }


# ─── HTML / Email ────────────────────────────────────────────────────────────

def _parse_html(content: bytes) -> dict:
    from bs4 import BeautifulSoup
    import markdownify

    soup = BeautifulSoup(content, "lxml")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    md = markdownify.markdownify(str(soup.body or soup), heading_style="ATX")
    return {"text": md, "title": title, "metadata": {}}


# ─── Markdown ────────────────────────────────────────────────────────────────

def _parse_markdown(content: bytes) -> dict:
    text = content.decode("utf-8", errors="replace")
    # Extract title from first H1
    first_h1 = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return {
        "text": text,
        "title": first_h1.group(1).strip() if first_h1 else None,
        "metadata": {},
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _detect_language(text: str) -> str:
    try:
        sample = text[:1000]
        return detect(sample)
    except LangDetectException:
        return "en"


def _infer_title(text: str, filename: str) -> str | None:
    first_line = text.strip().split("\n")[0].strip()
    if first_line and len(first_line) < 200:
        return first_line
    if filename:
        return filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    return None
