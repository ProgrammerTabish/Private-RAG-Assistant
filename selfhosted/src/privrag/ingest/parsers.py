"""PDF -> pages of text.

Backends
* ``pymupdf`` - fast, no model download; used locally and as the fallback.
* ``docling`` - layout-aware (tables, reading order); production default on
  Azure. Needs its layout models, which are downloaded from Hugging Face on the
  Azure host (or pre-staged in tenant storage).

Every file is validated *before* parsing (exists, size, PDF magic bytes,
encryption) so a bad upload produces a clear error code instead of a crash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from ..errors import ParseError
from ..logging_setup import get_logger
from ..models import DocMeta, PageText, ParsedDoc

log = get_logger("ingest.parse")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pdf_file(path: Path, max_mb: float) -> None:
    if not path.exists():
        raise ParseError(f"file not found: {path}", code="PARSE_NOT_FOUND", file=path.name)
    size = path.stat().st_size
    if size == 0:
        raise ParseError("file is empty (0 bytes)", code="PARSE_EMPTY_FILE", file=path.name)
    if size > max_mb * 1024 * 1024:
        raise ParseError(f"file too large ({size/1e6:.1f} MB > {max_mb} MB)", code="PARSE_TOO_LARGE", file=path.name)
    with open(path, "rb") as f:
        head = f.read(1024)
    if b"%PDF-" not in head:
        raise ParseError("not a PDF (missing %PDF- header)", code="PARSE_NOT_PDF", file=path.name)


class Parser(Protocol):
    name: str

    def parse(self, path: Path, meta: DocMeta) -> ParsedDoc: ...


class PyMuPDFParser:
    name = "pymupdf"

    def parse(self, path: Path, meta: DocMeta) -> ParsedDoc:
        import pymupdf

        warnings: list[str] = []
        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            raise ParseError(f"PDF is corrupt or unreadable: {exc}", code="PARSE_CORRUPT", file=path.name) from exc
        try:
            if doc.needs_pass:
                # many "protected" PDFs only restrict printing and open with an empty password
                if not doc.authenticate(""):
                    raise ParseError("PDF is password-protected", code="PARSE_ENCRYPTED", file=path.name)
                warnings.append("PDF had an owner password; opened with empty user password")
            if doc.page_count == 0:
                raise ParseError("PDF has no pages", code="PARSE_NO_PAGES", file=path.name)
            pages: list[PageText] = []
            flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
            for i in range(doc.page_count):
                try:
                    text = doc[i].get_text("text", flags=flags, sort=False)
                except Exception as exc:  # one broken page must not kill the document
                    warnings.append(f"page {i+1}: text extraction failed ({type(exc).__name__}: {exc})")
                    log.warning("page extraction failed", extra={"page": i + 1, "error_type": type(exc).__name__})
                    text = ""
                pages.append(PageText(page=i + 1, text=text))
            return ParsedDoc(meta=meta, pages=pages, parser=self.name, warnings=warnings)
        finally:
            doc.close()


class DoclingParser:
    """Docling layout parser. Imported lazily - not installed for local tests."""

    name = "docling"

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError as exc:
            raise ParseError(
                "docling is not installed (pip install 'privrag[docling]')", code="PARSE_BACKEND_MISSING"
            ) from exc
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def parse(self, path: Path, meta: DocMeta) -> ParsedDoc:
        try:
            result = self._converter.convert(str(path))
        except Exception as exc:
            raise ParseError(f"docling conversion failed: {exc}", code="PARSE_DOCLING_FAILED", file=path.name) from exc
        ddoc = result.document
        by_page: dict[int, list[str]] = {}
        # iterate in reading order; tables are exported as markdown so row/column structure survives
        for item, _level in ddoc.iterate_items():
            prov = getattr(item, "prov", None) or []
            page_no = prov[0].page_no if prov else 1
            label = str(getattr(item, "label", ""))
            if "table" in label.lower() and hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=ddoc)
                except TypeError:
                    text = item.export_to_markdown()
            else:
                text = getattr(item, "text", "") or ""
            if text.strip():
                by_page.setdefault(page_no, []).append(text)
        n_pages = max(by_page) if by_page else 0
        try:
            n_pages = max(n_pages, len(ddoc.pages))
        except Exception:
            pass
        if n_pages == 0:
            raise ParseError("docling returned no content", code="PARSE_NO_TEXT", file=path.name)
        pages = [PageText(page=p, text="\n".join(by_page.get(p, []))) for p in range(1, n_pages + 1)]
        return ParsedDoc(meta=meta, pages=pages, parser=self.name)


def get_parser(backend: str) -> Parser:
    if backend == "pymupdf":
        return PyMuPDFParser()
    if backend == "docling":
        return DoclingParser()
    raise ParseError(f"unknown parser backend {backend}", code="PARSE_BACKEND_UNKNOWN")
