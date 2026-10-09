"""Load and reconcile the document manifest (documents/manifest.csv).

The manifest is the same file used to set blob metadata for Variant A, so both
variants carry identical metadata. Problems are reported, never fatal per row:
a PDF without a manifest row is still ingested with defaults (and a warning).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ManifestError
from ..logging_setup import get_logger
from ..models import DocMeta

log = get_logger("ingest.manifest")

REQUIRED = ["file", "regulator", "doc_type", "title", "publication_date", "language"]
_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


@dataclass
class ManifestResult:
    docs: dict[str, DocMeta]                       # file name -> meta (manifest rows only)
    warnings: list[str] = field(default_factory=list)


def load_manifest(path: Path | None) -> ManifestResult:
    if path is None or not Path(path).exists():
        log.warning("manifest not found - all documents get default metadata", extra={"path": str(path)})
        return ManifestResult(docs={}, warnings=[f"manifest not found: {path}"])

    warnings: list[str] = []
    docs: dict[str, DocMeta] = {}
    try:
        # utf-8-sig strips the BOM Excel adds; fall back to cp1252 for Excel "CSV" saves
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = Path(path).read_text(encoding="cp1252")
            warnings.append("manifest was not UTF-8, read as cp1252")
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
    except Exception as exc:  # unreadable / empty file
        raise ManifestError(f"cannot read manifest: {exc}", code="MANIFEST_UNREADABLE", path=str(path)) from exc

    cols = [c.strip().lower() for c in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED if c not in cols]
    if "file" in missing:
        raise ManifestError("manifest has no 'file' column", code="MANIFEST_NO_FILE_COLUMN", columns=cols)
    if missing:
        warnings.append(f"manifest columns missing (defaults used): {missing}")

    for lineno, raw in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        fname = row.get("file", "")
        if not fname:
            warnings.append(f"line {lineno}: empty file name - skipped")
            continue
        if not fname.lower().endswith(".pdf"):
            warnings.append(f"line {lineno}: {fname} is not a .pdf - skipped")
            continue
        if fname in docs:
            warnings.append(f"line {lineno}: duplicate row for {fname} - first row kept")
            continue
        date = row.get("publication_date") or None
        if date and not _DATE.match(date):
            warnings.append(f"line {lineno}: {fname} bad publication_date '{date}' - ignored")
            date = None
        docs[fname] = DocMeta(
            doc_id=Path(fname).stem,
            file=fname,
            title=row.get("title") or Path(fname).stem,
            regulator=row.get("regulator") or "unknown",
            doc_type=row.get("doc_type") or "unknown",
            publication_date=date,
            language=(row.get("language") or "unknown").lower(),
        )
    for w in warnings:
        log.warning(w)
    log.info("manifest loaded", extra={"count": len(docs), "path": str(path)})
    return ManifestResult(docs=docs, warnings=warnings)


def reconcile(manifest: ManifestResult, pdf_files: list[Path]) -> tuple[list[DocMeta], list[str]]:
    """Match PDFs on disk with manifest rows. Returns (metas in file order, warnings)."""
    warnings: list[str] = []
    on_disk = {p.name: p for p in pdf_files}
    metas: list[DocMeta] = []
    for name in sorted(on_disk):
        meta = manifest.docs.get(name)
        if meta is None:
            meta = DocMeta(doc_id=Path(name).stem, file=name, title=Path(name).stem, from_manifest=False)
            warnings.append(f"{name}: no manifest row - default metadata used")
        metas.append(meta)
    for name in sorted(set(manifest.docs) - set(on_disk)):
        warnings.append(f"{name}: in manifest but file not found")
    for w in warnings:
        log.warning(w)
    return metas, warnings
