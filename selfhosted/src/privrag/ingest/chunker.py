"""Structure-aware chunking.

ParsedDoc -> clean lines -> sections (§ / Artikel / AT ...) -> LangChain
RecursiveCharacterTextSplitter *inside* each section -> Chunks that know their
section and exact page range.

Chunks never cross a section boundary, so a citation's section label is always
correct. Page ranges are computed from character offsets, so a chunk that runs
over a page break reports "pp. 17-18".
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..errors import ChunkError
from ..logging_setup import get_logger
from ..models import Chunk, ParsedDoc
from .cleaning import clean_page_lines, find_boilerplate, normalize_text
from .sections import Heading, detect_heading

log = get_logger("ingest.chunk")

STRONG_KINDS = {"paragraph", "article", "circular"}
_BARE_INT = re.compile(r"^\d{1,3}$")
SEPARATORS = ["\n\n", "\n(", "\n", ". ", "; ", ", ", " ", ""]


@dataclass
class _Section:
    heading: Heading | None
    segments: list[tuple[int, list[str]]] = field(default_factory=list)  # (page, lines)

    def add(self, page: int, line: str) -> None:
        if self.segments and self.segments[-1][0] == page:
            self.segments[-1][1].append(line)
        else:
            self.segments.append((page, [line]))


@dataclass
class ChunkStats:
    sections: int = 0
    chunks: int = 0
    dropped_small: int = 0
    empty_pages: int = 0
    toc_pages: int = 0
    boilerplate_lines: int = 0
    heading_kinds: dict[str, int] = field(default_factory=dict)


def _detect_all(pages_lines: list[tuple[int, list[str]]]) -> list[tuple[int, str, Heading | None]]:
    flat: list[tuple[int, str]] = [(p, l) for p, lines in pages_lines for l in lines]
    out: list[tuple[int, str, Heading | None]] = []
    for i, (p, line) in enumerate(flat):
        nxt = None
        for j in range(i + 1, min(i + 4, len(flat))):
            if flat[j][1].strip():
                nxt = flat[j][1]
                break
        out.append((p, line, detect_heading(line, nxt)))
    return out


def build_sections(parsed: ParsedDoc, header_footer_min_ratio: float, min_text_chars_per_page: int,
                   stats: ChunkStats | None = None) -> list[_Section]:
    stats = stats or ChunkStats()
    raw_pages = [(p.page, p.text.split("\n")) for p in parsed.pages]
    # contents pages laid out as "3.2.2 / Title / 23" lines: many bare page numbers near the start
    early = max(10, int(len(raw_pages) * 0.15))
    raw_toc: set[int] = set()
    for idx, (page, lines) in enumerate(raw_pages[:early]):
        nonempty = [l.strip() for l in lines if l.strip()]
        bare = sum(1 for l in nonempty if _BARE_INT.match(l))
        if bare >= 8 and bare / max(len(nonempty), 1) >= 0.2:
            raw_toc.add(page)
    boiler = find_boilerplate([lines for _, lines in raw_pages], header_footer_min_ratio)
    stats.boilerplate_lines = len(boiler)
    pages_lines: list[tuple[int, list[str]]] = []
    for page, lines in raw_pages:
        cleaned = clean_page_lines(lines, boiler)
        if sum(len(l.strip()) for l in cleaned) < min_text_chars_per_page:
            stats.empty_pages += 1
        pages_lines.append((page, cleaned))

    detected = _detect_all(pages_lines)
    kinds = Counter(h.kind for _, _, h in detected if h)
    stats.heading_kinds = dict(kinds)
    has_strong = sum(kinds[k] for k in STRONG_KINDS) >= 3
    allowed = (STRONG_KINDS | {"guideline"}) if has_strong else {"guideline", "numbered"} | STRONG_KINDS

    # table-of-contents pages: mostly heading lines -> drop the whole page
    toc_pages: set[int] = set()
    per_page: dict[int, list[int]] = {}
    for page, line, h in detected:
        if line.strip():
            c = per_page.setdefault(page, [0, 0])
            c[0] += 1
            c[1] += 1 if (h and h.kind in allowed) else 0
    for page, (n_lines, n_head) in per_page.items():
        if n_head >= 5 and n_head / n_lines >= 0.35:
            toc_pages.add(page)
    toc_pages |= raw_toc
    stats.toc_pages = len(toc_pages)

    # text before the first heading of an EU act are the recitals
    # (only for real EU acts: articles must dominate the headings, not just be mentioned twice)
    n_art = kinds.get("article", 0)
    is_eu_act = n_art >= 2 and n_art >= 0.4 * sum(kinds.values())
    preamble = Heading("Erwägungsgründe", "Erwägungsgründe / Recitals", "preamble") if is_eu_act else None
    sections: list[_Section] = [_Section(heading=preamble)]
    for page, line, h in detected:
        if page in toc_pages:
            continue
        if h and h.kind in allowed:
            sections.append(_Section(heading=h))
        # the heading line itself stays in the text so the chunk reads "§ 10 Allgemeine ..."
        sections[-1].add(page, line)
    stats.sections = sum(1 for s in sections if s.heading)
    return sections


def chunk_document(parsed: ParsedDoc, *, chunk_size: int, chunk_overlap: int, min_chunk_chars: int,
                   header_footer_min_ratio: float = 0.5, min_text_chars_per_page: int = 30,
                   stats: ChunkStats | None = None) -> list[Chunk]:
    stats = stats if stats is not None else ChunkStats()
    meta = parsed.meta
    try:
        sections = build_sections(parsed, header_footer_min_ratio, min_text_chars_per_page, stats)
    except Exception as exc:
        raise ChunkError(f"section detection failed: {exc}", code="CHUNK_SECTIONS_FAILED", doc=meta.doc_id) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS,
        keep_separator=True, add_start_index=True, strip_whitespace=True,
    )
    chunks: list[Chunk] = []
    seq = 0
    for sec in sections:
        # assemble section text, remembering where each page starts
        parts: list[str] = []
        page_starts: list[int] = []
        page_nums: list[int] = []
        offset = 0
        for page, lines in sec.segments:
            seg = normalize_text("\n".join(lines))
            if not seg:
                continue
            if parts:
                parts.append("\n")
                offset += 1
            page_starts.append(offset)
            page_nums.append(page)
            parts.append(seg)
            offset += len(seg)
        text = "".join(parts)
        if not text.strip():
            continue
        if len(text.strip()) < min_chunk_chars:
            stats.dropped_small += 1
            continue
        docs = splitter.create_documents([text])
        for d in docs:
            body = d.page_content.strip()
            if len(body) < min_chunk_chars:
                stats.dropped_small += 1
                continue
            start = int(d.metadata.get("start_index", 0))
            if start < 0:  # splitter could not locate the piece - fall back to search
                start = max(text.find(body[:50]), 0)
            end = start + len(body) - 1
            p_start = page_nums[max(bisect.bisect_right(page_starts, start) - 1, 0)]
            p_end = page_nums[max(bisect.bisect_right(page_starts, end) - 1, 0)]
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(meta.doc_id, seq, body),
                doc_id=meta.doc_id, file=meta.file, title=meta.title,
                regulator=meta.regulator, doc_type=meta.doc_type, language=meta.language,
                publication_date=meta.publication_date,
                section=sec.heading.title if sec.heading else None,
                page_start=p_start, page_end=max(p_end, p_start), seq=seq,
                text=body, char_len=len(body),
            ))
            seq += 1
    stats.chunks = len(chunks)
    if not chunks:
        raise ChunkError("document produced no chunks (no extractable text - scanned PDF needs OCR?)",
                         code="CHUNK_NO_TEXT", doc=meta.doc_id)
    return chunks
