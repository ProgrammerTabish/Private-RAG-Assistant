"""Turn raw LLM text into a validated, cited answer.

* parses ``[1]``, ``[2][3]``, ``[1, 4]`` citation markers
* drops markers that point to sources that were never shown (hallucinated
  citation numbers) and warns
* renumbers citations 1..k in order of first use so the UI list matches
* picks the best-matching sentence of each cited chunk as a highlight quote
* computes a heuristic confidence score (see ``confidence``)
"""
from __future__ import annotations

import math
import re

from ..index.sparse import tokenize
from ..models import Citation, RetrievedChunk

CITE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def parse_citations(text: str) -> list[int]:
    out: list[int] = []
    for m in CITE.finditer(text):
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            if n.isdigit() and int(n) not in out:
                out.append(int(n))
    return out


def clean_and_renumber(text: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """Return (text with valid renumbered markers, original source numbers in new order, invalid numbers)."""
    order: list[int] = []
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        nums = []
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            k = int(n)
            if 1 <= k <= n_sources:
                if k not in order:
                    order.append(k)
                nums.append(order.index(k) + 1)
            elif k not in invalid:
                invalid.append(k)
        return "".join(f"[{x}]" for x in dict.fromkeys(nums))

    new = CITE.sub(repl, text)
    new = re.sub(r"\s+([.,;])", r"\1", new).strip()
    return new, order, invalid


def best_quote(chunk_text: str, answer: str, max_len: int = 280) -> str:
    a = set(tokenize(answer))
    best, best_score = "", -1
    for sent in _SENT.split(chunk_text.replace("\n", " ")):
        if len(sent) < 20:
            continue
        sc = len(a & set(tokenize(sent)))
        if sc > best_score:
            best, best_score = sent, sc
    best = best or chunk_text[:max_len]
    return best if len(best) <= max_len else best[: max_len - 1] + "…"


def build_citations(order: list[int], chunks: list[RetrievedChunk], answer: str) -> list[Citation]:
    cits = []
    for new_n, old_n in enumerate(order, 1):
        c = chunks[old_n - 1].chunk
        cits.append(Citation(n=new_n, chunk_id=c.chunk_id, doc_id=c.doc_id, file=c.file, title=c.title,
                             section=c.section, page_start=c.page_start, page_end=c.page_end,
                             quote=best_quote(c.text, answer)))
    return cits


def citation_coverage(answer: str) -> float:
    sents = [s for s in _SENT.split(answer) if len(s.strip()) > 15]
    if not sents:
        return 0.0
    return sum(1 for s in sents if CITE.search(s)) / len(sents)


def confidence(chunks: list[RetrievedChunk], cited_old_numbers: list[int], answer: str, min_citations: int) -> float:
    """Heuristic 0..1 score shown in the UI.

    40% retrieval strength - reranker score if available, otherwise agreement of
        dense and keyword search on the cited chunks (both found it = strong)
    35% citation coverage - share of answer sentences that carry a citation
    25% source support - distinct cited sources relative to the required minimum
    """
    if not cited_old_numbers:
        return 0.0
    cited = [chunks[i - 1] for i in cited_old_numbers if 1 <= i <= len(chunks)]
    if cited and all(c.rerank_score is not None for c in cited):
        strength = sum(1 / (1 + math.exp(-c.rerank_score)) for c in cited) / len(cited)
    else:
        strength = sum((c.dense_rank is not None) + (c.sparse_rank is not None) for c in cited) / (2 * len(cited))
        top_bonus = sum(1 for i in cited_old_numbers if i <= 3) / len(cited_old_numbers)
        strength = 0.7 * strength + 0.3 * top_bonus
    coverage = citation_coverage(answer)
    support = min(len(set(cited_old_numbers)) / max(min_citations, 1), 1.0)
    return round(max(0.0, min(1.0, 0.40 * strength + 0.35 * coverage + 0.25 * support)), 3)


def confidence_label(score: float) -> str:
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
