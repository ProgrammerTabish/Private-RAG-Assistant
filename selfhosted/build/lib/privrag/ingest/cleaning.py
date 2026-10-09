"""Text clean-up applied to raw PDF page text.

Order matters:
1. strip repeated headers/footers (decided across *all* pages of a document)
2. drop page-number and table-of-contents dot-leader lines
3. section detection runs on these lines (it needs the non-breaking spaces)
4. ``normalize_text`` makes the final chunk text (NBSP->space, soft hyphens, ...)
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_DIGITS = re.compile(r"\d+")
_PAGE_NO = re.compile(
    r"^\s*(?:-\s*\d+\s*-|\d{1,4}|seite\s+\d+(\s+von\s+\d+)?|page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)
_DOT_LEADER = re.compile(r"\.{5,}\s*\d*\s*$|(?:\.\s){5,}")
_PRIVATE_USE = re.compile(r"[-]")       # bullet glyphs from Symbol/Wingdings fonts
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _norm_key(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())


def find_boilerplate(pages: list[list[str]], min_ratio: float = 0.5, edge: int = 4) -> set[str]:
    """Lines (digit-normalised) that appear near the top/bottom of >= min_ratio of pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        nonempty = [l for l in lines if l.strip()]
        candidates = nonempty[:edge] + nonempty[-edge:]
        counts.update({_norm_key(l) for l in candidates if len(l.strip()) > 2})
    threshold = max(3, int(len(pages) * min_ratio))
    return {k for k, c in counts.items() if c >= threshold}


def clean_page_lines(lines: list[str], boilerplate: set[str]) -> list[str]:
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if _norm_key(s) in boilerplate:
            continue
        if _PAGE_NO.match(s):
            continue
        if _DOT_LEADER.search(s):          # table of contents entry
            continue
        out.append(line.rstrip())
    return out


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub(" ", text)
    text = _PRIVATE_USE.sub("•", text)
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    # soft hyphen at a line end joins the word: "Risikomanage\xad\nmentrahmen"
    text = re.sub(r"\xad\s*\n\s*", "", text)
    text = text.replace("\xad", "")
    # hard hyphen line break followed by a lowercase letter: "Gesamtrisi-\nkoprofil"
    # (keep "Aufzeichnungs-\nund" style compounds: next word "und/oder/bzw" keeps the hyphen)
    text = re.sub(r"(\w)-\n(?!und\b|oder\b|bzw\b|sowie\b)([a-zäöüß])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # list markers on their own line ("1.\nDie Identifizierung", "a)\n...") -> same line
    text = re.sub(r"(?m)^(\d{1,3}\.|[a-z]{1,2}\)|\(\d{1,3}[a-z]?\)|[ivx]{1,5}\.)\n(?=\S)", r"\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
