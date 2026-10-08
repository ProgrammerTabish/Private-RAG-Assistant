"""Detect legal section headings so every chunk knows which § / Artikel / AT it
belongs to. This is what makes citations like "01_GwG.pdf, § 10, p. 17"
possible - the evaluation set expects exactly that granularity.

Recognised formats (calibrated on the 50-document corpus):
* German statutes (gesetze-im-internet.de):  "§ 10<NBSP>Allgemeine Sorgfaltspflichten"
  In-text references ("§ 12 Absatz 4") use a normal space and are NOT headings.
* EU regulations / directives:  a line that is only "Artikel 6" / "Article 6",
  the title follows on the next line.
* BaFin circulars (MaRisk, MaComp):  "AT 4.3.3 Stresstests", "BTO 1.2 ...", "BT 3.2"
* Guidelines (EBA/ESMA): "Title II", "Guideline 4", "Leitlinie 4" and short numbered
  headings like "4.1 Risk factors" (no sentence punctuation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PARA = re.compile(r"^(§{1,2})\s*(\d+[a-z]{0,2})(?:\xa0|\s{2,})(\S.{0,160})$")
_PARA_LOOSE = re.compile(r"^(§{1,2})\s*(\d+[a-z]{0,2}) (\S.{0,160})$")
_PARA_ALONE =re.compile(r"^(§{1,2})\s*\xa0?\s*(\d+[a-z]{0,2})\s*$")
_ARTICLE = re.compile(r"^(Artikel|Article|Art\.)\s+(\d+[a-z]{0,2})\s*$", re.IGNORECASE)
_BAFIN = re.compile(r"^(AT|BT|BTO|BTR|BTO|BTI)\s?(\d+(?:\.\d+){0,4})(?:\s+(\S.{0,120}))?$")
_GUIDELINE = re.compile(r"^(Title|Titel|Guideline|Leitlinie|Chapter|Kapitel|Abschnitt|Section)\s+([IVXLC]+|\d+)\b[:.\s-]*(.{0,120})$", re.IGNORECASE)
_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-ZÄÖÜ][^.;:!?]{2,80})$")
# "\b" does not work after "Abs." / "Nr." (no word boundary between "." and a space) -> lookahead
_REFERENCE_WORDS = re.compile(
    r"^(Absatz\w*|Abs\.|Satz\w*|S\.|Nummer\w*|Nr\.|Buchstabe\w*|Buchst\.|Unterabsatz\w*|UAbs\.|der|des|dem|und|oder|bis|ff\.|paragraph|para\.|of|the)(?=\s|$|[,;)])",
    re.IGNORECASE)


@dataclass
class Heading:
    label: str      # short id used in citations, e.g. "§ 10", "Artikel 19", "AT 4.4.2"
    title: str      # full heading text, e.g. "§ 10 Allgemeine Sorgfaltspflichten"
    kind: str


def _short(s: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def detect_heading(line: str, next_line: str | None = None) -> Heading | None:
    """Return a Heading if ``line`` starts a new section."""
    raw = line.rstrip()
    s = raw.strip()
    if not s or len(s) > 180:
        return None

    m = _PARA.match(s)
    if m and not _REFERENCE_WORDS.match(m.group(3)):
        label = f"{m.group(1)} {m.group(2)}"
        return Heading(label, _short(f"{label} {m.group(3)}"), "paragraph")

    # same heading after a converter turned the NBSP into a normal space: stricter checks
    m = _PARA_LOOSE.match(s)
    if m:
        title = m.group(3)
        first = title.split()[0]
        is_abbrev = len(first) <= 6 and sum(ch.isupper() for ch in first) >= 2   # "KWG", "WpHG", "GwG"
        if (title[:1].isupper() and not is_abbrev and not _REFERENCE_WORDS.match(title)
                and not s.endswith((".", ",", ";", ":")) and len(s) <= 140):
            label = f"{m.group(1)} {m.group(2)}"
            return Heading(label, _short(f"{label} {title}"), "paragraph")

    m = _PARA_ALONE.match(s)
    if m:
        label = f"{m.group(1)} {m.group(2)}"
        title = label
        if next_line and next_line.strip() and len(next_line.strip()) < 120 and not next_line.strip().startswith("("):
            title = f"{label} {next_line.strip()}"
        return Heading(label, _short(title), "paragraph")

    m = _ARTICLE.match(s)
    if m:
        word = "Artikel" if m.group(1).lower().startswith(("artikel", "art.")) else "Article"
        label = f"{word} {m.group(2)}"
        title = label
        if next_line and next_line.strip() and len(next_line.strip()) < 140 and not next_line.strip().startswith("("):
            title = f"{label} {next_line.strip()}"
        return Heading(label, _short(title), "article")

    m = _BAFIN.match(s)
    if m and (m.group(3) is None or not m.group(3)[:1].islower()):
        label = f"{m.group(1)} {m.group(2)}"
        title = f"{label} {m.group(3)}" if m.group(3) else label
        return Heading(label, _short(title), "circular")

    m = _GUIDELINE.match(s)
    if m and not s.endswith((".", ",", ";")) and (m.group(3) == "" or m.group(3)[:1].isupper() or m.group(3)[:1].isdigit()):
        label = f"{m.group(1).title()} {m.group(2)}"
        title = f"{label} {m.group(3)}".strip()
        return Heading(label, _short(title), "guideline")

    m = _NUMBERED.match(s)
    if m and len(s) <= 90 and not s.endswith((".", ",", ";", ":")):
        # a numbered heading is short, Title-like and not a wrapped paragraph line
        words = m.group(2).split()
        if 1 <= len(words) <= 12:
            return Heading(m.group(1), _short(s), "numbered")
    return None
