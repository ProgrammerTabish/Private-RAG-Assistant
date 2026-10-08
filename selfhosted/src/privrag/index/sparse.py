"""BM25 sparse vectors for Qdrant (keyword half of hybrid search).

Document vectors hold the BM25 term-frequency part; Qdrant multiplies in IDF
itself (``Modifier.IDF``), so new documents never require re-weighting old ones.
Token ids are a stable CRC32 hash, so no vocabulary file has to be kept in sync.

Exact legal references matter in this domain, so "§ 25h", "Art. 5b",
"AT 4.4.2" and "2022/2554" are kept as single tokens.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

STOP = set("""
der die das den dem des ein eine einer eines einem einen und oder aber auch als an auf aus bei bis durch für gegen
im in ist mit nach nicht noch nur ob so sowie über um unter von vor wie wird werden wurde zu zum zur sind sein
hat haben kann können muss müssen soll sollen dass da es er sie wir ihr sich dieser diese dieses jeder jede
the a an and or of to in on for by with is are be as at from that this it its which shall may must not
""".split())

_REF = re.compile(
    r"§{1,2}\s*\d+[a-z]{0,2}"                 # § 25h
    r"|\b(?:art\.?|artikel|article)\s*\d+[a-z]{0,2}"  # Art. 5b
    r"|\b(?:at|bt|bto|btr)\s?\d+(?:\.\d+)+"  # AT 4.4.2
    r"|\b\d{2,4}/\d{2,5}\b",                  # 2022/2554
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-zäöüß0-9]+(?:-[a-zäöüß0-9]+)*", re.IGNORECASE)

K1 = 1.2
B = 0.75


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().replace("\xa0", " ")


def _light_stem(tok: str) -> str:
    # very light German/English plural/inflection folding; keeps legal ids intact
    if tok.isdigit() or len(tok) <= 4:
        return tok
    for suf in ("ungen", "en", "er", "es", "e", "s", "n"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    t = _norm(text)
    toks = [re.sub(r"\s+", "", m.group(0)).replace("§§", "§").replace("artikel", "art").replace("article", "art")
            .replace("art.", "art") for m in _REF.finditer(t)]
    for m in _WORD.finditer(t):
        w = m.group(0)
        parts = [w] + (w.split("-") if "-" in w else [])
        for p in parts:
            if len(p) < 2 or p in STOP:
                continue
            toks.append(_light_stem(p))
    return toks


def token_id(tok: str) -> int:
    return zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF


def doc_vector(text: str, avg_len: float) -> tuple[list[int], list[float]]:
    toks = tokenize(text)
    if not toks:
        return [], []
    tf = Counter(token_id(t) for t in toks)
    dl = len(toks)
    norm = K1 * (1 - B + B * dl / max(avg_len, 1.0))
    idx = sorted(tf)
    return idx, [round(tf[i] * (K1 + 1) / (tf[i] + norm), 5) for i in idx]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    ids = sorted({token_id(t) for t in tokenize(text)})
    return ids, [1.0] * len(ids)


def average_length(texts: list[str]) -> float:
    if not texts:
        return 1.0
    return sum(len(tokenize(t)) for t in texts) / len(texts)
