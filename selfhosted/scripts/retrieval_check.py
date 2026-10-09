"""Measure retrieval only (no LLM): is the gold section among the top-k sources?

Usage (with the PRIVRAG_* settings of the index to test):
    python scripts/retrieval_check.py eval/hinschg_retrieval_bilingual.json [--k 4] [--german-query-file f.json]

Prints hit@k per language and per question, and the top sections for misses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from privrag.config import get_settings
from privrag.logging_setup import setup_logging
from privrag.retrieve.retriever import Retriever


def sec_label(section: str | None) -> str:
    if not section:
        return "-"
    parts = section.split()
    return " ".join(parts[:2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--german-queries", help="json {question: german_query} to simulate the LLM rewrite")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_dir, console=False)
    k = a.k or s.top_k
    items = json.loads(Path(a.file).read_text(encoding="utf-8"))["items"]
    rewrites = json.loads(Path(a.german_queries).read_text(encoding="utf-8")) if a.german_queries else {}
    r = Retriever(s)
    hits = {"de": 0, "en": 0}
    rows = []
    for it in items:
        for lang in ("de", "en"):
            q = it[lang]
            res, dbg = r.retrieve(q, german_query=rewrites.get(q), top_k=k)
            got = [sec_label(x.chunk.section) for x in res]
            ok = any(g in got for g in it["gold"])
            hits[lang] += ok
            rows.append((it["id"], lang, ok, got, dbg.get("german_query")))
    n = len(items)
    print(f"hit@{k}: DE {hits['de']}/{n}   EN {hits['en']}/{n}   total {hits['de'] + hits['en']}/{2 * n}")
    if not a.quiet:
        for qid, lang, ok, got, gq in rows:
            print(f"  {qid} {lang} {'OK  ' if ok else 'MISS'} {got}" + (f"   [de query: {gq}]" if (gq and not ok) else ""))


if __name__ == "__main__":
    main()
