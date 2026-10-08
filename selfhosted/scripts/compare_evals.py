"""Compare evaluation runs side by side (stand-in vs BGE-M3, later Variant A vs B on Azure).

Usage:
    python scripts/compare_evals.py NAME=path/to/eval_x.json NAME2=path/to/eval_y.json [-o report.md]

Prints a markdown table of the summary metrics and a per-question matrix
(expected doc retrieved / cited, section cited, number of citations, status).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [
    ("questions", "Questions"),
    ("errors", "Errors"),
    ("expected_doc_retrieved", "Expected document retrieved"),
    ("expected_section_retrieved", "Expected section retrieved"),
    ("expected_doc_cited", "Expected document cited"),
    ("all_expected_docs_cited", "All expected documents cited"),
    ("expected_section_cited", "Expected section cited"),
    ("citations_ok (>=2)", "Answers with >= 2 citations"),
    ("out_of_scope_refused", "Out-of-scope refused"),
    ("keypoint_proxy_mean", "Key-point proxy (mean)"),
    ("latency_ms_p50", "Latency p50 (ms)"),
    ("latency_ms_p95", "Latency p95 (ms)"),
    ("retrieve_ms_p50", "Retrieve p50 (ms)"),
    ("generate_ms_p50", "Generate p50 (ms)"),
    ("llm_prompt_tokens_mean", "Prompt tokens / question"),
    ("llm_completion_tokens_mean", "Completion tokens / question"),
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def mark(v) -> str:
    return {True: "Y", False: "-", None: " "}.get(v, str(v))


def build(runs: dict[str, dict]) -> str:
    names = list(runs)
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in METRICS:
        out.append(f"| {label} | " + " | ".join(str(runs[n]["summary"].get(key, "")) for n in names) + " |")
    out += ["", "Per question (R = expected doc retrieved, C = expected doc cited, S = expected section cited, "
            "#c = citations):", ""]
    head = "| ID | Type | " + " | ".join(f"{n} R/C/S #c status" for n in names) + " |"
    out += [head, "|---|---|" + "---|" * len(names)]
    by_id = {n: {r["id"]: r for r in runs[n]["rows"]} for n in names}
    for qid in by_id[names[0]]:
        cells = []
        for n in names:
            r = by_id[n].get(qid)
            if not r:
                cells.append("")
                continue
            if r.get("out_of_scope"):
                cells.append(f"OOS refused: {mark(r.get('oos_correct'))} ({r['status']})")
            else:
                cells.append(f"{mark(r.get('doc_retrieved'))}/{mark(r.get('doc_cited'))}/{mark(r.get('section_cited'))} "
                             f"{r.get('n_citations')} {r['status']}")
        out.append(f"| {qid} | {by_id[names[0]][qid].get('type')} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path/to/eval.json")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    runs = {}
    for spec in a.runs:
        name, _, path = spec.partition("=")
        runs[name] = load(path)
    md = build(runs)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
