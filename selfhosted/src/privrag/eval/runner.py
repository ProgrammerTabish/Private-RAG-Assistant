"""Evaluation on the UC13 question set (documents/UC13_Evaluation_Question_Set_Students.xlsx).

Automatic metrics (no human needed):
* doc_retrieved  - an expected source document is among the retrieved chunks
* doc_cited      - an expected source document is cited in the answer
* section_cited  - the expected § / Artikel / AT section is cited (where the set names one)
* citations_ok   - KPI: >= 2 citations
* oos_correct    - out-of-scope questions are refused
* keypoint_proxy - share of numbers/terms from the expected key points found in the answer
                   (a rough proxy only; real accuracy is graded by the coaches)
* latency p50/p95 and cost per 1k queries (from the configured GPU hour price)

Outputs data/reports/eval_<run_id>.json and .xlsx (the .xlsx has empty
"Score (1/0.5/0)" and "Sources OK (Y/N)" columns for the coaches).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from ..config import Settings
from ..errors import PrivRagError
from ..generate.service import RagService, safe_ask
from ..logging_setup import current_run_id, get_logger, stage, step
from ..retrieve.retriever import Retriever

log = get_logger("eval")

_DOC = re.compile(r"(\d{2}_[A-Za-z0-9_]+)\.pdf")
_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "twenty-four": "24", "seventy-two": "72"}


def parse_expected(src: str) -> list[dict]:
    """'01_GwG.pdf, section 10(3); 10_X.pdf' -> [{'doc': '01_GwG', 'sections': ['§ 10']}, {'doc': '10_X', ...}]"""
    out = []
    for part in str(src or "").split(";"):
        m = _DOC.search(part)
        if not m:
            continue
        doc = m.group(1)
        rest = part[m.end():]
        secs: list[str] = []
        for at in re.findall(r"\b(AT|BT|BTO|BTR)\s?(\d+(?:\.\d+)*)", rest):
            secs.append(f"{at[0]} {at[1]}")
        sm = re.search(r"\bsections?\s+(.+)", rest, re.I)
        if sm:
            secs += [f"§ {n}" for n in re.findall(r"(\d+[a-z]?)(?:\(\d+\))?", sm.group(1))]
        am = re.search(r"\barticles?\s+(.+)", rest, re.I)
        if am:
            secs += [f"Artikel {n}" for n in re.findall(r"(\d+[a-z]?)(?:\(\d+\))?", am.group(1))]
        out.append({"doc": doc, "sections": list(dict.fromkeys(secs))})
    return out


def _section_match(label: str | None, expected: str) -> bool:
    if not label:
        return False
    return label == expected or label.startswith(expected + " ")


def keypoint_proxy(expected: str, answer: str) -> float | None:
    exp = str(expected or "").lower()
    for w, d in _NUM_WORDS.items():
        exp = re.sub(rf"\b{w}\b", d, exp)
    exp = re.sub(r"(?:^|\s)\d{1,2}\)\s", " ", exp)            # drop list markers "1) 2) ..."
    nums = set(re.findall(r"\d+(?:[.,]\d+)*", exp))
    if not nums:
        return None
    ans = answer.lower().replace(".", "").replace(" ", "")
    hits = sum(1 for n in nums if n.replace(",", "").replace(".", "") in ans)
    return round(hits / len(nums), 2)


def load_questions(path: Path) -> list[dict]:
    import pandas as pd

    if not path.exists():
        raise PrivRagError(f"evaluation file not found: {path}", code="EVAL_FILE_MISSING", stage="eval")
    df = pd.read_excel(path, sheet_name="Question set")
    need = {"ID", "Question", "Type", "Expected source(s)", "Expected answer (key points)"}
    if missing := need - set(df.columns):
        raise PrivRagError(f"question sheet lacks columns {missing}", code="EVAL_BAD_FORMAT", stage="eval")
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


def run_eval(s: Settings, limit: int | None = None, use_glossary: bool = True, service: RagService | None = None) -> dict:
    s.ensure_dirs()
    with stage("eval"):
        questions = load_questions(s.eval_path)
        if limit:
            questions = questions[:limit]
        svc = service or RagService(s, retriever=Retriever(s, use_glossary=use_glossary))
        rows: list[dict] = []
        t_all = time.perf_counter()
        for q in questions:
            qid = q["ID"]
            exp = parse_expected(q.get("Expected source(s)"))
            is_oos = str(q.get("Type", "")).lower().startswith("out of scope")
            with step(f"question {qid}", log) as r:
                a = safe_ask(svc, q["Question"], request_id=f"eval-{current_run_id()[-6:]}-{qid}")
                exp_docs = {e["doc"] for e in exp}
                cited_docs = {c.doc_id for c in a.citations}
                ret = [tuple(x.split(" | ")[:2]) for x in a.retrieved_refs]
                ret_docs = {d for d, _ in ret}
                exp_secs = [(e["doc"], sec) for e in exp for sec in e["sections"]]
                row = {
                    "id": qid, "topic": q.get("Topic"), "type": q.get("Type"), "difficulty": q.get("Difficulty (1-3)"),
                    "question": q["Question"], "status": a.status, "answer": a.answer,
                    "citations": [f"{c.file} | {c.section or '-'} | p. {c.page_start}-{c.page_end}" for c in a.citations],
                    "n_citations": len(a.citations), "confidence": a.confidence,
                    "latency_ms": a.latency_ms.get("total"), "warnings": a.warnings,
                    "expected_sources": q.get("Expected source(s)"), "expected_answer": q.get("Expected answer (key points)"),
                    "out_of_scope": is_oos,
                }
                if is_oos:
                    row["oos_correct"] = a.status in ("out_of_scope", "insufficient_sources")
                else:
                    row["citations_ok"] = len(a.citations) >= s.min_citations
                    row["doc_retrieved"] = bool(exp_docs & ret_docs) if exp_docs else None
                    row["section_retrieved"] = (any(d2 == d and _section_match(sec2 if sec2 != "-" else None, sec)
                                                    for d, sec in exp_secs for d2, sec2 in ret) if exp_secs else None)
                    row["doc_cited"] = bool(exp_docs & cited_docs) if exp_docs else None
                    row["all_docs_cited"] = exp_docs <= cited_docs if exp_docs else None
                    row["section_cited"] = (any(c.doc_id == d and _section_match(c.section, sec)
                                                for d, sec in exp_secs for c in a.citations) if exp_secs else None)
                    row["keypoint_proxy"] = keypoint_proxy(q.get("Expected answer (key points)"), a.answer)
                rows.append(row)
                r.update(status=a.status, citations=len(a.citations), doc_cited=row.get("doc_cited"),
                         section_cited=row.get("section_cited"))
        wall = time.perf_counter() - t_all

        def rate(key: str) -> str | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else None

        lats = [r["latency_ms"] for r in rows if r["latency_ms"]]
        kp = [r["keypoint_proxy"] for r in rows if r.get("keypoint_proxy") is not None]
        summary = {
            "run_id": current_run_id(), "questions": len(rows),
            "backends": {"parser": s.parser_backend, "embed": s.embed_backend, "rerank": s.rerank_backend,
                         "llm": s.llm_backend, "llm_model": s.llm_model, "glossary": use_glossary},
            "answered": sum(r["status"] == "answered" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "citations_ok (>=2)": rate("citations_ok"),
            "expected_doc_retrieved": rate("doc_retrieved"),
            "expected_section_retrieved": rate("section_retrieved"),
            "expected_doc_cited": rate("doc_cited"),
            "all_expected_docs_cited": rate("all_docs_cited"),
            "expected_section_cited": rate("section_cited"),
            "out_of_scope_refused": rate("oos_correct"),
            "keypoint_proxy_mean": round(statistics.mean(kp), 2) if kp else None,
            "latency_ms_p50": round(statistics.median(lats), 1) if lats else None,
            "latency_ms_p95": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 1) if lats else None,
            "wall_s": round(wall, 1),
            "cost_per_1k_queries_eur": round(s.gpu_hour_cost * (wall / 3600) / max(len(rows), 1) * 1000, 2),
            "note": ("Local run with stand-in models (LSA embeddings, extractive fake LLM): measures the pipeline, "
                     "not final answer quality. Accuracy >= 85% is graded by coaches on the Azure run "
                     "(BGE-M3 + Llama 3.3 via vLLM).") if s.llm_backend == "fake" or s.embed_backend == "lsa" else "",
        }
        base = s.reports_dir / f"eval_{current_run_id()}"
        base.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False,
                                                        default=str), encoding="utf-8")
        _write_xlsx(base.with_suffix(".xlsx"), rows, summary)
        log.info("eval finished", extra={k: v for k, v in summary.items() if k not in ("backends", "note")})
        return {"summary": summary | {"report": str(base.with_suffix(".json")), "sheet": str(base.with_suffix(".xlsx"))},
                "rows": rows}


def _write_xlsx(path: Path, rows: list[dict], summary: dict) -> None:
    import pandas as pd

    cols = ["id", "topic", "type", "difficulty", "question", "status", "answer", "citations", "n_citations",
            "confidence", "latency_ms", "doc_retrieved", "section_retrieved", "doc_cited", "section_cited", "citations_ok", "oos_correct",
            "keypoint_proxy", "expected_sources", "expected_answer"]
    df = pd.DataFrame([{c: (" \n".join(r[c]) if isinstance(r.get(c), list) else r.get(c)) for c in cols} for r in rows])
    df["Score (1/0.5/0)"] = ""
    df["Sources OK (Y/N)"] = ""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Answers", index=False)
        pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, dict) else v}
                      for k, v in summary.items()]).to_excel(xw, sheet_name="Summary", index=False)
        ws = xw.sheets["Answers"]
        for col, width in zip("ABCDEFGHIJKLMNOPQRST", (6, 18, 12, 6, 50, 12, 80, 60, 6, 8, 8, 8, 8, 8, 8, 8, 40, 60, 10, 10)):
            ws.column_dimensions[col].width = width
