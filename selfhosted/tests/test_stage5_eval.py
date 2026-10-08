"""Stage 5 - evaluation runner."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from privrag.errors import PrivRagError
from privrag.eval.runner import keypoint_proxy, parse_expected, run_eval
from privrag.index.pipeline import run_index
from privrag.index.store import close_clients
from privrag.ingest.pipeline import run_ingest


@pytest.fixture(autouse=True)
def _fresh():
    close_clients()
    yield
    close_clients()


def test_parse_expected_sources():
    assert parse_expected("01_GwG.pdf, section 10(3)") == [{"doc": "01_GwG", "sections": ["§ 10"]}]
    assert parse_expected("21_MaRisk_06_2026.pdf, AT 4.4.2")[0]["sections"] == ["AT 4.4.2"]
    assert parse_expected("27_DORA_2022_2554.pdf, Articles 2 and 64")[0]["sections"] == ["Artikel 2", "Artikel 64"]
    assert parse_expected(None) == [] and parse_expected("") == []


def test_keypoint_proxy():
    assert keypoint_proxy("At least five years, max ten years", "mindestens 5 Jahre, höchstens 10 Jahre") == 1.0
    assert keypoint_proxy("1) when X. 2) from EUR 15,000", "ab 15 000 Euro") == 1.0
    assert keypoint_proxy("no numbers here", "x") is None


def test_eval_end_to_end(settings, tmp_path):
    run_ingest(settings)
    run_index(settings)
    qs = pd.DataFrame([
        {"ID": "T1", "Topic": "AML", "Question": "Wann hat der Verpflichtete unverzüglich der Zentralstelle zu melden "
         "und wann sind Sorgfaltspflichten bei Transaktionen zu erfüllen?", "Type": "Fact", "Difficulty (1-3)": 1,
         "Expected answer (key points)": "Report without delay; due diligence from EUR 15,000",
         "Expected source(s)": "01_Test_GwG.pdf, section 43(1)", "Key terms (DE)": "Meldepflicht"},
        {"ID": "T2", "Topic": "OOS", "Question": "Wie backe ich einen Apfelkuchen mit Zimt?", "Type": "Out of scope",
         "Difficulty (1-3)": 1, "Expected answer (key points)": "Refuse", "Expected source(s)": None, "Key terms (DE)": None},
    ])
    path = tmp_path / "eval.xlsx"
    with pd.ExcelWriter(path) as xw:
        qs.to_excel(xw, sheet_name="Question set", index=False)
    settings.eval_path = path
    r = run_eval(settings)
    s = r["summary"]
    assert s["questions"] == 2 and s["errors"] == 0
    assert s["out_of_scope_refused"].startswith("1/1")
    row = r["rows"][0]
    assert row["doc_cited"] is True and row["section_cited"] is True and row["citations_ok"] is True
    out = pd.read_excel(s["sheet"], sheet_name="Answers")
    assert list(out["id"]) == ["T1", "T2"] and "Score (1/0.5/0)" in out.columns
    assert json.loads(open(s["report"], encoding="utf-8").read())["summary"]["questions"] == 2
    partial = list(settings.reports_dir.glob("eval_*.partial.jsonl"))
    assert len(partial) == 1 and len(partial[0].read_text(encoding="utf-8").splitlines()) == 2


def test_eval_missing_file(settings, tmp_path):
    settings.eval_path = tmp_path / "nope.xlsx"
    with pytest.raises(PrivRagError) as e:
        run_eval(settings)
    assert e.value.code == "EVAL_FILE_MISSING"


def test_embedding_progress_logged(settings, tmp_path):
    from privrag.embed.base import LSAEmbedder, embed_in_batches
    texts = [f"Text {i} über Meldepflichten und Sorgfaltspflichten Nummer {i}" for i in range(400)]
    e = LSAEmbedder(dim=8, model_path=tmp_path / "m.joblib")
    e.fit(texts)
    v = embed_in_batches(e, texts, batch_size=32)
    assert v.shape == (400, 8)
    log = (settings.log_dir / "latest.jsonl").read_text(encoding="utf-8")
    assert "embedding progress 400/400 (100%)" in log and "eta_s" in log
