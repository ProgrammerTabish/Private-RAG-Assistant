"""Stage 1 - ingestion: unit tests and robustness (failure-injection) tests."""
from __future__ import annotations

import json
import shutil

import pytest

from conftest import make_image_only_pdf, make_pdf
from privrag.errors import ManifestError, ParseError
from privrag.ingest.cleaning import find_boilerplate, normalize_text
from privrag.ingest.manifest import load_manifest, reconcile
from privrag.ingest.parsers import PyMuPDFParser, validate_pdf_file
from privrag.ingest.pipeline import load_chunks, run_ingest
from privrag.ingest.sections import detect_heading
from privrag.models import DocMeta


# ---------------------------------------------------------------- sections
@pytest.mark.parametrize("line,nxt,label", [
    ("§ 10\xa0Allgemeine Sorgfaltspflichten", None, "§ 10"),
    ("§ 10 Allgemeine Sorgfaltspflichten", None, "§ 10"),          # NBSP lost by a converter
    ("§ 25h\xa0Interne Sicherungsmaßnahmen", None, "§ 25h"),
    ("§ \xa01", "Begriffsbestimmungen", "§ 1"),
    ("Artikel 19", "Meldung schwerwiegender Vorfälle", "Artikel 19"),
    ("Article 5b", "Prohibitions", "Article 5b"),
    ("AT 4.4.2 Risikocontrolling-Funktion", None, "AT 4.4.2"),
    ("BTO 1.2 Anforderungen an die Prozesse im Kreditgeschäft", None, "BTO 1.2"),
    ("Title II", None, "Title II"),
    ("4.1 Risk factors", None, "4.1"),
])
def test_headings_detected(line, nxt, label):
    h = detect_heading(line, nxt)
    assert h is not None and h.label == label


@pytest.mark.parametrize("line", [
    "§ 12 Absatz 4 Satz 1 ein.",                      # in-text reference, normal space
    "nach § 43 Absatz 1 und",
    "§ 21 die wirtschaftlich Berechtigten nach besonderen Merkmalen",
    "§ 25 KWG gilt entsprechend",
    "§ 261 des Strafgesetzbuches.",
    "§ 46 Abs. 1 Satz 5, § 59 Abs. 3 +++)",           # regression: '\b' after 'Abs.' never matched
    "§ 46\xa0Abs. 1 Satz 5",
    "§ 2 Nr. 4 bis 6",
    "Artikel 114 des Vertrags über die Arbeitsweise",  # reference, not heading
    "1. This document contains Guidelines issued pursuant to Article 16.",
    "4. Notifications will be published on the EBA website, in line with Article 16(3).",
    "",
])
def test_references_are_not_headings(line):
    assert detect_heading(line, None) is None


# ---------------------------------------------------------------- cleaning
def test_normalize_text_hyphens_nbsp_lists():
    raw = "Gesamtrisi-\nkoprofil und IKT-Risikomanage\xad\nmentrahmen\xa0X\nAufzeichnungs-\nund Aufbewahrung\n1.\nErste Pflicht"
    out = normalize_text(raw)
    assert "Gesamtrisikoprofil" in out
    assert "IKT-Risikomanagementrahmen" in out
    assert "\xa0" not in out
    assert "Aufzeichnungs-\nund" in out          # compound kept
    assert "1. Erste Pflicht" in out


def test_boilerplate_detected_only_when_repeated():
    pages = [["Header Bank", f"Inhalt {i}", f"Seite {i} von 9"] for i in range(9)]
    bp = find_boilerplate(pages, 0.5)
    assert "header bank" in bp and "seite # von #" in bp
    assert not any(k.startswith("inhalt") and "#" not in k for k in bp)


# ---------------------------------------------------------------- manifest
def test_manifest_variants(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("file;regulator;doc_type;title;publication_date;language\n"
                 "a.pdf;BaFin;law;A;2024-13-45x;DE\n"
                 "a.pdf;dup;law;A2;;de\n"
                 ";x;x;x;;\n"
                 "b.txt;x;x;x;;\n", encoding="utf-8-sig")
    r = load_manifest(p)
    assert list(r.docs) == ["a.pdf"]
    assert r.docs["a.pdf"].publication_date is None and r.docs["a.pdf"].language == "de"
    assert len(r.warnings) == 4  # bad date, duplicate, empty name, not pdf


def test_manifest_missing_file_column(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("name,title\na.pdf,A\n", encoding="utf-8")
    with pytest.raises(ManifestError) as e:
        load_manifest(p)
    assert e.value.code == "MANIFEST_NO_FILE_COLUMN"


def test_manifest_absent_is_not_fatal(tmp_path):
    r = load_manifest(tmp_path / "nope.csv")
    assert r.docs == {} and r.warnings


def test_reconcile_reports_both_directions(tmp_path):
    r = load_manifest(None)
    r.docs["ghost.pdf"] = DocMeta(doc_id="ghost", file="ghost.pdf", title="g")
    metas, warns = reconcile(r, [tmp_path / "real.pdf"])
    assert metas[0].from_manifest is False
    assert any("ghost.pdf" in w for w in warns) and any("real.pdf" in w for w in warns)


# ---------------------------------------------------------------- file validation / parsing
@pytest.mark.parametrize("content,code", [
    (b"", "PARSE_EMPTY_FILE"),
    (b"hello this is not a pdf", "PARSE_NOT_PDF"),
])
def test_validate_rejects_bad_files(tmp_path, content, code):
    p = tmp_path / "x.pdf"
    p.write_bytes(content)
    with pytest.raises(ParseError) as e:
        validate_pdf_file(p, 10)
    assert e.value.code == code


def test_validate_rejects_too_large(tmp_path):
    p = tmp_path / "big.pdf"
    p.write_bytes(b"%PDF-1.7\n" + b"0" * 2_000_000)
    with pytest.raises(ParseError) as e:
        validate_pdf_file(p, 1)
    assert e.value.code == "PARSE_TOO_LARGE"


def test_encrypted_pdf_rejected(tmp_path):
    p = make_pdf(tmp_path / "enc.pdf", ["geheim"], password="secret")
    with pytest.raises(ParseError) as e:
        PyMuPDFParser().parse(p, DocMeta(doc_id="enc", file="enc.pdf", title="enc"))
    assert e.value.code == "PARSE_ENCRYPTED"


def test_truncated_pdf_rejected(tmp_path, corpus):
    src = (corpus / "01_Test_GwG.pdf").read_bytes()
    p = tmp_path / "trunc.pdf"
    p.write_bytes(src[: len(src) // 3])
    with pytest.raises(ParseError):
        PyMuPDFParser().parse(p, DocMeta(doc_id="t", file="trunc.pdf", title="t"))


# ---------------------------------------------------------------- pipeline
def test_ingest_happy_path(settings):
    r = run_ingest(settings)
    assert r["ok"] == 2 and r["failed"] == 0
    chunks = load_chunks(settings.chunks_path)
    gwg = [c for c in chunks if c.doc_id == "01_Test_GwG"]
    sections = {c.section.split(" ")[0] + " " + c.section.split(" ")[1] for c in gwg if c.section}
    assert {"§ 1", "§ 10", "§ 43"} <= sections
    s10 = next(c for c in gwg if c.section and c.section.startswith("§ 10"))
    s10_text = " ".join(" ".join(c.text for c in gwg if c.section == s10.section).split())
    assert s10.page_start == 2 and "15 000 Euro" in s10_text
    # header/footer stripped, hyphenation fixed, in-text reference did not open a section
    all_text = " ".join(c.text for c in chunks)
    assert "Ein Service des Bundesamts" not in all_text and "Seite 2 von 3" not in all_text
    assert "Gesamtrisikoprofile" in all_text
    assert not any(c.section and c.section.startswith("§ 12") for c in chunks)
    dora = [c for c in chunks if c.doc_id == "02_Test_DORA"]
    assert any(c.section and c.section.startswith("Artikel 19 Meldung") for c in dora)
    assert dora[0].section.startswith("Erwägungsgründe")
    # metadata from manifest carried onto chunks
    assert gwg[0].regulator == "Bundestag" and gwg[0].publication_date == "2026-06-29"
    # chunk ids unique and deterministic
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert json.loads((settings.reports_dir / "ingest_latest.json").read_text())["ok"] == 2


def test_ingest_is_deterministic(settings):
    run_ingest(settings)
    a = [c.chunk_id for c in load_chunks(settings.chunks_path)]
    run_ingest(settings, force=True)
    b = [c.chunk_id for c in load_chunks(settings.chunks_path)]
    assert a == b


def test_bad_files_are_isolated(settings, corpus):
    (corpus / "03_empty.pdf").write_bytes(b"")
    (corpus / "04_fake.pdf").write_bytes(b"<html>not a pdf</html>")
    make_pdf(corpus / "05_locked.pdf", ["x"], password="pw")
    make_image_only_pdf(corpus / "06_scan.pdf")
    shutil.copy(corpus / "01_Test_GwG.pdf", corpus / "07_copy_of_gwg.pdf")
    r = run_ingest(settings)
    by = {d["doc_id"]: d for d in r["docs"]}
    assert by["01_Test_GwG"]["status"] == "ok" and by["02_Test_DORA"]["status"] == "ok"
    assert by["03_empty"]["error_code"] == "PARSE_EMPTY_FILE"
    assert by["04_fake"]["error_code"] == "PARSE_NOT_PDF"
    assert by["05_locked"]["error_code"] == "PARSE_ENCRYPTED"
    assert by["06_scan"]["error_code"] == "CHUNK_NO_TEXT"
    assert by["07_copy_of_gwg"]["status"] == "duplicate"
    assert r["failed"] == 4 and r["ok"] == 2
    # the good documents still made it into chunks.jsonl
    assert {c.doc_id for c in load_chunks(settings.chunks_path)} == {"01_Test_GwG", "02_Test_DORA"}
    # every failure is in the log with its error code
    log_text = (settings.log_dir / "latest.jsonl").read_text()
    for code in ("PARSE_EMPTY_FILE", "PARSE_NOT_PDF", "PARSE_ENCRYPTED", "CHUNK_NO_TEXT"):
        assert code in log_text


def test_rerun_skips_unchanged_and_reprocesses_changed(settings, corpus):
    run_ingest(settings)
    r2 = run_ingest(settings)
    assert r2["skipped_unchanged"] == 2 and r2["total_chunks"] > 0
    make_pdf(corpus / "02_Test_DORA.pdf", ["Artikel 1\nGegenstand\nNeuer Inhalt der Verordnung über Resilienz " * 3])
    r3 = run_ingest(settings)
    st = {d["doc_id"]: d["status"] for d in r3["docs"]}
    assert st == {"01_Test_GwG": "skipped_unchanged", "02_Test_DORA": "ok"}


def test_settings_change_invalidates_checkpoints(settings):
    run_ingest(settings)
    settings.chunk_size = 300
    r = run_ingest(settings)
    assert r["skipped_unchanged"] == 0 and r["ok"] == 2


def test_resume_after_crash(settings, monkeypatch):
    """Simulate a crash while processing the 2nd document: the 1st is checkpointed."""
    import privrag.ingest.pipeline as pl

    real = pl.chunk_document

    def boom(parsed, **kw):
        if parsed.meta.doc_id == "02_Test_DORA":
            raise KeyboardInterrupt("simulated crash")
        return real(parsed, **kw)

    monkeypatch.setattr(pl, "chunk_document", boom)
    with pytest.raises(KeyboardInterrupt):
        run_ingest(settings)
    monkeypatch.setattr(pl, "chunk_document", real)
    r = run_ingest(settings)
    st = {d["doc_id"]: d["status"] for d in r["docs"]}
    assert st == {"01_Test_GwG": "skipped_unchanged", "02_Test_DORA": "ok"}


def test_corrupted_state_file_recovers(settings):
    run_ingest(settings)
    (settings.data_dir / "ingest_state.json").write_text("{not json", encoding="utf-8")
    r = run_ingest(settings)
    assert r["ok"] == 2


def test_missing_pdf_dir_raises_clear_error(settings, tmp_path):
    settings.pdf_dir = tmp_path / "missing"
    with pytest.raises(ParseError) as e:
        run_ingest(settings)
    assert e.value.code == "PARSE_DIR_MISSING"


def test_missing_docling_falls_back_to_pymupdf(settings):
    settings.parser_backend = "docling"   # not installed locally
    r = run_ingest(settings, force=True)
    assert r["ok"] == 2
    assert all(d["parser"] == "pymupdf" for d in r["docs"])


# ---------------------------------------------------------------- logging robustness
def test_logging_reserved_keys_do_not_crash(settings):
    from privrag.errors import ParseError
    from privrag.logging_setup import get_logger
    log = get_logger("test.reserved")
    err = ParseError("bad", code="X")
    log.error("with reserved keys", extra=err.to_dict() | {"name": "n", "args": 1, "module": "m"})
    text = (settings.log_dir / "latest.jsonl").read_text()
    assert '"x_message": "bad"' in text and '"x_name": "n"' in text


def test_code_change_invalidates_checkpoints(settings, monkeypatch):
    import privrag.ingest.pipeline as pl
    run_ingest(settings)
    monkeypatch.setattr(pl, "_code_fingerprint", lambda: "different-code")
    r = run_ingest(settings)
    assert r["skipped_unchanged"] == 0 and r["ok"] == 2
