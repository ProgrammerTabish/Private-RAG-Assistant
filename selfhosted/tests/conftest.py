"""Shared fixtures: synthetic PDFs (good and deliberately broken) and isolated settings."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from privrag.config import Settings, reset_settings_cache
from privrag.logging_setup import setup_logging

LAW_PAGES = [
    "Ein Service des Bundesamts für Justiz\n§ 1\xa0Begriffsbestimmungen\n(1) Geldwäsche ist eine Straftat nach § 261 des "
    "Strafgesetzbuches.\n(2) Terrorismusfinanzierung ist die Bereitstellung von Vermögensgegenständen.\nSeite 1 von 3",
    "Ein Service des Bundesamts für Justiz\n§ 10\xa0Allgemeine Sorgfaltspflichten\n(1) Die allgemeinen Sorgfaltspflichten sind:\n1.\n"
    "die Identifizierung des Vertragspartners,\n2.\ndie Abklärung, ob der Vertragspartner für einen wirtschaftlich "
    "Berechtigten handelt.\n(3) Die Sorgfaltspflichten sind zu erfüllen bei der Begründung einer Geschäftsbeziehung und "
    "bei Transaktionen ab 15 000 Euro. Die Aufbewahrungsfrist beträgt fünf Jahre.\nSeite 2 von 3",
    "Ein Service des Bundesamts für Justiz\nnach § 12 Absatz 4 Satz 1 sind die Unterlagen aufzubewahren und die "
    "Gesamtrisi-\nkoprofile zu prüfen.\n§ 43\xa0Meldepflicht von Verpflichteten\n(1) Liegen Tatsachen vor, die darauf "
    "hindeuten, dass ein Vermögensgegenstand aus einer strafbaren Handlung stammt, hat der Verpflichtete dies "
    "unverzüglich der Zentralstelle für Finanztransaktionsuntersuchungen zu melden.\nSeite 3 von 3",
]

EU_PAGES = [
    "Erwägungsgründe: Die Union sollte die digitale operationale Resilienz stärken, damit Finanzunternehmen "
    "IKT-Risiken beherrschen können. " * 3,
    "Artikel 19\nMeldung schwerwiegender IKT-bezogener Vorfälle\n(1) Finanzunternehmen melden schwerwiegende "
    "IKT-bezogene Vorfälle der zuständigen Behörde. Die Erstmeldung erfolgt innerhalb von vier Stunden nach der "
    "Klassifizierung und spätestens 24 Stunden nach Kenntnisnahme.",
    "Artikel 30\nWesentliche Vertragsbestimmungen\n(1) Die Rechte und Pflichten des Finanzunternehmens und des "
    "IKT-Drittdienstleisters werden schriftlich festgelegt. Der vollständige Vertrag umfasst die Dienstleistungsgütevereinbarungen.",
]


def make_pdf(path: Path, pages: list[str], password: str | None = None) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10, fontname="helv")
    kwargs = {}
    if password:
        kwargs = dict(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "x")
    doc.save(path, **kwargs)
    doc.close()
    return path


def make_image_only_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(path)
    doc.close()
    return path


MANIFEST = (
    "file,regulator,doc_type,title,publication_date,language\n"
    "01_Test_GwG.pdf,Bundestag,law,Geldwäschegesetz (Test),2026-06-29,de\n"
    "02_Test_DORA.pdf,EU,regulation,DORA (Test),2022-12-14,de\n"
)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "pdfs"
    d.mkdir()
    make_pdf(d / "01_Test_GwG.pdf", LAW_PAGES)
    make_pdf(d / "02_Test_DORA.pdf", EU_PAGES)
    (tmp_path / "manifest.csv").write_text(MANIFEST, encoding="utf-8-sig")
    return d


@pytest.fixture()
def settings(tmp_path: Path, corpus: Path) -> Settings:
    reset_settings_cache()
    s = Settings(
        env="test", pdf_dir=corpus, manifest_path=tmp_path / "manifest.csv",
        data_dir=tmp_path / "data", log_dir=tmp_path / "logs", qdrant_mode="memory",
        chunk_size=400, chunk_overlap=50, min_chunk_chars=40, embed_dim=32, top_k=4, candidate_k=20,
        _env_file=None,
    )
    s.ensure_dirs()
    setup_logging(s.log_dir, console=False, level="DEBUG")
    return s
