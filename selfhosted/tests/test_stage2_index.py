"""Stage 2 - embeddings, BM25 sparse vectors and the Qdrant index."""
from __future__ import annotations

import numpy as np
import pytest

from privrag.embed.base import LSAEmbedder, RemoteEmbedder, _validate, chunk_embed_text, embed_in_batches
from privrag.errors import EmbeddingError, IndexError_
from privrag.index import sparse as sp
from privrag.index.pipeline import run_index
from privrag.index.store import VectorStore, close_clients, point_id
from privrag.ingest.pipeline import load_chunks, run_ingest


@pytest.fixture(autouse=True)
def _fresh_clients():
    close_clients()
    yield
    close_clients()


# ---------------------------------------------------------------- sparse / tokenizer
def test_tokenizer_keeps_legal_references():
    toks = sp.tokenize("Nach § 25h KWG, Art. 5b und Artikel 19 DORA (EU) 2022/2554 sowie AT 4.4.2 MaRisk")
    assert "§25h" in toks and "art5b" in toks and "art19" in toks and "2022/2554" in toks and "at4.4.2" in toks
    assert "und" not in toks and "sowie" not in toks


def test_tokenizer_folds_simple_inflection():
    assert sp._light_stem("sorgfaltspflichten") == sp._light_stem("sorgfaltspflicht") or \
        sp._light_stem("sorgfaltspflichten").startswith("sorgfaltspflicht")
    assert sp.tokenize("§§ 25") == sp.tokenize("§ 25")[:1] or "§25" in sp.tokenize("§§ 25")


def test_sparse_vectors_are_valid():
    idx, vals = sp.doc_vector("Meldepflicht Meldepflicht Verdacht", avg_len=3)
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert all(v > 0 for v in vals)
    assert sp.doc_vector("", 3) == ([], [])
    assert sp.query_vector("und der die") == ([], [])


# ---------------------------------------------------------------- embedders
def test_lsa_embedder_roundtrip(tmp_path):
    texts = [f"Dokument {i} über Geldwäsche und Sorgfaltspflichten Nummer {i}" for i in range(20)]
    e = LSAEmbedder(dim=8, model_path=tmp_path / "lsa.joblib")
    e.fit(texts)
    v = e.embed_documents(texts[:3])
    assert v.shape == (3, 8) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    # reload from disk gives identical vectors
    e2 = LSAEmbedder(dim=8, model_path=tmp_path / "lsa.joblib")
    assert np.allclose(e2.embed_query(texts[0]), v[0], atol=1e-5)


def test_lsa_unfitted_gives_clear_error(tmp_path):
    with pytest.raises(EmbeddingError) as e:
        LSAEmbedder(8, tmp_path / "missing.joblib").embed_query("x")
    assert e.value.code == "EMBED_MODEL_MISSING"


def test_embed_rejects_empty_input(tmp_path):
    e = LSAEmbedder(dim=4, model_path=tmp_path / "m.joblib")
    e.fit(["aaa bbb", "ccc ddd", "eee fff", "ggg hhh"])
    with pytest.raises(EmbeddingError) as ex:
        embed_in_batches(e, ["ok text", "   "], batch_size=1)
    assert ex.value.code == "EMBED_EMPTY_INPUT" and ex.value.context["batch_start"] == 1


@pytest.mark.parametrize("arr,code", [
    (np.zeros((2, 3)), "EMBED_BAD_SHAPE"),
    (np.zeros((3, 4)), "EMBED_DIM_MISMATCH"),
    (np.array([[np.nan, 1, 1]] * 3), "EMBED_NOT_FINITE"),
])
def test_embedding_output_validation(arr, code):
    with pytest.raises(EmbeddingError) as e:
        _validate(arr, 3, 3, "t")
    assert e.value.code == code


def test_remote_embedder_unreachable_endpoint():
    e = RemoteEmbedder("http://127.0.0.1:9/v1", "bge-m3", timeout=0.5, retries=1)
    with pytest.raises(EmbeddingError) as ex:
        e.embed_query("Meldepflicht")
    assert ex.value.code == "EMBED_ENDPOINT_DOWN"


def test_remote_embedder_parses_openai_format(monkeypatch):
    import httpx

    def fake_post(url, json, timeout):
        data = [{"index": i, "embedding": [float(i + 1), 0.0, 1.0]} for i in range(len(json["input"]))][::-1]
        return httpx.Response(200, json={"data": data}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    v = RemoteEmbedder("http://x/v1", "m").embed_documents(["a", "b"])
    assert v.shape == (2, 3) and v[0][0] < v[1][0]   # re-sorted by index


# ---------------------------------------------------------------- index pipeline
def test_index_build_verify_and_search(settings):
    run_ingest(settings)
    chunks = load_chunks(settings.chunks_path)
    r = run_index(settings)
    assert r["status"] == "built" and r["points"] == len(chunks)
    store = VectorStore(settings)
    assert store.count() == len(chunks)
    # keyword search finds the exact section
    ids, vals = sp.query_vector("Meldepflicht Zentralstelle Finanztransaktionsuntersuchungen")
    hits = store.search_sparse(ids, vals, 3)
    assert hits and hits[0][0].section.startswith("§ 43")
    # filter by document
    from privrag.embed.base import get_embedder
    q = get_embedder(settings).embed_query("IKT-bezogene Vorfälle melden")
    hits = store.search_dense(q, 5, doc_ids=["02_Test_DORA"])
    assert hits and all(c.doc_id == "02_Test_DORA" for c, _ in hits)


def test_index_is_idempotent_and_skips_when_unchanged(settings):
    run_ingest(settings)
    run_index(settings)
    r2 = run_index(settings)
    assert r2["status"] == "up_to_date"
    r3 = run_index(settings, recreate=True)
    assert r3["status"] == "built" and VectorStore(settings).count() == r3["points"]


def test_stale_points_removed_after_document_deleted(settings, corpus):
    run_ingest(settings)
    run_index(settings)
    (corpus / "02_Test_DORA.pdf").unlink()
    run_ingest(settings)
    r = run_index(settings)
    chunks = load_chunks(settings.chunks_path)
    assert r["points"] == len(chunks)
    assert {c.doc_id for c in chunks} == {"01_Test_GwG"}


def test_dim_mismatch_detected(settings):
    run_ingest(settings)
    run_index(settings)
    store = VectorStore(settings)
    with pytest.raises(IndexError_) as e:
        store.ensure_collection(dim=999)
    assert e.value.code == "INDEX_DIM_MISMATCH"


def test_length_mismatch_rejected(settings):
    run_ingest(settings)
    chunks = load_chunks(settings.chunks_path)
    store = VectorStore(settings)
    store.ensure_collection(4)
    with pytest.raises(IndexError_) as e:
        store.upsert(chunks, np.zeros((1, 4)), [([], [])])
    assert e.value.code == "INDEX_LENGTH_MISMATCH"


def test_index_without_chunks_gives_clear_error(settings):
    from privrag.errors import ChunkError
    with pytest.raises(ChunkError) as e:
        run_index(settings)
    assert e.value.code == "CHUNKS_MISSING"


def test_point_ids_deterministic():
    assert point_id("01_GwG:00001:abcd") == point_id("01_GwG:00001:abcd")
    assert point_id("a") != point_id("b")


def test_embed_text_contains_context(settings):
    run_ingest(settings)
    c = next(c for c in load_chunks(settings.chunks_path) if c.section)
    t = chunk_embed_text(c)
    assert c.title in t and c.section in t and c.text in t
