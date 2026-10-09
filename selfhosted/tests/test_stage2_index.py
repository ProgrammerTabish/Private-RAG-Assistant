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


# ---------------------------------------------------------------- resumable / incremental indexing
class CountingEmbedder:
    """Deterministic 'remote' embedder that counts calls and can fail after N texts."""
    name, dim = "remote", 16

    def __init__(self, fp="fake-v1", fail_after=None):
        self.fp, self.fail_after, self.seen = fp, fail_after, 0

    def fit(self, texts):
        return None

    def embed_documents(self, texts):
        if self.fail_after is not None and self.seen + len(texts) > self.fail_after:
            raise KeyboardInterrupt("simulated Ctrl+C")
        self.seen += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dim] = 1.0
            out[i, (hash(t) // 7) % self.dim] += 0.5
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def embed_query(self, t):
        return self.embed_documents([t])[0]

    def fingerprint(self):
        return self.fp


@pytest.fixture()
def remote_settings(settings, monkeypatch):
    import privrag.index.pipeline as pl
    settings.embed_backend = "remote"
    settings.embed_url = "http://unused"
    settings.embed_batch_size = 2          # group = 64 -> several groups per corpus
    state = {"emb": CountingEmbedder()}
    monkeypatch.setattr(pl, "get_embedder", lambda s: state["emb"])
    run_ingest(settings)
    return settings, state


def _many_chunks(settings, n=150):
    """Make the corpus big enough for several upsert groups."""
    from privrag.ingest.pipeline import load_chunks as lc
    from privrag.models import Chunk
    base = lc(settings.chunks_path)
    extra = []
    for k in range(n):
        c = base[k % len(base)].model_copy()
        c.text = f"{c.text} Variante {k}"
        c.chunk_id = Chunk.make_id(c.doc_id, 1000 + k, c.text)
        extra.append(c)
    settings.chunks_path.write_text("".join(c.model_dump_json() + "\n" for c in base + extra), encoding="utf-8")
    return len(base) + n


def test_index_resumes_after_interruption(remote_settings):
    settings, state = remote_settings
    total = _many_chunks(settings)
    state["emb"] = CountingEmbedder(fail_after=100)          # dies inside the 2nd group
    with pytest.raises(KeyboardInterrupt):
        run_index(settings)
    saved = VectorStore(settings).count()
    assert saved == 64                                        # first group was saved
    state["emb"] = CountingEmbedder()
    r = run_index(settings)
    assert r["mode"] == "resume" and r["points"] == total
    assert state["emb"].seen == total - 64                    # only the missing chunks were embedded
    log = (settings.log_dir / "latest.jsonl").read_text(encoding="utf-8")
    assert "index plan: resume - 64 chunks already have dense vectors" in log and "embedding progress" in log


def test_index_incremental_embeds_only_new_chunks(remote_settings, corpus):
    from conftest import make_pdf
    settings, state = remote_settings
    first = run_index(settings)
    make_pdf(corpus / "02_Test_DORA.pdf", ["Artikel 1\nGegenstand\nNeuer Inhalt der Verordnung über Resilienz " * 3])
    run_ingest(settings)
    state["emb"] = CountingEmbedder()
    r = run_index(settings)
    from privrag.ingest.pipeline import load_chunks as lc
    new_dora = [c for c in lc(settings.chunks_path) if c.doc_id == "02_Test_DORA"]
    assert r["mode"] == "incremental" and state["emb"].seen == len(new_dora)
    assert r["points"] == len(lc(settings.chunks_path)) < first["points"] + len(new_dora)


def test_index_rebuilds_when_embedder_changes(remote_settings):
    settings, state = remote_settings
    first = run_index(settings)
    state["emb"] = CountingEmbedder(fp="fake-v2")
    r = run_index(settings, recreate=False)
    assert r["mode"] == "full" and state["emb"].seen == first["points"] == r["points"]


def test_progress_reporter_logs_eta(settings):
    from privrag.embed.base import Progress
    p = Progress(total=1000, already_done=200, every_s=0)
    import time as _t
    _t.sleep(0.01)
    p.advance(100)
    log = (settings.log_dir / "latest.jsonl").read_text(encoding="utf-8")
    assert "embedding progress 300/1000 (30%)" in log and '"eta_s"' in log


# ---------------------------------------------------------------- time-budgeted (laptop) indexing
def test_budget_mode_keyword_points_for_all_and_partial_dense(remote_settings, monkeypatch):
    import privrag.index.pipeline as pl
    settings, state = remote_settings
    total = _many_chunks(settings, n=200)

    class Slow(CountingEmbedder):
        def embed_documents(self, texts):
            import time as _t
            _t.sleep(0.01 * len(texts))
            return super().embed_documents(texts)

    state["emb"] = Slow()
    settings.index_time_budget_s = 0.6
    r = pl.run_index(settings)
    assert r["points"] == total                               # every chunk is keyword-searchable
    assert 0 < r["dense_points"] < total                      # dense only for part of it
    store = VectorStore(settings)
    pts = store.existing_points()
    dense_docs = {cid.split(":")[0] for cid, has in pts.items() if has}
    assert dense_docs == {"01_Test_GwG", "02_Test_DORA"}      # spread over all documents
    # keyword search finds a chunk that has no dense vector
    no_dense = [cid for cid, has in pts.items() if not has]
    assert no_dense
    # a second run continues the dense coverage
    r2 = pl.run_index(settings)
    assert r2["dense_points"] > r["dense_points"]
    # unlimited run completes it; then the index is up to date
    settings.index_time_budget_s = None
    r3 = pl.run_index(settings)
    assert r3["dense_points"] == total
    assert pl.run_index(settings)["status"] == "up_to_date"


def test_budget_zero_is_quick_start(remote_settings):
    import privrag.index.pipeline as pl
    settings, state = remote_settings
    settings.index_time_budget_s = 0
    r = pl.run_index(settings)
    assert r["dense_points"] == 0 and r["points"] > 0
    assert pl.run_index(settings)["status"] == "up_to_date"   # budget 0: no endless re-runs


def test_retrieval_works_with_partial_dense(remote_settings):
    import privrag.index.pipeline as pl
    from privrag.retrieve.retriever import Retriever
    settings, state = remote_settings
    settings.index_time_budget_s = 0
    pl.run_index(settings)
    res, dbg = Retriever(settings, embedder=state["emb"]).retrieve("Meldepflicht Zentralstelle Finanztransaktionsuntersuchungen")
    assert res and any(r.chunk.section and r.chunk.section.startswith("§ 43") for r in res)


def test_spread_order_round_robin():
    from privrag.index.pipeline import spread_order
    from privrag.models import Chunk
    mk = lambda d, s: Chunk(chunk_id=f"{d}:{s}", doc_id=d, file=f"{d}.pdf", title=d, section=None, page_start=1,
                            page_end=1, seq=s, text="x" * 50, char_len=50, regulator="r", doc_type="t", language="de")
    chunks = [mk("A", i) for i in range(8)] + [mk("B", i) for i in range(2)]
    order = spread_order(chunks, list(range(10)))
    first = [chunks[i].chunk_id for i in order[:4]]
    assert first == ["A:0", "B:0", "A:4", "B:1"]              # round robin, A jumps to its middle
    assert sorted(order) == list(range(10))


def test_budget_progress_shows_budget_not_full_eta(settings):
    import time as _t
    from privrag.embed.base import Progress
    p = Progress(total=12186, every_s=0, label="BGE-M3 vectors", deadline=_t.perf_counter() + 200)
    _t.sleep(0.01)
    p.advance(8)
    log = (settings.log_dir / "latest.jsonl").read_text(encoding="utf-8")
    assert "BGE-M3 vectors 8/12186" in log and "time budget: 3m" in log and "h" not in log.split("time budget:")[1][:6]
