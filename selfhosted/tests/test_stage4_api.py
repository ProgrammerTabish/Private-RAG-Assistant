"""Stage 4 - FastAPI backend and audit trail."""
from __future__ import annotations

import concurrent.futures as cf

import pytest
from fastapi.testclient import TestClient

from privrag.api.app import create_app
from privrag.generate.llm import FakeLLM
from privrag.generate.service import RagService
from privrag.index.pipeline import run_index
from privrag.index.store import close_clients
from privrag.ingest.pipeline import run_ingest

Q = "Wann hat der Verpflichtete unverzüglich der Zentralstelle zu melden und wann sind Sorgfaltspflichten bei Transaktionen zu erfüllen?"


@pytest.fixture(autouse=True)
def _fresh():
    close_clients()
    yield
    close_clients()


@pytest.fixture()
def client(settings):
    run_ingest(settings)
    run_index(settings)
    settings.log_console = False
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["index_points"] > 0 and h["variant"] == "self-hosted"
    assert client.get("/ready").json()["ready"] is True


def test_ask_returns_cited_answer_and_writes_audit(client):
    r = client.post("/ask", json={"question": Q, "user": "compliance.officer@bank.de"},
                    headers={"X-Request-ID": "rid-001"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "rid-001"
    a = r.json()
    assert a["status"] == "answered" and len(a["citations"]) >= 2 and a["request_id"] == "rid-001"
    row = client.get("/audit/rid-001").json()
    assert row["user"] == "compliance.officer@bank.de" and row["question"] == Q
    assert row["citations"][0]["file"].endswith(".pdf") and row["status"] == "answered"
    assert client.get("/audit", params={"limit": 5}).json()[0]["request_id"] == "rid-001"


def test_source_lookup_for_highlighting(client):
    a = client.post("/ask", json={"question": Q}).json()
    cid = a["citations"][0]["chunk_id"]
    src = client.get(f"/sources/{cid}").json()
    assert src["chunk_id"] == cid and a["citations"][0]["quote"][:30] in " ".join(src["text"].split())
    assert client.get("/sources/does-not-exist").status_code == 404


def test_documents_endpoint(client):
    docs = client.get("/documents").json()
    assert {d["doc_id"] for d in docs} == {"01_Test_GwG", "02_Test_DORA"}
    assert docs[0]["title"] and docs[0]["chunks"] > 0


@pytest.mark.parametrize("body,code", [
    ({}, 422), ({"question": ""}, 422), ({"question": "   "}, 422), ({"question": 123}, 422),
    ({"question": "x" * 5000}, 422),
])
def test_input_validation(client, body, code):
    assert client.post("/ask", json=body).status_code == code


def test_filters_passed_through(client):
    a = client.post("/ask", json={"question": "Meldung IKT-bezogene Vorfälle Behörde", "doc_ids": ["02_Test_DORA"]}).json()
    assert all(c["doc_id"] == "02_Test_DORA" for c in a["citations"])


def test_llm_down_returns_503_but_is_audited(settings):
    run_ingest(settings)
    run_index(settings)
    settings.log_console = False
    svc = RagService(settings, llm=FakeLLM(mode="down"))
    with TestClient(create_app(settings, service=svc)) as c:
        r = c.post("/ask", json={"question": "Meldepflicht"}, headers={"X-Request-ID": "rid-down"})
        assert r.status_code == 503 and r.json()["status"] == "error"
        assert c.get("/audit/rid-down").json()["status"] == "error"


def test_degraded_mode_without_index(settings):
    settings.log_console = False
    with TestClient(create_app(settings)) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["startup_error"]["error_code"] in ("INDEX_NOT_BUILT", "CHUNKS_MISSING")
        r = c.post("/ask", json={"question": "Meldepflicht"})
        assert r.status_code == 503 and "INDEX_NOT_BUILT" in r.text


def test_api_key_enforced(settings):
    run_ingest(settings)
    run_index(settings)
    settings.api_key = "s3cret"
    settings.log_console = False
    with TestClient(create_app(settings)) as c:
        assert c.post("/ask", json={"question": Q}).status_code == 401
        assert c.post("/ask", json={"question": Q}, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200        # liveness stays open


def test_audit_failure_does_not_lose_answer(client, monkeypatch):
    # break the audit DB under the running app
    import privrag.api.audit as audit_mod
    monkeypatch.setattr(audit_mod.AuditLog, "record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    r = client.post("/ask", json={"question": Q})
    assert r.status_code == 200 and "audit write failed" in r.json()["warnings"]


def test_concurrent_requests(client):
    def call(i):
        return client.post("/ask", json={"question": Q}, headers={"X-Request-ID": f"c-{i}"}).status_code
    with cf.ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(call, range(16)))
    assert codes == [200] * 16
    assert len(client.get("/audit", params={"limit": 100}).json()) == 16


def test_request_logged_with_id(client, settings):
    client.post("/ask", json={"question": Q}, headers={"X-Request-ID": "trace-me"})
    lines = [l for l in (settings.log_dir / "latest.jsonl").read_text().splitlines() if "trace-me" in l]
    assert any("POST /ask -> 200" in l for l in lines) and any("answer ready" in l for l in lines)
