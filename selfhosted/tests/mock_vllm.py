"""A tiny OpenAI-compatible server that stands in for vLLM / TEI in tests.

It lets the *real* LangChain ChatOpenAI client and the RemoteEmbedder run
against HTTP, including slow responses (timeouts) and server errors.
"""
from __future__ import annotations

import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

STATE = {"mode": "ok", "delay": 0.0, "answer": "Die Erstmeldung erfolgt innerhalb von vier Stunden [1]. Die Pflicht folgt aus Artikel 19 [2].",
         "requests": []}

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    STATE["requests"].append(body)
    STATE["headers"] = dict(req.headers)
    if STATE["delay"]:
        time.sleep(STATE["delay"])
    if STATE["mode"] == "500":
        return JSONResponse({"error": {"message": "CUDA out of memory"}}, status_code=500)
    if STATE["mode"] == "400":
        return JSONResponse({"error": {"message": "context length exceeded"}}, status_code=400)
    system = body["messages"][0]["content"]
    content = "Meldung schwerwiegender IKT-Vorfälle Frist" if system.startswith("SEARCH QUERY REWRITE") else STATE["answer"]
    if STATE["mode"] == "empty":
        content = ""
    return {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
    return {"data": [{"index": i, "embedding": [float(len(t) % 7 + 1), 1.0, 0.5, float(i)]} for i, t in enumerate(inp)]}


@app.post("/rerank")
async def rerank(req: Request):
    body = await req.json()
    if STATE["mode"] == "rerank_down":
        return JSONResponse({"error": "down"}, status_code=503)
    q = set(body["query"].lower().split())
    return [{"index": i, "score": float(len(q & set(t.lower().split())))} for i, t in enumerate(body["texts"])]


class MockServer:
    def __init__(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=5)
