"""LLM clients.

* ``openai`` - any OpenAI-compatible chat endpoint via LangChain ``ChatOpenAI``.
               On Azure this is **vLLM serving Llama 3.3 / Mixtral inside the tenant**
               (``PRIVRAG_LLM_BASE_URL=http://<private-ip>:8000/v1``). No data leaves the VNet.
* ``fake``   - deterministic extractive stand-in for local tests: picks the source
               sentences that best match the question and cites them. It also
               simulates failures (timeout, no citations, invalid citations).
"""
from __future__ import annotations

import re
import time
from typing import Protocol

from ..config import Settings
from ..errors import LLMError
from ..index.sparse import tokenize
from ..logging_setup import get_logger
from .prompts import NOT_IN_SOURCES

log = get_logger("generate.llm")


class ChatLLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str: ...


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, s: Settings):
        from langchain_openai import ChatOpenAI

        self.model = s.llm_model
        self.usage_log: list[dict] = []     # token usage per call (read by the service)
        self._llm = ChatOpenAI(
            base_url=s.llm_base_url, api_key=s.llm_api_key, model=s.llm_model,
            temperature=s.llm_temperature, max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s, max_retries=s.llm_max_retries,
        )

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        import openai

        llm = self._llm.bind(max_tokens=max_tokens) if max_tokens else self._llm
        try:
            msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except openai.APITimeoutError as exc:
            raise LLMError(f"LLM timed out: {exc}", code="LLM_TIMEOUT") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"LLM endpoint unreachable: {exc}", code="LLM_UNREACHABLE") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"LLM returned HTTP {exc.status_code}: {str(exc)[:300]}", code="LLM_HTTP_ERROR",
                           status=exc.status_code) from exc
        except Exception as exc:
            raise LLMError(f"LLM call failed: {type(exc).__name__}: {exc}", code="LLM_FAILED") from exc
        text = (msg.content or "").strip() if isinstance(msg.content, str) else str(msg.content)
        if not text:
            raise LLMError("LLM returned an empty answer", code="LLM_EMPTY")
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or {}
        self.usage_log.append({"prompt_tokens": usage.get("prompt_tokens") or 0,
                               "completion_tokens": usage.get("completion_tokens") or 0})
        log.debug("llm usage", extra={"prompt_tokens": usage.get("prompt_tokens"),
                                      "completion_tokens": usage.get("completion_tokens"),
                                      "finish_reason": meta.get("finish_reason")})
        if meta.get("finish_reason") == "length":
            log.warning("LLM answer was cut off at max_tokens", extra={"max_tokens": max_tokens})
        return text


_SRC = re.compile(r"^\[(\d+)\] (.+?)\n(.*?)(?=\n\n\[\d+\] |\n\nAnswer|\Z)", re.S | re.M)
_SENT = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÄÖÜ(\d])")


class FakeLLM:
    """Deterministic extractive 'LLM' for tests. ``mode`` simulates failures:
    ok | timeout | down | empty | no_citations | bad_citations | one_citation | injection_echo"""

    name = "fake"
    model = "fake-extractive"

    def __init__(self, mode: str = "ok", delay_s: float = 0.0):
        self.mode = mode
        self.delay_s = delay_s
        self.calls: list[tuple[str, str]] = []
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append((system, user))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.mode == "timeout":
            raise LLMError("LLM timed out (simulated)", code="LLM_TIMEOUT")
        if self.mode == "down":
            raise LLMError("LLM endpoint unreachable (simulated)", code="LLM_UNREACHABLE")
        if self.mode == "empty":
            raise LLMError("LLM returned an empty answer (simulated)", code="LLM_EMPTY")
        if system.startswith("SEARCH QUERY REWRITE"):
            return ""  # the fake cannot translate; the glossary covers German terms locally
        question = user.split("\n", 1)[0].removeprefix("Question: ")
        sources = {int(m.group(1)): m.group(3) for m in _SRC.finditer(user.split("Sources:\n", 1)[-1])}
        if not sources:
            return NOT_IN_SOURCES
        q = set(tokenize(question))
        scored = []
        for n, text in sources.items():
            for sent in _SENT.split(text.replace("\n", " ")):
                if 30 <= len(sent) <= 500:
                    overlap = len(q & set(tokenize(sent)))
                    # behave like an instructed LLM: one shared word ("best") is not support
                    if overlap >= min(2, len(q)):
                        scored.append((overlap, -n, n, sent.strip()))
        if not scored:
            return NOT_IN_SOURCES
        scored.sort(reverse=True)
        picked, used = [], set()
        for _, _, n, sent in scored:           # best sentence per source, max 3 sources
            if n not in used:
                picked.append((n, sent))
                used.add(n)
            if len(picked) == 3:
                break
        if self.mode == "one_citation":
            picked = picked[:1]
        if self.mode == "no_citations":
            return " ".join(s for _, s in picked)
        if self.mode == "bad_citations":
            return " ".join(f"{s} [{n + 40}]" for n, s in picked) + f" {picked[0][1]} [{picked[0][0]}]"
        return " ".join(f"{s} [{n}]" for n, s in picked)


def get_llm(s: Settings) -> ChatLLM:
    if s.llm_backend == "fake":
        return FakeLLM()
    return OpenAICompatLLM(s)
