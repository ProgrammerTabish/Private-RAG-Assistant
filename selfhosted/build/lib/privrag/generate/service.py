"""The RAG service: question -> (German rewrite) -> retrieve -> generate -> validate.

Every step is timed and logged under the request id. Failures degrade where it
is safe (rewrite or reranker failing) and return a structured ``status="error"``
answer where it is not (LLM down) - the caller and the audit trail always get a
record of what happened.
"""
from __future__ import annotations

import time
import uuid

from ..config import Settings
from ..errors import LLMError, PrivRagError, RetrievalError
from ..logging_setup import get_logger, request_context, stage, step
from ..models import Answer
from ..retrieve.retriever import Retriever
from . import prompts
from .answer import build_citations, clean_and_renumber, confidence, confidence_label, parse_citations
from .llm import ChatLLM, get_llm

log = get_logger("rag")

OUT_OF_SCOPE_TEXT = ("The curated regulatory documents do not contain information to answer this question. "
                     "Please rephrase it or consult the responsible department.")


def _usage_since(llm, mark: int) -> dict[str, int]:
    log_ = getattr(llm, "usage_log", None) or []
    calls = log_[mark:]
    return {"llm_calls": len(calls),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls)}


def interpret_refusal(raw: str) -> tuple[bool, str]:
    """Small models sometimes write the refusal marker *and then* a cited answer
    ("NOT_IN_SOURCES\n\nThe act also protects ... [1]"). Treat it as a refusal only if
    nothing substantial remains once the marker is removed; otherwise drop the marker
    and keep the answer. Returns (refused, cleaned_text)."""
    import re
    marker = prompts.NOT_IN_SOURCES
    if marker not in raw:
        return False, raw
    rest = re.sub(rf"\s*{marker}[.:]?\s*", " ", raw).strip()
    if len(rest) < 60 or not parse_citations(rest):
        return True, raw
    return False, MARKED + rest


MARKED = "\u2063"      # invisible flag: the model also emitted the refusal marker


class RagService:
    def __init__(self, s: Settings, retriever: Retriever | None = None, llm: ChatLLM | None = None):
        self.s = s
        self.retriever = retriever or Retriever(s)
        self.llm = llm or get_llm(s)

    def _rewrite(self, question: str, warnings: list[str]) -> str | None:
        if not self.s.query_rewrite:
            return None
        try:
            with step("rewrite query (German)", log, level=10) as r:
                out = self.llm.complete(prompts.REWRITE_SYSTEM, question, max_tokens=80).strip()
                out = out.splitlines()[0][:400] if out else ""
                r.update(german_query=out)
            return out or None
        except Exception as exc:  # rewrite is an optimisation, never fatal
            warnings.append(f"query rewrite failed ({getattr(exc, 'code', type(exc).__name__)})")
            log.warning("query rewrite failed - using original question", extra={"error": str(exc)})
            return None

    def ask(self, question: str, doc_ids: list[str] | None = None, regulators: list[str] | None = None,
            request_id: str | None = None) -> Answer:
        request_id = request_id or uuid.uuid4().hex[:12]
        t_all = time.perf_counter()
        lat: dict[str, float] = {}
        warnings: list[str] = []
        question = (question or "").strip()

        with request_context(request_id), stage("rag"):
            log.info("question received", extra={"chars": len(question), "doc_filter": doc_ids, "reg_filter": regulators})

            refs: list[str] = []
            usage_mark = len(getattr(self.llm, "usage_log", []))

            def done(status: str, text: str, cits=None, conf: float = 0.0, retrieved: int = 0) -> Answer:
                lat["total"] = round((time.perf_counter() - t_all) * 1000, 1)
                ans = Answer(question=question, answer=text, status=status, citations=cits or [],
                             confidence=conf, confidence_label=confidence_label(conf), retrieved=retrieved,
                             retrieved_refs=refs, model=self.llm.model, latency_ms=lat, warnings=warnings,
                             usage=_usage_since(self.llm, usage_mark),
                             request_id=request_id)
                lvl = log.error if status == "error" else log.info
                lvl("answer ready", extra={"status": status, "citations": len(ans.citations),
                                           "confidence": conf, "duration_ms": lat["total"]})
                return ans

            # 1. rewrite
            t0 = time.perf_counter()
            de_query = self._rewrite(question, warnings)
            lat["rewrite"] = round((time.perf_counter() - t0) * 1000, 1)

            # 2. retrieve
            t0 = time.perf_counter()
            try:
                with stage("rag.retrieve"), step("retrieve", log) as r:
                    chunks, dbg = self.retriever.retrieve(question, german_query=de_query, doc_ids=doc_ids,
                                                          regulators=regulators)
                    warnings += dbg["warnings"]
                    r.update(count=len(chunks), candidates=dbg.get("candidates"), german_query=dbg.get("german_query"),
                             top=[f"{c.chunk.doc_id}|{c.chunk.section}|{c.chunk.pages_label}" for c in chunks[:5]])
            except RetrievalError as exc:
                warnings.append(f"retrieval failed: {exc.code}")
                return done("error", f"Retrieval failed ({exc.code}). The request was logged as {request_id}.")
            lat["retrieve"] = round((time.perf_counter() - t0) * 1000, 1)
            refs.extend(f"{c.chunk.doc_id} | {c.chunk.section or '-'} | {c.chunk.pages_label}" for c in chunks)

            if not chunks:
                return done("out_of_scope", OUT_OF_SCOPE_TEXT)
            best_rerank = chunks[0].rerank_score
            if best_rerank is not None and best_rerank < self.s.min_relevance:
                warnings.append(f"best relevance {best_rerank:.2f} below threshold {self.s.min_relevance}")
                return done("out_of_scope", OUT_OF_SCOPE_TEXT, retrieved=len(chunks))

            # 3. generate (+1 retry if too few citations)
            t0 = time.perf_counter()
            user = prompts.answer_user_prompt(question, chunks, self.s.max_source_chars)
            try:
                with stage("rag.generate"), step("generate", log) as r:
                    raw = self.llm.complete(prompts.ANSWER_SYSTEM, user)
                    refused, raw = interpret_refusal(raw)
                    if refused and chunks and self.s.refusal_recheck:
                        log.warning("model refused although sources were found - asking it to re-check once")
                        refused2, raw2 = interpret_refusal(self.llm.complete(
                            prompts.ANSWER_SYSTEM, user + "\n\n" + prompts.RECHECK_REFUSAL))
                        r.update(rechecked=True, recheck_refused=refused2)
                        if not refused2:
                            refused, raw = False, raw2
                            warnings.append("answered after a second check (the first attempt refused)")
                    distinct = len([n for n in parse_citations(raw) if 1 <= n <= len(chunks)])
                    if (not refused and distinct < self.s.min_citations
                            and len(chunks) >= self.s.min_citations):
                        log.warning("too few citations - retrying once", extra={"cited": distinct})
                        retry_user = user + "\n\n" + prompts.RETRY_FEW_CITATIONS.format(n=self.s.min_citations)
                        refused2, raw2 = interpret_refusal(self.llm.complete(prompts.ANSWER_SYSTEM, retry_user))
                        d2 = 0 if refused2 else len([n for n in parse_citations(raw2) if 1 <= n <= len(chunks)])
                        if d2 > distinct:
                            raw, distinct = raw2, d2
                        r.update(retried=True)
                    r.update(chars=len(raw), cited=distinct)
            except LLMError as exc:
                lat["generate"] = round((time.perf_counter() - t0) * 1000, 1)
                warnings.append(f"generation failed: {exc.code}")
                return done("error", f"The language model is unavailable ({exc.code}). Request id {request_id}.",
                            retrieved=len(chunks))
            lat["generate"] = round((time.perf_counter() - t0) * 1000, 1)

            # 4. validate
            if refused:
                return done("out_of_scope", OUT_OF_SCOPE_TEXT, retrieved=len(chunks))
            if raw.startswith(MARKED):
                raw = raw[len(MARKED):]
                warnings.append("the model marked part of the question as not covered by the sources")
            text, order, invalid = clean_and_renumber(raw, len(chunks))
            if invalid:
                warnings.append(f"removed citations to non-existent sources: {invalid}")
                log.warning("invalid citation numbers removed", extra={"invalid": invalid})
            cits = build_citations(order, chunks, text)
            conf = confidence(chunks, order, text, self.s.min_citations)
            if not cits:
                warnings.append("answer has no citations - not shown as verified")
                return done("insufficient_sources", text, [], 0.0, len(chunks))
            if len({c.chunk_id for c in cits}) < self.s.min_citations:
                warnings.append(f"only {len(cits)} source cited (target {self.s.min_citations})")
            return done("answered", text, cits, conf, len(chunks))


def safe_ask(service: RagService, question: str, **kw) -> Answer:
    """ask() that never raises - unexpected bugs become a logged error answer."""
    try:
        return service.ask(question, **kw)
    except PrivRagError:
        raise
    except Exception as exc:
        log.exception("unexpected error in ask()", extra={"error_code": "UNEXPECTED"})
        return Answer(question=question, answer=f"Internal error ({type(exc).__name__}).", status="error",
                      citations=[], confidence=0.0, confidence_label="low", retrieved=0, model=service.llm.model,
                      latency_ms={}, warnings=[f"unexpected: {exc}"], request_id=kw.get("request_id"))
