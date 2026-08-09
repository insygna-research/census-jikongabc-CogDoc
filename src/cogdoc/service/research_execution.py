from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from functools import partial
from threading import RLock
from typing import Any

from cogdoc.api.research_job_store import (
    ResearchJobStateConflictError,
    ResearchJobStore,
)
from cogdoc.config.settings import get_settings
from cogdoc.service.kb_readers import kb_read_lease
from cogdoc.service.retrieval_pipeline import (
    build_retrieval_queries,
    retrieve_candidate_pool,
)
from cogdoc.service.retriever_factory import RetrieverFactory
from cogdoc.tools.reranker import BGEReranker, skipped_cpu_rerank_docs


ResearchRetriever = Callable[[str, str], Sequence[Mapping[str, Any]]]
ResearchReportBuilder = Callable[[Mapping[str, Any]], Any]


def retrieve_research_evidence(
    kb_id: str,
    query: str,
    *,
    state_runtime,
    top_k: int = 8,
) -> list[Mapping[str, Any]]:
    """Reuse the production hybrid retrieval path for one research section."""

    settings = get_settings()
    with kb_read_lease(kb_id):
        engine = RetrieverFactory.get_engine(kb_id)
        result = retrieve_candidate_pool(
            engine,
            state_runtime.derived_knowledge_retriever,
            state_runtime.retrieval_feedback_store,
            kb_id=kb_id,
            original_query=query,
            queries=build_retrieval_queries(query, max_queries=1),
            top_k=top_k,
            rrf_k=float(settings.hybrid_rrf_k),
        )
        docs = list(result.docs)
        if not docs:
            return []
        target_device = BGEReranker.default_device()
        if target_device == "cpu" and not settings.qa_rerank_on_cpu:
            return skipped_cpu_rerank_docs(docs, min(top_k, len(docs)))
        return BGEReranker.rerank(
            query,
            docs,
            top_n=min(top_k, len(docs)),
            device=target_device,
        )


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def public_research_evidence(
    docs: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
    preview_chars: int = 480,
) -> list[dict[str, Any]]:
    """Persist only bounded public evidence coordinates, never full chunk text."""

    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        meta = doc.get("meta") if isinstance(doc.get("meta"), Mapping) else {}
        retrieval = (
            doc.get("retrieval")
            if isinstance(doc.get("retrieval"), Mapping)
            else {}
        )
        chunk_id = str(meta.get("chunk_id") or "")
        knowledge_id = str(meta.get("knowledge_id") or "")
        identity = (chunk_id, knowledge_id)
        if not any(identity) or identity in seen:
            continue
        seen.add(identity)
        page = meta.get("page")
        item = {
            "chunk_id": chunk_id,
            "source_type": str(meta.get("source_type") or "document"),
            "knowledge_id": knowledge_id,
            "source": str(meta.get("source") or ""),
            "page": page,
            "page_start": meta.get("page_start", page),
            "page_end": meta.get("page_end", page),
            "section_title": str(meta.get("section_title") or ""),
            "text_preview": " ".join(str(doc.get("text") or "").split())[
                :preview_chars
            ],
            "search_channel": str(retrieval.get("search_channel") or ""),
            "rerank_score": _safe_number(retrieval.get("rerank_score")),
            "rrf_score": _safe_number(retrieval.get("rrf_score")),
        }
        evidence.append(item)
        if len(evidence) >= limit:
            break
    return evidence


class ResearchExecutionManager:
    """Durable section-at-a-time research evidence executor."""

    def __init__(
        self,
        store: ResearchJobStore,
        *,
        retrieve: ResearchRetriever,
        kb_exists: Callable[[str], bool],
        report_builder: ResearchReportBuilder | None = None,
        max_workers: int = 2,
    ):
        self._store = store
        self._retrieve = retrieve
        self._kb_exists = kb_exists
        self._report_builder = report_builder
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="cogdoc-research",
        )
        self._lock = RLock()
        self._active: dict[str, Future] = {}
        self._closed = False

    @classmethod
    def from_runtime(
        cls,
        store: ResearchJobStore,
        *,
        state_runtime,
        kb_exists: Callable[[str], bool],
        max_workers: int = 2,
        top_k: int = 8,
    ) -> "ResearchExecutionManager":
        # Local import avoids coupling the evidence executor to report-generation
        # dependencies during module import.
        from cogdoc.service.research_report import ResearchReportBuilder as Builder

        return cls(
            store,
            retrieve=partial(
                retrieve_research_evidence,
                state_runtime=state_runtime,
                top_k=top_k,
            ),
            kb_exists=kb_exists,
            report_builder=Builder.from_runtime(
                state_runtime=state_runtime,
                is_local=False,
            ),
            max_workers=max_workers,
        )

    def reconcile_orphans(self) -> int:
        return self._store.reconcile_running()

    def start(self, job_id: str) -> dict[str, Any]:
        row = self._store.start(job_id)
        self._schedule_evidence(job_id, str(row.get("execution_id") or ""))
        return row

    def resume(self, job_id: str) -> dict[str, Any]:
        row = self._store.resume(job_id)
        self._schedule_evidence(job_id, str(row.get("execution_id") or ""))
        return row

    def compile(self, job_id: str) -> dict[str, Any]:
        if self._report_builder is None:
            raise ResearchJobStateConflictError("research report builder is unavailable")
        row = self._store.begin_report(job_id)
        self._schedule_report(
            job_id,
            str(row.get("report_execution_id") or ""),
        )
        return row

    def pause(self, job_id: str) -> dict[str, Any]:
        return self._store.pause(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._store.cancel(job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _schedule_evidence(self, job_id: str, execution_id: str) -> None:
        if not execution_id:
            return
        active_key = f"evidence:{job_id}"
        with self._lock:
            if self._closed:
                raise RuntimeError("ResearchExecutionManager is closed")
            current = self._active.get(active_key)
            if current is not None and not current.done():
                return
            future = self._executor.submit(self._run_job, job_id, execution_id)
            self._active[active_key] = future
            future.add_done_callback(
                lambda completed: self._forget(active_key, completed)
            )

    def _schedule_report(self, job_id: str, report_execution_id: str) -> None:
        if not report_execution_id:
            return
        active_key = f"report:{job_id}:{report_execution_id}"
        with self._lock:
            if self._closed:
                raise RuntimeError("ResearchExecutionManager is closed")
            current = self._active.get(active_key)
            if current is not None and not current.done():
                return
            future = self._executor.submit(
                self._run_report,
                job_id,
                report_execution_id,
            )
            self._active[active_key] = future
            future.add_done_callback(
                lambda completed: self._forget(active_key, completed)
            )

    def _forget(self, active_key: str, future: Future) -> None:
        with self._lock:
            if self._active.get(active_key) is future:
                self._active.pop(active_key, None)

    def _run_report(self, job_id: str, report_execution_id: str) -> None:
        try:
            job = self._store.get(job_id)
            if job is None:
                return
            result = self._report_builder(job) if self._report_builder else None
            if result is None:
                raise RuntimeError("research report builder returned no result")
            if isinstance(result, Mapping):
                payload = dict(result)
            elif is_dataclass(result):
                payload = asdict(result)
            else:
                raise TypeError("research report builder returned unsupported result")
            self._store.complete_report(
                job_id,
                report_execution_id=report_execution_id,
                result=payload,
            )
        except Exception as exc:
            self._store.fail_report(
                job_id,
                report_execution_id=report_execution_id,
                error_class=type(exc).__name__,
            )

    def _run_job(self, job_id: str, execution_id: str) -> None:
        while True:
            row, section = self._store.claim_next_section(job_id, execution_id)
            if section is None:
                return
            section_id = str(section.get("section_id") or "")
            started = time.monotonic()
            try:
                kb_id = str(row.get("kb_id") or "")
                if not self._kb_exists(kb_id):
                    raise LookupError("knowledge base no longer exists")
                query = str(section.get("research_question") or "")
                docs = list(self._retrieve(kb_id, query))
                evidence = public_research_evidence(docs)
                self._store.complete_section(
                    job_id,
                    section_id,
                    execution_id=execution_id,
                    evidence_status="partial" if evidence else "missing",
                    evidence=evidence,
                    execution_metrics={
                        "candidate_count": len(docs),
                        "evidence_count": len(evidence),
                        "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    },
                )
            except Exception as exc:
                self._store.fail_section(
                    job_id,
                    section_id,
                    execution_id=execution_id,
                    error_class=type(exc).__name__,
                )
                return
