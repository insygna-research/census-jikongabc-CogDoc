import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from typing import Callable
from uuid import uuid4

from config.settings import get_settings
from observability.logger import log_event
from service.ingest_service import build_kb_index


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KBExistsError(Exception):
    pass


class KnowledgeBaseRegistry:
    # 知识库元数据的 JSON 注册表；source/chroma/bm25/manifest 仍按 kb_id 物理隔离。
    def __init__(
        self,
        registry_path: str | None = None,
        source_dir_for: Callable[[str], str] | None = None,
    ):
        settings = get_settings()
        self._path = registry_path or settings.kb_registry_path
        self._source_dir_for = source_dir_for or settings.kb_source_dir
        self._lock = RLock()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._entries = self._load()

    def _load(self) -> dict:
        # registry 损坏（写入中崩溃留半截 JSON）时退回空表，保证服务仍能启动。
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        # 原子写：先写临时文件再 rename，避免中途崩溃留下半截 JSON。
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def source_dir(self, kb_id: str) -> str:
        return self._source_dir_for(kb_id)

    def create(self, kb_id: str) -> dict:
        with self._lock:
            if kb_id in self._entries:
                raise KBExistsError(kb_id)
            os.makedirs(self._source_dir_for(kb_id), exist_ok=True)
            # tenant_id/owner_id 现填默认值，为未来多租户隔离预留。
            record = {
                "kb_id": kb_id,
                "created_at": _now_iso(),
                "tenant_id": "default",
                "owner_id": "default",
            }
            self._entries[kb_id] = record
            self._save()
            return dict(record)

    def exists(self, kb_id: str) -> bool:
        with self._lock:
            return kb_id in self._entries

    def get(self, kb_id: str) -> dict | None:
        with self._lock:
            record = self._entries.get(kb_id)
            return dict(record) if record else None

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(record) for record in self._entries.values()]


class IndexJobManager:
    # 入库跑在独立单线程池里，串行化同库写、且绝不占用 chat 的 offload 线程。
    def __init__(
        self,
        ingest_fn: Callable[[str, str], object] = build_kb_index,
        source_dir_for: Callable[[str], str] | None = None,
        max_workers: int = 1,
    ):
        self._ingest_fn = ingest_fn
        self._source_dir_for = source_dir_for or get_settings().kb_source_dir
        self._jobs: dict[str, dict] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cogdoc-ingest"
        )

    def submit(self, kb_id: str) -> dict:
        job_id = uuid4().hex
        record = {
            "job_id": job_id,
            "kb_id": kb_id,
            "status": "pending",
            "created_at": _now_iso(),
            "finished_at": None,
            "document_count": None,
            "chunk_count": None,
            "error_code": None,
            "message": None,
        }
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run, job_id)
        return dict(record)

    def _run(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            kb_id = self._jobs[job_id]["kb_id"]
        try:
            result = self._ingest_fn(kb_id, self._source_dir_for(kb_id))
            with self._lock:
                self._jobs[job_id].update(
                    status="succeeded",
                    document_count=result.document_count,
                    chunk_count=result.chunk_count,
                    finished_at=_now_iso(),
                )
            log_event(
                "ingest",
                "index_job_succeeded",
                {"trace_id": job_id},
                kb_id=kb_id,
                document_count=result.document_count,
            )
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(
                    status="failed",
                    error_code="INGEST_FAILED",
                    message=str(exc),
                    finished_at=_now_iso(),
                )
            log_event(
                "ingest",
                "index_job_failed",
                {"trace_id": job_id},
                level=logging.ERROR,
                kb_id=kb_id,
                error_class=type(exc).__name__,
            )

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
