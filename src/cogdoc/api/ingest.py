import inspect
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Callable
from uuid import uuid4
from cogdoc.api.persistence import InMemoryJobStore
from cogdoc.config.settings import get_settings
from cogdoc.observability.logger import log_event
from cogdoc.service.ingest_service import KBCleanupError, build_kb_index_transactional
from cogdoc.service.kb_epoch import shared_epoch_store
from cogdoc.service.kb_lifecycle import LIFECYCLE_ACTIVE, shared_lifecycle_store
from cogdoc.service.mutation_journal import shared_mutation_journal


# 返回当前 UTC 时间字符串。
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 完成 silentremove 处理。
def _silent_remove(path: str) -> bool:
    # 删除文件，成功或本就不存在返回 True；删除失败返回 False，供调用方据此保留 journal 重试。
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


# 表示 KBExistsError 异常。
class KBExistsError(Exception):
    pass


# registry 损坏：宁可拒绝启动也不退回空表，否则现存 KB 全部消失、同名重建会复用旧 source/state/index。
class RegistryCorruptError(Exception):
    pass


# 知识库元数据的 JSON 注册表；source/chroma/bm25/manifest 仍按 kb_id 物理隔离。
class KnowledgeBaseRegistry:
    # 知识库元数据的 JSON 注册表；source/chroma/bm25/manifest 仍按 kb_id 物理隔离。
    def __init__(
        self,
        registry_path: str | None = None,
        source_dir_for: Callable[[str], str] | None = None,
    ):
        settings = get_settings()
        self._path = registry_path or settings.kb_registry_path
        self._degraded_path = f"{self._path}.degraded"
        self._source_dir_for = source_dir_for or settings.kb_source_dir
        self._lock = RLock()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._entries = self._load()

    # 加载。
    def _load(self) -> dict:
        # 文件不存在=全新系统，空表。损坏（语法/结构）则隔离原文件并抛错 fail-closed，绝不退回空表。
        if os.path.exists(self._degraded_path):
            raise RegistryCorruptError(
                f"registry 处于 degraded 状态，需人工恢复: {self._path}"
            )
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            self._quarantine_corrupt()
            raise RegistryCorruptError(f"registry 损坏已隔离: {self._path}")
        if not isinstance(data, dict) or any(
            not isinstance(k, str) or not isinstance(v, dict) or v.get("kb_id") != k
            for k, v in data.items()
        ):
            self._quarantine_corrupt()
            raise RegistryCorruptError(f"registry 顶层非 dict 已隔离: {self._path}")
        return data

    # 隔离损坏文件。
    def _quarantine_corrupt(self) -> None:
        try:
            os.replace(self._path, f"{self._path}.corrupt-{time.time_ns()}")
        except OSError:
            pass
        try:
            with open(self._degraded_path, "w", encoding="utf-8") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass

    # 保存 entries。
    def _save_entries(self, entries: dict) -> None:
        # 原子写候选表：先写临时文件再 rename，避免中途崩溃留下半截 JSON。
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    # 返回目录。
    def source_dir(self, kb_id: str) -> str:
        return self._source_dir_for(kb_id)

    # 创建。
    def create(self, kb_id: str) -> dict:
        with self._lock:
            if kb_id in self._entries:
                raise KBExistsError(kb_id)
            # 新 incarnation：epoch 自增，令删库前在飞、捕获旧 epoch 的任务在重建后仍被守卫拦下。
            shared_epoch_store().bump(kb_id)
            os.makedirs(self._source_dir_for(kb_id), exist_ok=True)
            # tenant_id/owner_id 现填默认值，为未来多租户隔离预留。
            record = {
                "kb_id": kb_id,
                "created_at": _now_iso(),
                "tenant_id": "default",
                "owner_id": "default",
            }
            # registry 持久化是提交点：先写盘成功再更新内存。提交前 lifecycle 仍是旧态（如 deleted→读被拦）， 故"registry 已存在但 lifecycle 未 active"是 fail-closed，不会出现"registry 缺失但可读旧数据"。
            candidate = {**self._entries, kb_id: record}
            self._save_entries(candidate)
            self._entries = candidate
            # 提交后切 active，清除同名 KB 的 deleted tombstone，恢复读写。
            try:
                shared_lifecycle_store().set(kb_id, LIFECYCLE_ACTIVE)
            except Exception:
                # 先清目录、再撤 registry。目录清理失败时保留 registry 记录，让调用方可显式 DELETE 重试， 不能移除记录后让同名 create 复用半创建目录。
                kb_dir = os.path.dirname(self._source_dir_for(kb_id))
                try:
                    shutil.rmtree(kb_dir)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_exc:
                    raise KBCleanupError(
                        f"KB 创建 finalize 失败且目录补偿失败: {kb_dir}"
                    ) from cleanup_exc
                candidate = {k: v for k, v in self._entries.items() if k != kb_id}
                self._save_entries(candidate)
                self._entries = candidate
                raise
            return dict(record)

    # 检查存在性。
    def exists(self, kb_id: str) -> bool:
        with self._lock:
            return kb_id in self._entries

    # 返回结果。
    def get(self, kb_id: str) -> dict | None:
        with self._lock:
            record = self._entries.get(kb_id)
            return dict(record) if record else None

    # 列出。
    def list(self) -> list[dict]:
        with self._lock:
            return [dict(record) for record in self._entries.values()]

    # 删除。
    def delete(self, kb_id: str) -> bool:
        # 先删源目录，成功后才从 registry 移除：目录删失败时 registry 仍保留该 KB，DELETE 可重试不返回 404。
        with self._lock:
            if kb_id not in self._entries:
                return False
            kb_dir = os.path.dirname(self._source_dir_for(kb_id))
            try:
                shutil.rmtree(kb_dir)
            except FileNotFoundError:
                pass  # 已删或上次删一半：幂等放过，继续移除 registry 记录
            except OSError as exc:
                raise KBCleanupError(f"KB 目录删除失败: {kb_dir}") from exc
            # 目录已清，再原子写出不含该 KB 的候选表并更新内存。
            candidate = {k: v for k, v in self._entries.items() if k != kb_id}
            self._save_entries(candidate)
            self._entries = candidate
            return True


_MAX_KB_EXECUTORS = 256  # 防止持续创建/删库积累无界线程对象
_BAK_SUFFIX = ".cogdoc-bak"  # 源文件回滚备份后缀；不以 .pdf 结尾故不被索引扫描


# 每个 kb_id 独享一个单线程 executor：不同 KB 并发构建，同 KB 内 mutation + 构建全部串行。
class IndexJobManager:
    # 每个 kb_id 独享一个单线程 executor：不同 KB 并发构建，同 KB 内 mutation + 构建全部串行。
    def __init__(
        self,
        ingest_fn: Callable[[str, str], object] = build_kb_index_transactional,
        source_dir_for: Callable[[str], str] | None = None,
        job_store: object | None = None,
        kb_exists: "Callable[[str], bool] | None" = None,
        journal: object | None = None,
    ):
        self._ingest_fn = ingest_fn
        self._source_dir_for = source_dir_for or get_settings().kb_source_dir
        self._store = job_store or InMemoryJobStore()
        self._kb_exists = kb_exists  # 防复活：KB 已删未重建时拒绝陈旧 mutation
        self._journal = (
            journal or shared_mutation_journal()
        )  # 源文件 mutation 崩溃恢复日志
        # ingest_fn 是否支持 on_commit：支持则把 journal 提交点贴到索引提交点（switch_active）。
        try:
            self._ingest_takes_on_commit = (
                "on_commit" in inspect.signature(ingest_fn).parameters
            )
        except (TypeError, ValueError):
            self._ingest_takes_on_commit = False
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._retired_executors: set[ThreadPoolExecutor] = set()
        self._inflight: dict[
            str, int
        ] = {}  # 每 KB 在途命令数，0 且久未活动才可淘汰 executor
        self._last_active: dict[str, float] = {}
        self._ex_lock = Lock()
        self._closed = False

    # 返回执行器locked。
    def _get_executor_locked(self, kb_id: str) -> ThreadPoolExecutor:
        # 调用方必须已持 _ex_lock；与 release_executor/shutdown 互斥。
        if self._closed:
            raise RuntimeError("IndexJobManager is closed")
        self._prune_retired_locked()
        ex = self._executors.get(kb_id)
        if ex is None:
            live_retired = sum(
                any(thread.is_alive() for thread in getattr(retired, "_threads", ()))
                for retired in self._retired_executors
            )
            if len(self._executors) + live_retired >= _MAX_KB_EXECUTORS:
                raise RuntimeError(
                    f"per-KB executor 数量已达上限 {_MAX_KB_EXECUTORS}，拒绝新 KB"
                )
            ex = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"cogdoc-kb-{kb_id[:8]}"
            )
            self._executors[kb_id] = ex
        return ex

    # 完成 prune已退役执行器locked 处理。
    def _prune_retired_locked(self) -> None:
        # shutdown(wait=False) 后 executor 对象仍持有 Thread 引用；线程全部退出后即可丢弃句柄。
        self._retired_executors = {
            ex
            for ex in self._retired_executors
            if any(thread.is_alive() for thread in getattr(ex, "_threads", ()))
        }

    # 返回执行器。
    def _get_executor(self, kb_id: str) -> ThreadPoolExecutor:
        with self._ex_lock:
            return self._get_executor_locked(kb_id)

    # 新建记录。
    def _new_record(self, kb_id: str) -> dict:
        return {
            "job_id": uuid4().hex,
            "kb_id": kb_id,
            "status": "pending",
            "created_at": _now_iso(),
            "finished_at": None,
            "document_count": None,
            "chunk_count": None,
            "error_code": None,
            "message": None,
        }

    # 记录失败任务。
    def _fail_job(
        self, job_id: str, kb_id: str, exc: Exception, error_code: str = "INGEST_FAILED"
    ) -> None:
        self._store.update(
            job_id,
            status="failed",
            error_code=error_code,
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

    # 完成 safe恢复状态 处理。
    def _safe_restore(self, src: str, dst: str, job_id: str, kb_id: str) -> bool:
        # 回滚恢复源文件，成功返回 True。失败不外逃但返回 False：调用方据此保留 journal 供恢复重试。
        try:
            os.replace(src, dst)
            return True
        except OSError as exc:
            try:
                self._store.update(
                    job_id, message=f"回滚失败，源目录与索引不一致: {exc}"
                )
            except Exception:
                pass
            log_event(
                "ingest",
                "index_rollback_failed",
                {"trace_id": job_id},
                level=logging.ERROR,
                kb_id=kb_id,
                error_class=type(exc).__name__,
            )
            return False

    # 回滚 upload。
    def _rollback_upload(self, job_id, kb_id, dest, backup, had_old) -> bool:
        # 把源目录恢复到上传前状态，返回是否恢复成功（失败则调用方保留 journal 供启动重试）。
        if had_old:
            if os.path.exists(backup):
                return self._safe_restore(backup, dest, job_id, kb_id)  # 恢复旧文件
            return True  # 备份未生成（replace 前失败），dest 仍是原文件
        return _silent_remove(dest)  # 新增上传：删除残缺/未提交的新文件

    # 结束回滚。
    def _finish_rollback(self, job_id: str) -> None:
        # 进程内已确认磁盘恢复到一致态：先 best-effort 写 rolled_back 终态（供 clear 失败时的崩溃恢复）， 再无条件尝试清除条目。即便标记失败，只要清除成功就不会留下 source_moved 条目阻塞下次启动。
        self._journal.mark_rolled_back(job_id)
        self._journal.clear(job_id)

    # 准备提交。
    def _prepare_commit(self, job_id: str, gen_id: str) -> None:
        # 两份证据都在 switch_active 前写入；任一步失败都会中止提交并由外层回滚源文件。
        self._journal.record_generation(job_id, gen_id)
        self._store.update(job_id, committed_generation_id=gen_id)
        stored = self._store.get(job_id)
        if stored is None or stored.get("committed_generation_id") != gen_id:
            raise RuntimeError(f"job {job_id} 的 generation 提交证据未持久化")

    # 拒绝ifunresolved。
    def _reject_if_unresolved(self, job_id: str, kb_id: str) -> bool:
        if not self._journal.has_entries(kb_id):
            return False
        self._fail_job(
            job_id,
            kb_id,
            RuntimeError(f"KB {kb_id} 存在未恢复 mutation journal，拒绝继续写入"),
        )
        return True

    # 提交tracked。
    def _submit_tracked(self, ex, kb_id: str, fn: Callable, *args):
        # 调用方须已持 _ex_lock。包一层计数：在途归零且久未活动时 sweeper 才可淘汰该 executor。
        self._inflight[kb_id] = self._inflight.get(kb_id, 0) + 1
        self._last_active[kb_id] = time.time()

        # 执行后台任务并完成收尾。
        def runner():
            try:
                return fn(*args)
            finally:
                with self._ex_lock:
                    # executor 已被 release（如删库内自释放）则不再回写，避免遗留陈旧计数。
                    if self._executors.get(kb_id) is ex:
                        self._inflight[kb_id] = max(0, self._inflight.get(kb_id, 1) - 1)
                        self._last_active[kb_id] = time.time()

        return ex.submit(runner)

    # 入队。
    def _enqueue(self, kb_id: str, fn: Callable, *args) -> dict:
        # get-create-submit 全程持锁与 release_executor 互斥：失败不留 pending，入队成功则 ex 必存活。
        with self._ex_lock:
            ex = self._get_executor_locked(kb_id)
            record = self._new_record(kb_id)
            base_epoch = shared_epoch_store().current(kb_id)  # 执行期错配守卫基线
            self._store.create(record)
            try:
                self._submit_tracked(
                    ex, kb_id, fn, record["job_id"], kb_id, base_epoch, *args
                )
            except Exception as exc:
                # 线程创建失败/资源耗尽：record 已建，标记失败而非遗留 pending。
                self._inflight[kb_id] = max(0, self._inflight.get(kb_id, 1) - 1)
                self._fail_job(record["job_id"], kb_id, exc)
                raise
        return dict(record)

    # 提交结果。
    def submit(self, kb_id: str) -> dict:
        # 向后兼容：仅触发索引，不含文件 mutation（文件变更已在调用方完成）。
        return self._enqueue(kb_id, self._run)

    # 提交上传。
    def submit_upload(
        self, kb_id: str, source_dir: str, filename: str, content: bytes
    ) -> dict:
        # 写文件与构建索引作为一个 executor command：保证每个 job 快照与其 mutation 精确对应。
        return self._enqueue(kb_id, self._run_with_write, source_dir, filename, content)

    # 提交删除文档。
    def submit_delete_doc(self, kb_id: str, path: str) -> dict:
        # 存在性检查在 executor command 内进行，保证与上传队列有序：upload 排在前则文件已落盘。
        return self._enqueue(kb_id, self._run_with_delete_doc, path)

    # 运行blocking。
    def run_blocking(self, kb_id: str, fn: Callable, *args) -> object:
        # 同 KB executor 线程内调用会单线程自等待死锁，运行时直接拒绝而非仅靠注释约束。
        if threading.current_thread().name.startswith(f"cogdoc-kb-{kb_id[:8]}"):
            raise RuntimeError(f"run_blocking 不可从 KB {kb_id} 自身 executor 线程调用")
        with self._ex_lock:
            ex = self._get_executor_locked(kb_id)
            fut = self._submit_tracked(ex, kb_id, fn, *args)
        return fut.result()

    # 释放执行器。
    def release_executor(self, kb_id: str) -> None:
        # 释放槽位防上限耗尽；排队的陈旧命令仍跑但被 epoch/exists 守卫拦下，不写已删或重建的 KB。
        with self._ex_lock:
            ex = self._executors.pop(kb_id, None)
            self._inflight.pop(kb_id, None)
            self._last_active.pop(kb_id, None)
            if ex is not None:
                self._retired_executors.add(ex)
        if ex is not None:
            ex.shutdown(wait=False)

    # 淘汰空闲执行器。
    def evict_idle(
        self, max_idle_seconds: float, now: float | None = None
    ) -> list[str]:
        # sweeper 调用：淘汰在途归零且超过 max_idle_seconds 未活动的 executor，回收 #13 的活跃 KB 上限。
        now = now if now is not None else time.time()
        evicted = []
        with self._ex_lock:
            self._prune_retired_locked()
            for kb_id in list(self._executors):
                if self._inflight.get(kb_id, 0) != 0:
                    continue
                if now - self._last_active.get(kb_id, 0.0) <= max_idle_seconds:
                    continue
                ex = self._executors.pop(kb_id)
                self._inflight.pop(kb_id, None)
                self._last_active.pop(kb_id, None)
                evicted.append((kb_id, ex))
                self._retired_executors.add(ex)
        for _, ex in evicted:
            ex.shutdown(wait=False)
        return [kb_id for kb_id, _ in evicted]

    # ---- executor commands ----

    # _stale：处理对应功能。
    def _stale(self, kb_id: str, base_epoch: int) -> bool:
        # epoch 变更 = KB 已删（可能已重建）；exists=False = 已删未重建；非 active = 删库进行中，禁新 mutation。 epoch/lifecycle 读损坏抛错时 fail-closed 视为 stale：拦掉 mutation 而非把损坏当 epoch 0。
        try:
            if shared_epoch_store().current(kb_id) != base_epoch:
                return True
        except Exception:
            return True
        if self._kb_exists is not None and not self._kb_exists(kb_id):
            return True
        try:
            if shared_lifecycle_store().status(kb_id) != LIFECYCLE_ACTIVE:
                return True
        except Exception:
            return True
        return False

    # 运行结果。
    def _run(self, job_id: str, kb_id: str, base_epoch: int) -> None:
        if self._store.get(job_id) is None:
            return
        if self._stale(kb_id, base_epoch):
            self._fail_job(
                job_id, kb_id, RuntimeError(f"KB {kb_id} 已被删除或重建，构建取消")
            )
            return
        if self._reject_if_unresolved(job_id, kb_id):
            return
        self._run_ingest(job_id, kb_id, self._source_dir_for(kb_id))

    # 运行with写入。
    def _run_with_write(
        self,
        job_id: str,
        kb_id: str,
        base_epoch: int,
        source_dir: str,
        filename: str,
        content: bytes,
    ) -> None:
        if self._store.get(job_id) is None:
            return
        if self._stale(kb_id, base_epoch):
            self._fail_job(
                job_id, kb_id, RuntimeError(f"KB {kb_id} 已被删除或重建，上传取消")
            )
            return
        if self._reject_if_unresolved(job_id, kb_id):
            return
        dest = os.path.join(source_dir, filename)
        backup = f"{dest}.{job_id}{_BAK_SUFFIX}"  # 唯一名，绝不覆盖上次崩溃遗留的备份
        had_old = os.path.exists(dest)
        try:
            os.makedirs(source_dir, exist_ok=True)
            # 先写 journal 再动文件：崩溃在任意点都能据 journal 恢复。
            self._journal.begin_upload(job_id, kb_id, dest, backup, had_old)
            if had_old:
                os.replace(dest, backup)  # 覆盖前备份旧文件，构建失败可回滚
                self._journal.mark_source_moved(job_id)
            with open(dest, "wb") as f:
                f.write(content)
        except Exception as exc:
            # 写入中途失败：恢复到上传前状态，仅当磁盘一致才清 journal，否则保留供启动恢复。
            if self._rollback_upload(job_id, kb_id, dest, backup, had_old):
                self._finish_rollback(job_id)
            self._fail_job(job_id, kb_id, exc)
            return
        ok = self._run_ingest(
            job_id,
            kb_id,
            source_dir,
            on_commit=lambda gid: self._prepare_commit(job_id, gid),
        )
        if ok:
            # 已提交：先打不可逆 committed 标记，再 best-effort 清备份，最后无条件清 journal。 不能因备份清理失败而保留 journal——否则后续切代/删库会让它被误判未提交而回滚已提交源文件。
            committed_marked = self._journal.mark_committed(job_id)
            if had_old:
                _silent_remove(backup)  # 孤儿备份无害（不被索引扫描）
            if committed_marked:
                self._journal.clear(job_id)
            else:
                log_event(
                    "ingest",
                    "mutation_journal_commit_mark_failed",
                    {"trace_id": job_id},
                    level=logging.ERROR,
                    kb_id=kb_id,
                )
        elif self._rollback_upload(job_id, kb_id, dest, backup, had_old):
            self._finish_rollback(job_id)

    # 运行with删除文档。
    def _run_with_delete_doc(
        self, job_id: str, kb_id: str, base_epoch: int, path: str
    ) -> None:
        if self._store.get(job_id) is None:
            return
        if self._stale(kb_id, base_epoch):
            self._fail_job(
                job_id, kb_id, RuntimeError(f"KB {kb_id} 已被删除或重建，删除取消")
            )
            return
        if self._reject_if_unresolved(job_id, kb_id):
            return
        if not os.path.exists(path):
            self._fail_job(
                job_id,
                kb_id,
                FileNotFoundError(f"文档不存在: {os.path.basename(path)}"),
                error_code="DOCUMENT_NOT_FOUND",
            )
            return
        quarantine = f"{path}.{job_id}{_BAK_SUFFIX}"  # 唯一名，避免覆盖遗留备份
        try:
            self._journal.begin_delete(job_id, kb_id, path, quarantine)
            os.replace(path, quarantine)  # 移入隔离区而非直接删除，构建失败可恢复
            self._journal.mark_source_moved(job_id)
        except Exception as exc:
            if os.path.exists(quarantine):
                restored = self._safe_restore(quarantine, path, job_id, kb_id)
            else:
                restored = os.path.exists(path)  # replace 前失败，原文件仍在
            if restored:
                self._finish_rollback(job_id)
            self._fail_job(job_id, kb_id, exc)
            return
        ok = self._run_ingest(
            job_id,
            kb_id,
            os.path.dirname(path),
            on_commit=lambda gid: self._prepare_commit(job_id, gid),
        )
        if ok:
            committed_marked = self._journal.mark_committed(job_id)
            _silent_remove(quarantine)  # 孤儿隔离文件无害，best-effort
            if committed_marked:
                self._journal.clear(job_id)
            else:
                log_event(
                    "ingest",
                    "mutation_journal_commit_mark_failed",
                    {"trace_id": job_id},
                    level=logging.ERROR,
                    kb_id=kb_id,
                )
        elif self._safe_restore(quarantine, path, job_id, kb_id):
            self._finish_rollback(job_id)

    # 运行ingest。
    def _run_ingest(self, job_id, kb_id, source_dir, on_commit=None) -> bool:
        try:
            self._store.update(job_id, status="running")
        except Exception:
            pass
        try:
            if self._ingest_takes_on_commit:
                # 提交点贴死 switch_active：build 在提交前用 gen_id 回调 record_generation。
                result = self._ingest_fn(kb_id, source_dir, on_commit=on_commit)
            else:
                result = self._ingest_fn(
                    kb_id, source_dir
                )  # 不支持回调的旧 fn：无 journal gen 记录
        except Exception as exc:
            # 构建未提交（active 仍是旧代）：返回 False 触发源文件回滚。
            self._fail_job(job_id, kb_id, exc)
            return False
        # 索引已提交：终态状态写入做退避重试（缓解 SQLite 瞬时锁），仍失败则记 error，不回滚已生效源文件。
        last_exc = None
        for attempt in range(4):
            try:
                self._store.update(
                    job_id,
                    status="succeeded",
                    document_count=result.document_count,
                    chunk_count=result.chunk_count,
                    finished_at=_now_iso(),
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(
                        0.1 * (2**attempt)
                    )  # 0.1/0.2/0.4s 退避，给长事务锁释放窗口
        if last_exc is None:
            log_event(
                "ingest",
                "index_job_succeeded",
                {"trace_id": job_id},
                kb_id=kb_id,
                document_count=result.document_count,
            )
        else:
            # 任务实际已成功，状态持久化反复失败：记 error 供运维介入，避免长期停在 running 而无痕。
            log_event(
                "ingest",
                "index_job_commit_record_failed",
                {"trace_id": job_id},
                level=logging.ERROR,
                kb_id=kb_id,
                error_class=type(last_exc).__name__,
            )
        return True

    # 判断 busy 是否成立。
    def is_busy(self, kb_id: str) -> bool:
        # 该 KB 是否有在途命令；sweeper 据此避免重复排队同一个重建任务。
        with self._ex_lock:
            return self._inflight.get(kb_id, 0) > 0

    # 返回结果。
    def get(self, job_id: str) -> dict | None:
        return self._store.get(job_id)

    # 协调孤儿任务。
    def reconcile_orphans(self) -> None:
        reconcile = getattr(self._store, "reconcile_orphans", None)
        if callable(reconcile):
            reconcile()

    # 完成 shutdown 处理。
    def shutdown(self, wait: bool = True) -> None:
        with self._ex_lock:
            self._closed = True
            executors = list(self._executors.values())
            executors.extend(self._retired_executors)
            self._executors.clear()
            self._retired_executors.clear()
            self._inflight.clear()
            self._last_active.clear()
        # 锁外排空：wait=True 等在途 mutation 跑完再返回，保证 lifespan 释放进程锁前无后台写线程。 不持 _ex_lock 等待，否则 runner finally 取 _ex_lock 会与之死锁。
        for ex in executors:
            ex.shutdown(wait=wait)
