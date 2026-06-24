import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from config.settings import get_settings
from graph.subgraphs.qa import RetrieverFactory
from observability.logger import log_event
from service.kb_epoch import shared_epoch_store
from service.kb_lifecycle import (
    LIFECYCLE_DELETED,
    LIFECYCLE_DELETING,
    shared_lifecycle_store,
)
from service.kb_locks import kb_write_lock
from service.kb_readers import has_readers
from service.purge_queue import shared_purge_queue
from service.kb_state import KBState, StaleGenerationError
from tools.chunk_identity import CHUNK_IDENTITY_VERSION, build_chunk_id
from tools.chunker import chunk_paper
from tools.manifest import (
    load_index_manifest,
    manifest_path,
    save_index_manifest,
    stamp_chunk_identity_contract,
)
from tools.embedder import Embedder
from tools.parser import PARSER_VERSION, smart_parse
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.hybrid import HybridRetriever
from tools.retriever.vector_retriever import VectorRetriever
from tools.rust_core_loader import ensure_rust_core
from tools.tokenizer import TOKENIZER_VERSION


# 增量复用门控：任一构建组件版本变化都使旧索引不可复用，强制全量重建。
INDEX_BUILD_VERSION = (
    f"{CHUNK_IDENTITY_VERSION}"
    f"|parser={PARSER_VERSION}"
    f"|tokenizer={TOKENIZER_VERSION}"
    f"|embedder={Embedder.EMBEDDING_CONTRACT_VERSION}"
)


# 处理 stamp index build version 相关逻辑。
def stamp_index_build_version(manifest: dict) -> dict:
    # 写入当前构建版本；run.py 启动检查与入库共用，保证两处门控一致。
    manifest["index_build_version"] = INDEX_BUILD_VERSION
    return manifest


# 封装 IngestDocResult 的状态与行为。
@dataclass(frozen=True)
class IngestDocResult:
    name: str
    chunk_count: int


# 封装 IngestResult 的状态与行为。
@dataclass(frozen=True)
class IngestResult:
    kb_id: str
    document_count: int
    chunk_count: int
    documents: list[IngestDocResult] = field(default_factory=list)


# 封装 IndexInconsistencyError 的状态与行为。
class IndexInconsistencyError(Exception):
    # 写后两路索引不一致（如向量清理静默失败、部分写）：标记入库失败而非误报成功。
    pass


# 封装 KBCleanupError 的状态与行为。
class KBCleanupError(Exception):
    # 删库时部分代资源清理失败：manifest 保留以支持调用方重试，避免孤儿 Chroma/BM25 数据丢失 GC 记录。
    pass


# 封装 IncrementalPlan 的状态与行为。
@dataclass(frozen=True)
class IncrementalPlan:
    # 需要重新解析的文档名（新增+内容改变）与需要从索引删除的文件名（删除+改变）。
    to_parse: list[str]
    removed_sources: set[str]


# 列出 list pdf files 相关逻辑。
def list_pdf_files(source_dir: str) -> list[str]:
    if not os.path.isdir(source_dir):
        return []
    return sorted(f for f in os.listdir(source_dir) if f.lower().endswith(".pdf"))


# 使缓存失效 invalidate engine cache 相关逻辑。
def _invalidate_engine_cache(kb_id: str) -> None:
    # 只失效本库引擎，否则 /chat 命中旧引擎读旧索引；不波及其他 kb。
    RetrieverFactory.invalidate(kb_id)


# 移除 remove manifest 相关逻辑。
def _remove_manifest(kb_id: str) -> None:
    manifest_file = manifest_path(kb_id)
    if os.path.exists(manifest_file):
        os.remove(manifest_file)


# 删除 delete kb index 相关逻辑。
def delete_kb_index(kb_id: str) -> None:
    # 删库索引：清向量/BM25 + 删 manifest + 失效引擎缓存。clear 失败不删 manifest，避免旧文档残留泄露。 取 KB 写锁，与正在运行/排队的入库任务串行，避免删库后任务又把索引/manifest 写回去。
    with kb_write_lock(kb_id):
        # 先 bump epoch（tombstone）：删库前在飞的构建任务切换 active 时会因 epoch 不符被拒。
        shared_epoch_store().bump(kb_id)
        try:
            RetrieverFactory.get_engine(kb_id).clear()
        except Exception:
            _invalidate_engine_cache(kb_id)
            raise
        _remove_manifest(kb_id)
        _invalidate_engine_cache(kb_id)


# 清理 purge generation external 相关逻辑。
def _purge_generation_external(kb_id: str, gen_id: str) -> None:
    # 删库专用：只清理 KB 目录外的 Chroma 集合与 BM25 pkl；state.json 与 gen 快照随 KB 目录整体删除。 不调用 remove_generation（其禁止删 active），避免正常非空 KB 删库被误判失败。
    if has_readers(kb_id):
        raise KBCleanupError(f"KB {kb_id} 仍有在途读者，延后清理 generation {gen_id}")
    settings = get_settings()
    collection_id = settings.kb_collection_id(kb_id, gen_id)
    ok = True
    try:
        import chromadb

        chromadb.PersistentClient(path=settings.chroma_persist_dir).delete_collection(
            f"col-{collection_id}"
        )
    except ValueError:
        pass
    except Exception:
        ok = False
    bm25_path = os.path.join(settings.bm25_persist_dir, f"bm25_{collection_id}.pkl")
    try:
        os.remove(bm25_path)
    except FileNotFoundError:
        pass
    except Exception:
        ok = False
    if not ok:
        raise KBCleanupError(f"generation {gen_id} 外部资源未清理")


# 后台 daemon Timer 注册表：统一在进程关闭时取消，避免释放进程锁后旧线程仍操作索引。
_active_timers: set = set()
_timers_lock = threading.Lock()


# 启动 start tracked timer 相关逻辑。
def _start_tracked_timer(delay: float, fn, args=()) -> None:
    # 执行后台任务并完成收尾。
    def runner():
        try:
            fn(*args)
        finally:
            with _timers_lock:
                _active_timers.discard(t)

    t = threading.Timer(delay, runner)
    t.daemon = True
    with _timers_lock:
        _active_timers.add(t)
    try:
        t.start()
    except Exception:
        with _timers_lock:
            _active_timers.discard(t)
        raise


# 处理 cancel all timers 相关逻辑。
def cancel_all_timers(join_timeout: float | None = 30.0) -> bool:
    # 关闭期调用：取消未触发的 Timer，并有界 join 已进入执行的 runner。 返回是否全部排空（无存活线程）。默认 30s 上界：正常清理能跑完，又不让卡死清理永久挂起 shutdown。 未排空时调用方不应显式释放进程锁——留给进程退出由 OS 释放，保证不会"锁已放但旧线程仍在写"。
    with _timers_lock:
        timers = list(_active_timers)
        _active_timers.clear()
    for t in timers:
        t.cancel()  # 未启动则取消；已在跑则 cancel 无效，靠下面 join 等其结束
    for t in timers:
        if t.ident is not None:
            t.join(timeout=join_timeout)
    return not any(t.is_alive() for t in timers)


# 处理 drain purge queue 相关逻辑。
def drain_purge_queue(now: float | None = None) -> int:
    # 重试持久化 purge 队列中已过 grace period 的外部资源清理；成功才出队。sweeper 与启动时调用。
    queue = shared_purge_queue()
    done = 0
    for item in queue.due(now):
        try:
            _purge_generation_external(item["kb_id"], item["gen_id"])
            queue.remove(item["kb_id"], item["gen_id"])
            done += 1
        except Exception:
            pass  # 失败保留条目，下一轮 sweeper 重试
    return done


# 处理 schedule kb purge 相关逻辑。
def _schedule_kb_purge(kb_id: str, gen_ids: list) -> None:
    # 物理清理入持久队列（带 grace period），并起一个 Timer 促其尽快执行；失败/退出由 sweeper 兜底重试。
    not_before = time.time() + GENERATION_CLEANUP_DELAY_SECONDS
    for gen_id in gen_ids:
        shared_purge_queue().add(kb_id, gen_id, not_before)
    try:
        _start_tracked_timer(GENERATION_CLEANUP_DELAY_SECONDS, drain_purge_queue)
    except Exception as exc:
        # 队列已持久化，Timer 仅是低延迟优化；sweeper/下次启动仍会可靠重试。
        log_event(
            "purge",
            "purge_timer_start_failed",
            {},
            level=logging.ERROR,
            kb_id=kb_id,
            error_class=type(exc).__name__,
        )


# 删除 delete kb index transactional 相关逻辑。
def delete_kb_index_transactional(kb_id: str) -> None:
    # 逻辑删库同步完成：deleting 门控读路径与新 mutation、bump epoch、删 manifest、失效缓存。 Chroma/BM25 物理清理入持久队列、延迟 grace period 后执行，避免删掉在途检索正持有的索引。
    with kb_write_lock(kb_id):
        shared_lifecycle_store().set(kb_id, LIFECYCLE_DELETING)
        shared_epoch_store().bump(kb_id)
        gen_ids = KBState(kb_id).generation_ids()
        _remove_manifest(kb_id)
        RetrieverFactory.invalidate(kb_id)
        _schedule_kb_purge(kb_id, gen_ids)


# 标记 mark kb deleted 相关逻辑。
def mark_kb_deleted(kb_id: str) -> None:
    # 删库全流程（含 registry 删除）成功后落 deleted tombstone，防旧任务复活读写。
    shared_lifecycle_store().set(kb_id, LIFECYCLE_DELETED)


# 处理 documents by name 相关逻辑。
def _documents_by_name(manifest: dict) -> dict[str, str]:
    return {doc["name"]: doc["sha256"] for doc in manifest.get("documents", [])}


# 规划 plan incremental 相关逻辑。
def plan_incremental(previous: dict, current: dict) -> IncrementalPlan | None:
    # 无上一版、库标识或分块身份契约版本变化时返回 None，交由全量重建。
    if not previous:
        return None
    if previous.get("doc_id") != current.get("doc_id"):
        return None
    # index_build_version 已含 chunk 身份版本，并覆盖解析器/分词器版本；任一变化都禁止复用。
    if previous.get("index_build_version") != current.get("index_build_version"):
        return None

    prev = _documents_by_name(previous)
    cur = _documents_by_name(current)
    added = [name for name in cur if name not in prev]
    changed = [name for name in cur if name in prev and prev[name] != cur[name]]
    removed = [name for name in prev if name not in cur]

    # 按文件名（文档身份）删除：删除+改变的文档清旧 chunk。文件名唯一，同内容不同名互不影响。
    removed_sources = set(removed) | set(changed)
    return IncrementalPlan(sorted(added + changed), removed_sources)


# 解析 parse and chunk 相关逻辑。
def _parse_and_chunk(
    source_dir: str,
    names: list[str],
    source_hash_by_name: dict[str, str],
    start_index: int = 0,
) -> tuple[list, list[IngestDocResult]]:
    all_chunks = []
    next_chunk_index = start_index
    doc_results = []
    for pdf in names:
        pages = smart_parse(os.path.join(source_dir, pdf))
        chunks = chunk_paper(pages, source_sha256=source_hash_by_name[pdf])
        for chunk in chunks:
            # chunk_index 仅用于展示，chunk_id 才是身份键。
            chunk["meta"]["chunk_index"] = next_chunk_index
            next_chunk_index += 1
        all_chunks.extend(chunks)
        doc_results.append(IngestDocResult(pdf, len(chunks)))
    return all_chunks, doc_results


# 校验 verify consistent 相关逻辑。
def _verify_consistent(engine) -> None:
    # 写后校验两路 chunk_id 一致：识破静默的清理失败/部分写，避免残留旧块却报成功。
    if not engine.is_consistent():
        raise IndexInconsistencyError("index stores inconsistent after write")


# 处理 full rebuild 相关逻辑。
def _full_rebuild(
    engine, kb_id, source_dir, pdf_files, manifest, source_hash_by_name
) -> IngestResult:
    all_chunks, doc_results = _parse_and_chunk(
        source_dir, pdf_files, source_hash_by_name
    )
    try:
        if all_chunks:
            engine.index(all_chunks)
            _verify_consistent(engine)
        else:
            # 有 PDF 但没抽出任何 chunk（扫描件/空 PDF）：index([]) 会早退不清，必须显式清旧索引。
            engine.clear()
    except Exception:
        # 失败也驱逐被破坏的缓存引擎，否则 /chat 继续读半更新索引；manifest 未保存，下次入库自愈。
        _invalidate_engine_cache(kb_id)
        raise
    save_index_manifest(manifest)
    _invalidate_engine_cache(kb_id)
    return IngestResult(kb_id, len(pdf_files), len(all_chunks), doc_results)


# 处理 incremental apply 相关逻辑。
def _incremental_apply(
    engine, kb_id, source_dir, pdf_files, manifest, plan, source_hash_by_name
) -> IngestResult:
    # 新块从现存最大编号续号：保证唯一、单调，且不重编未变文档的展示编号。
    new_chunks, doc_results = _parse_and_chunk(
        source_dir,
        plan.to_parse,
        source_hash_by_name,
        start_index=engine.max_chunk_index() + 1,
    )
    try:
        if plan.to_parse or plan.removed_sources:
            engine.upsert_documents(new_chunks, plan.removed_sources)
            _verify_consistent(engine)
    except Exception:
        # 半更新（已删向量但嵌入/BM25 失败）必须失效缓存，避免 /chat 读到坏索引；下次入库自愈。
        _invalidate_engine_cache(kb_id)
        raise
    save_index_manifest(manifest)
    _invalidate_engine_cache(kb_id)
    # chunk_count 取索引现存总数（含未变文档），document_count 为库内文档总数。
    return IngestResult(kb_id, len(pdf_files), engine.count(), doc_results)


# 构建 build kb index 相关逻辑。
def build_kb_index(kb_id: str, source_dir: str) -> IngestResult:
    # 取 KB 写锁串行化整个入库：扫描→hash→解析→写索引期间，删库与文件增删都被挡住， 杜绝「manifest hash 属旧文件、chunk 来自新文件」与并发任务交错。
    with kb_write_lock(kb_id):
        return _build_kb_index_locked(kb_id, source_dir)


# 构建 build kb index locked 相关逻辑。
def _build_kb_index_locked(kb_id: str, source_dir: str) -> IngestResult:
    engine = RetrieverFactory.get_engine(kb_id)
    pdf_files = list_pdf_files(source_dir)
    if not pdf_files:
        # 空库：清索引并删 manifest，否则重新加回相同文件会被 diff 误判为「未变」而不重建。
        try:
            engine.clear()
        except Exception:
            # clear 失败（残留旧块）不能删 manifest 报成功，否则旧文档仍可被检索。
            _invalidate_engine_cache(kb_id)
            raise
        _remove_manifest(kb_id)
        _invalidate_engine_cache(kb_id)
        return IngestResult(kb_id, 0, 0, [])

    rust_core = ensure_rust_core("scan_pdf_manifest_native")
    abs_dir = os.path.abspath(source_dir)
    manifest = stamp_index_build_version(
        stamp_chunk_identity_contract(
            rust_core.scan_pdf_manifest_native(kb_id, abs_dir)
        )
    )
    source_hash_by_name = _documents_by_name(manifest)

    # 有可比对的上一版且分块契约未变则增量；两路索引缺失/不一致（向量或 BM25 丢失）则强制全量自愈。
    plan = plan_incremental(load_index_manifest(kb_id), manifest)
    if plan is None or not engine.is_consistent():
        return _full_rebuild(
            engine, kb_id, source_dir, pdf_files, manifest, source_hash_by_name
        )
    return _incremental_apply(
        engine, kb_id, source_dir, pdf_files, manifest, plan, source_hash_by_name
    )


# ────────────────────────────────────────────── Phase 3：事务化构建 snapshot → staging generation → validate → switch_active → invalidate → async clean ──────────────────────────────────────────────

# 旧代 grace period：切代后延迟回收，给在途请求留出持有旧引擎的时间窗口。
GENERATION_CLEANUP_DELAY_SECONDS = 60.0


# 处理 hardlink snapshot 相关逻辑。
def _hardlink_snapshot(source_dir: str, gen_dir: str, filenames: list[str]) -> None:
    # 源文件不可变快照：硬链接到 generation 工作区，避免构建期间源文件被改写。 跨文件系统时退化为 copy（语义不变，只是多占磁盘）。
    os.makedirs(gen_dir, exist_ok=True)
    for name in filenames:
        src = os.path.join(source_dir, name)
        dst = os.path.join(gen_dir, name)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


# 校验 verify staging 相关逻辑。
def _verify_staging(staging: HybridRetriever, all_chunks: list) -> None:
    # staging 入库后精确校验：count 与 chunk_id 集合都要与 all_chunks 完全吻合。 仅靠两路集合相等（_verify_consistent）无法发现「两路以相同方式少写」的情况。
    expected_count = len(all_chunks)
    actual_count = staging.count()
    if actual_count != expected_count:
        raise IndexInconsistencyError(
            f"staging count mismatch: expected {expected_count}, got {actual_count}"
        )
    expected_ids = {str(c["meta"]["chunk_id"]) for c in all_chunks}
    actual_ids = staging.chunk_ids()
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise IndexInconsistencyError(
            f"staging chunk_id mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    # 增量复用时向量与 BM25 经不同路径填充，须交叉核对两路 chunk_id 一致，识破向量漏写。
    if not staging.is_consistent():
        raise IndexInconsistencyError("staging vector/bm25 chunk_id sets diverge")


# 处理 cleanup generation storage 相关逻辑。
def _cleanup_generation_storage(kb_id: str, gen_id: str) -> None:
    # 回收单个非 active generation 的全部磁盘资源：Chroma 集合、BM25 pkl、gen 快照目录。 持 kb_write_lock 与构建/删库串行，避免 GC 与 state.json 写入并发丢更新或 .tmp 互相覆盖。
    with kb_write_lock(kb_id):
        if has_readers(kb_id):
            raise KBCleanupError(
                f"KB {kb_id} 仍有在途读者，延后清理 generation {gen_id}"
            )
        settings = get_settings()
        collection_id = settings.kb_collection_id(kb_id, gen_id)
        all_ok = True

        try:
            import chromadb

            chromadb.PersistentClient(
                path=settings.chroma_persist_dir
            ).delete_collection(f"col-{collection_id}")
        except ValueError:
            pass  # Chroma 对 not-found 抛 ValueError：集合已清，视为成功
        except Exception:
            all_ok = False

        bm25_path = os.path.join(settings.bm25_persist_dir, f"bm25_{collection_id}.pkl")
        try:
            os.remove(bm25_path)
        except FileNotFoundError:
            pass
        except Exception:
            all_ok = False

        gen_dir = settings.kb_generation_dir(kb_id, gen_id)
        if os.path.exists(gen_dir):
            try:
                shutil.rmtree(gen_dir)
            except Exception:
                all_ok = False

        if all_ok:
            # 全部资源清理成功后才移除 state 记录；失败时 stale GC 下次扫描可重试。
            try:
                KBState(kb_id).remove_generation(gen_id)
            except Exception:
                all_ok = False

        # 任一资源未清理则向上抛出：调用方据此保留记录并报告可重试失败。
        if not all_ok:
            raise KBCleanupError(f"generation {gen_id} 资源未完全清理")


# 处理 cleanup generation storage quiet 相关逻辑。
def _cleanup_generation_storage_quiet(kb_id: str, gen_id: str) -> None:
    # 异步 GC 包装：清理失败由下次 stale GC 扫描重试，daemon 线程不向上抛噪声。
    try:
        _cleanup_generation_storage(kb_id, gen_id)
    except Exception:
        pass


# 处理 schedule generation cleanup 相关逻辑。
def _schedule_generation_cleanup(kb_id: str, gen_id: str) -> None:
    # 延迟 GENERATION_CLEANUP_DELAY_SECONDS 秒后在 daemon 线程异步清理旧代（受统一 Timer 注册表管理）。 grace period 保证切代前已获取旧引擎的在途请求能完成，再物理删除 Chroma 集合。
    _start_tracked_timer(
        GENERATION_CLEANUP_DELAY_SECONDS,
        _cleanup_generation_storage_quiet,
        args=(kb_id, gen_id),
    )


# 构建 build staging engine 相关逻辑。
def _build_staging_engine(kb_id: str, gen_id: str) -> HybridRetriever:
    collection_id = get_settings().kb_collection_id(kb_id, gen_id)
    return HybridRetriever(
        vector_retriever=VectorRetriever(collection_id=collection_id),
        bm25_retriever=BM25Retriever(collection_id=collection_id),
    )


# 规划 plan transactional incremental 相关逻辑。
def _plan_transactional_incremental(state: KBState, manifest: dict):
    # 以 active generation 的文档清单作 diff 基准：它是已提交集合的权威记录，绝不像 manifest 文件那样滞后。
    prev_active = state.active()
    if not prev_active:
        return None, None
    prev_snapshot = {
        "doc_id": manifest.get("doc_id"),
        "index_build_version": prev_active.get("index_build_version"),
        "documents": prev_active.get("documents", []),
    }
    plan = plan_incremental(prev_snapshot, manifest)
    return (plan, prev_active) if plan is not None else (None, None)


# 处理 fill staging incremental 相关逻辑。
def _fill_staging_incremental(
    kb_id, staging, prev_active, plan, gen_dir, source_hash_by_name
):
    # 复用上一代未变文档的 chunk+向量（不重算 embedding），只解析新增/改动文档并嵌入；BM25 整体重建。
    prev_collection_id = get_settings().kb_collection_id(kb_id, prev_active["id"])
    prev_vector = VectorRetriever(collection_id=prev_collection_id)
    prev_bm25 = BM25Retriever(collection_id=prev_collection_id)

    # 旧代两路 chunk_id 集合必须相等且非空，且数量与提交时记录吻合：仅校验 count 无法识破"同数量但内容损坏"。
    embedding_by_id = prev_vector.embeddings_by_chunk_id()
    bm25_registry = prev_bm25.export_registry()
    bm25_ids = {str(d["meta"]["chunk_id"]) for d in bm25_registry}
    expected_prev = prev_active.get("expected_count")
    if not bm25_ids or set(embedding_by_id) != bm25_ids:
        raise IndexInconsistencyError("previous generation stores diverge")
    if isinstance(expected_prev, int) and len(bm25_ids) != expected_prev:
        raise IndexInconsistencyError("previous generation size mismatch")

    # 文本/metadata 以 BM25 registry 为权威，向量按 chunk_id 关联，杜绝向量侧损坏被洗白。 复用前逐块校验内容自洽：source 属于 active 文档、source_sha256 与之一致、chunk_id 与 metadata 自洽， 识破"同 ID 同数量但 source/hash/metadata 损坏"的旧数据（chunk_id 本身编码了 hash+name+页跨度+局部序号）。
    active_hashes = {
        d.get("name"): d.get("sha256") for d in prev_active.get("documents", [])
    }
    drop = {str(s) for s in plan.removed_sources if s}
    reused_chunks, reused_embeddings = [], []
    for doc in bm25_registry:
        meta = doc["meta"]
        source = str(meta["source"])
        if source in drop:
            continue
        if active_hashes.get(source) != str(meta["source_sha256"]):
            raise IndexInconsistencyError("previous generation source/hash corrupt")
        expected_id = build_chunk_id(
            str(meta["source_sha256"]),
            source,
            int(meta["page_start"]),
            int(meta["page_end"]),
            int(meta["local_chunk_index"]),
        )
        if expected_id != str(meta["chunk_id"]):
            raise IndexInconsistencyError(
                "previous generation chunk_id/metadata mismatch"
            )
        reused_chunks.append(doc)
        reused_embeddings.append(embedding_by_id[str(meta["chunk_id"])])

    # 新块续号：从复用块的最大展示编号之后开始，保证 chunk_index 唯一且不与复用块冲突。
    start_index = (
        max((int(c["meta"]["chunk_index"]) for c in reused_chunks), default=-1) + 1
    )
    new_chunks, doc_results = _parse_and_chunk(
        gen_dir, plan.to_parse, source_hash_by_name, start_index=start_index
    )
    all_chunks = reused_chunks + new_chunks
    if reused_chunks:
        staging.vector_retriever.add_with_embeddings(reused_chunks, reused_embeddings)
    if new_chunks:
        staging.vector_retriever.add_documents(new_chunks)
    staging.bm25_retriever.index(all_chunks)
    return all_chunks, doc_results


# 处理 populate staging 相关逻辑。
def _populate_staging(
    kb_id, state, gen_dir, pdf_files, manifest, source_hash_by_name, staging
):
    # 决定增量复用还是全量填充 staging，返回 (all_chunks, doc_results)。
    plan, prev_active = _plan_transactional_incremental(state, manifest)
    if plan is not None:
        try:
            return _fill_staging_incremental(
                kb_id, staging, prev_active, plan, gen_dir, source_hash_by_name
            )
        except Exception as exc:
            # 复用失败（旧集合缺失/损坏/导出异常）：清空 staging 回退全量重建，保证自愈。 必须记录：否则增量长期失效只表现为性能退化，缺乏可观测性。
            log_event(
                "ingest",
                "incremental_reuse_fallback",
                {},
                level=logging.WARNING,
                kb_id=kb_id,
                error_class=type(exc).__name__,
            )
            staging.clear()
    all_chunks, doc_results = _parse_and_chunk(gen_dir, pdf_files, source_hash_by_name)
    if all_chunks:
        staging.index(all_chunks)
    return all_chunks, doc_results


# 构建 build kb index transactional 相关逻辑。
def build_kb_index_transactional(
    kb_id: str, source_dir: str, on_commit=None
) -> IngestResult:
    # 事务化构建入口：取 KB 写锁串行化同一知识库的所有写操作。 on_commit 在 switch_active 前同步记录待提交 gen_id；回调失败会中止提交。
    with kb_write_lock(kb_id):
        return _build_transactional_locked(kb_id, source_dir, on_commit)


# 构建 build transactional locked 相关逻辑。
def _build_transactional_locked(
    kb_id: str, source_dir: str, on_commit=None
) -> IngestResult:
    state = KBState(kb_id)
    pdf_files = list_pdf_files(source_dir)

    if not pdf_files:
        return _transactional_empty(kb_id, state, on_commit)

    rust_core = ensure_rust_core("scan_pdf_manifest_native")
    manifest = stamp_index_build_version(
        stamp_chunk_identity_contract(
            rust_core.scan_pdf_manifest_native(kb_id, os.path.abspath(source_dir))
        )
    )
    source_hash_by_name = _documents_by_name(manifest)

    gen_id = state.begin_generation(Embedder.MODEL_NAME, INDEX_BUILD_VERSION)
    gen_dir = get_settings().kb_generation_dir(kb_id, gen_id)

    try:
        _hardlink_snapshot(source_dir, gen_dir, pdf_files)
        staging = _build_staging_engine(kb_id, gen_id)
        all_chunks, doc_results = _populate_staging(
            kb_id, state, gen_dir, pdf_files, manifest, source_hash_by_name, staging
        )
        if all_chunks:
            _verify_staging(staging, all_chunks)
        state.mark_ready(
            gen_id,
            expected_count=len(all_chunks),
            documents=manifest.get("documents", []),
        )
        # 提交前记录 gen_id 到 journal：写失败则抛出，在 switch_active 前中止，杜绝"已提交但 journal 未记"。
        if on_commit is not None:
            on_commit(gen_id)
        old_gen = state.switch_active(gen_id)  # 提交点：持有 kb_write_lock 保证原子性
    except Exception:
        # 各步独立容错，保证原始异常不被 mark_failed/cleanup 的次级异常覆盖。
        try:
            state.mark_failed(gen_id)
        except Exception:
            pass
        try:
            _cleanup_generation_storage(kb_id, gen_id)
        except Exception:
            pass
        raise

    # post-commit：gen 已 active，以下操作 best-effort，失败不回滚也不向上抛。
    try:
        RetrieverFactory.invalidate(kb_id)
    except Exception:
        pass
    try:
        save_index_manifest(manifest)
    except Exception:
        pass
    if old_gen:
        try:
            _schedule_generation_cleanup(kb_id, old_gen)
        except Exception:
            pass

    return IngestResult(kb_id, len(pdf_files), len(all_chunks), doc_results)


# 执行事务化 transactional empty 相关逻辑。
def _transactional_empty(kb_id: str, state: KBState, on_commit=None) -> IngestResult:
    gen_id = state.begin_generation(Embedder.MODEL_NAME, INDEX_BUILD_VERSION)
    try:
        state.mark_ready(gen_id, expected_count=0, documents=[])
        if on_commit is not None:
            on_commit(gen_id)  # 提交前记录 gen_id，写失败则中止提交
        old_gen = state.switch_active(gen_id)  # 提交点
    except Exception:
        try:
            state.mark_failed(gen_id)
        except Exception:
            pass
        raise

    # post-commit：best-effort，失败不回滚也不向上抛。
    try:
        RetrieverFactory.invalidate(kb_id)
    except Exception:
        pass
    try:
        _remove_manifest(kb_id)
    except Exception:
        pass
    if old_gen:
        try:
            _schedule_generation_cleanup(kb_id, old_gen)
        except Exception:
            pass
    return IngestResult(kb_id, 0, 0, [])
