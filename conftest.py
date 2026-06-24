import os
import sys
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    # 测试进程需优先导入仓库源码。
    sys.path.insert(0, ROOT)


# 处理 reset retriever engine cache 相关逻辑。
@pytest.fixture(autouse=True)
def _reset_retriever_engine_cache():
    # 防止进程级引擎缓存在测试间留脏；仅在用过检索栈时清，不强行拉起重依赖。
    yield
    module = sys.modules.get("cogdoc.graph.subgraphs.qa")
    if module is not None:
        factory = module.RetrieverFactory
        with factory._lock:
            factory._engines.clear()


# 处理 isolate epoch store 相关逻辑。
@pytest.fixture(autouse=True)
def _isolate_epoch_store(tmp_path, monkeypatch):
    # 全局 epoch / lifecycle / journal / purge 单例隔离到每个测试 tmp，避免污染仓库或跨测试串状态。
    import cogdoc.service.kb_epoch as ke
    import cogdoc.service.kb_lifecycle as kl
    import cogdoc.service.mutation_journal as mj
    import cogdoc.service.purge_queue as pq

    monkeypatch.setattr(
        ke, "_shared", ke.EpochStore(path=str(tmp_path / "epochs.json"))
    )
    monkeypatch.setattr(
        kl, "_shared", kl.LifecycleStore(path=str(tmp_path / "lifecycle.json"))
    )
    monkeypatch.setattr(
        mj, "_shared", mj.MutationJournal(journal_dir=str(tmp_path / "journal"))
    )
    monkeypatch.setattr(
        pq, "_shared", pq.PurgeQueue(path=str(tmp_path / "purge_queue.json"))
    )
    # 测试在同一进程内反复拉起 lifespan，关掉严格单实例避免进程锁争用误杀。
    monkeypatch.setenv("COGDOC_ALLOW_MULTI", "1")
    yield
    # 取消测试中残留的后台 Timer，避免 daemon 线程跨测试触发真实清理。
    import cogdoc.service.ingest_service as isvc

    isvc.cancel_all_timers()
