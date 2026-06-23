import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    # 测试进程需优先导入仓库源码。
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _reset_retriever_engine_cache():
    # 防止进程级引擎缓存在测试间留脏；仅在用过检索栈时清，不强行拉起重依赖。
    yield
    module = sys.modules.get("graph.subgraphs.qa")
    if module is not None:
        factory = module.RetrieverFactory
        with factory._lock:
            factory._engines.clear()
