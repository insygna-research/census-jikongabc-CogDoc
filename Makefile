# CogDoc 统一开发命令，覆盖原生扩展构建、健康检查、测试与启动。

PYTHON  ?= python
MATURIN ?= maturin

.PHONY: help native check test run eval eval-quality serve frontend

help:
	@echo "make native  - 构建 rust_core 原生扩展 (cd rust_core && maturin develop --release)"
	@echo "make check   - 校验原生扩展是否就绪 (scripts/check_native.py)"
	@echo "make test    - 运行 pytest 全量测试"
	@echo "make eval    - 离线检索评测 recall@k/MRR (scripts/eval_retrieval.py)"
	@echo "make eval-quality - 离线质量评测 router/citation/faithfulness (scripts/eval_quality.py)"
	@echo "make run     - 启动 RAG 问答控制台 (run.py)"
	@echo "make serve   - 启动 FastAPI 服务 (uvicorn api.app:app)"
	@echo "make frontend - 启动 Streamlit 前端 (frontend/app.py)"

# 编辑过 rust_core/src 下任何 .rs 后都必须重跑，否则加载的是旧 .so。
native:
	cd rust_core && $(MATURIN) develop --release

check:
	$(PYTHON) scripts/check_native.py

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) scripts/eval_retrieval.py

eval-quality:
	$(PYTHON) scripts/eval_quality.py

run:
	$(PYTHON) run.py

serve:
	$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000

frontend:
	$(PYTHON) -m streamlit run frontend/app.py
