# CogDoc 统一开发命令，覆盖原生扩展构建、健康检查、测试与启动。

PYTHON  ?= python
MATURIN ?= maturin

# src-layout：包源码在 src/，入口经 PYTHONPATH 注入，无需先安装即可 run/serve/test。
export PYTHONPATH := src

.PHONY: help install native check test smoke-api run debug eval eval-coverage eval-quality eval-quality-coverage eval-suite eval-suite-report eval-suite-baseline eval-suite-update-baseline serve frontend

help:
	@echo "make install - 可编辑安装含开发依赖 (pip install -e '.[dev]')"
	@echo "make native  - 构建 rust_core 原生扩展 (cd rust_core && maturin develop --release)"
	@echo "make check   - 校验原生扩展是否就绪 (scripts/check_native.py)"
	@echo "make test    - 运行 pytest 全量测试"
	@echo "make smoke-api - 运行不依赖真实模型/索引的 API E2E smoke"
	@echo "make eval    - 离线检索评测 recall@k/MRR (scripts/eval_retrieval.py)"
	@echo "make eval-coverage - 只检查检索评测集覆盖面，不执行真实检索"
	@echo "make eval-quality - 离线质量评测 router/citation/faithfulness (scripts/eval_quality.py)"
	@echo "make eval-quality-coverage - 检查质量评测集覆盖面"
	@echo "make eval-suite - 运行组合评测门禁（覆盖审计 + 质量评测）"
	@echo "make eval-suite-report - 写入 eval/eval_suite_report.json"
	@echo "make eval-suite-baseline - 对比 eval/eval_suite_baseline.json"
	@echo "make eval-suite-update-baseline - 更新 eval/eval_suite_baseline.json"
	@echo "make run     - 启动多库多对话控制台 (python -m cogdoc.cli)"
	@echo "make debug   - 启动独立 Debug 控制台 (python -m cogdoc.debug)"
	@echo "make serve   - 启动 FastAPI 服务 (uvicorn cogdoc.api.app:app)"
	@echo "make frontend - 启动 Streamlit 前端 (src/cogdoc/frontend/app.py)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

# 编辑过 rust_core/src 下任何 .rs 后都必须重跑，否则加载的是旧 .so。
native:
	cd rust_core && $(MATURIN) develop --release

check:
	$(PYTHON) scripts/check_native.py

test:
	$(PYTHON) -m pytest

smoke-api:
	$(PYTHON) scripts/smoke_api.py

eval:
	$(PYTHON) scripts/eval_retrieval.py

eval-coverage:
	$(PYTHON) scripts/eval_retrieval.py --coverage-only

eval-quality:
	$(PYTHON) scripts/eval_quality.py

eval-quality-coverage:
	$(PYTHON) scripts/eval_quality.py --check-coverage

eval-suite:
	$(PYTHON) scripts/eval_suite.py

eval-suite-report:
	$(PYTHON) scripts/eval_suite.py --json eval/eval_suite_report.json

eval-suite-baseline:
	$(PYTHON) scripts/eval_suite.py --baseline eval/eval_suite_baseline.json

eval-suite-update-baseline:
	$(PYTHON) scripts/eval_suite.py --update-baseline eval/eval_suite_baseline.json

run:
	$(PYTHON) -m cogdoc.cli

debug:
	$(PYTHON) -m cogdoc.debug

serve:
	$(PYTHON) -m uvicorn cogdoc.api.app:app --host 0.0.0.0 --port 8000

frontend:
	$(PYTHON) -m streamlit run src/cogdoc/frontend/app.py
