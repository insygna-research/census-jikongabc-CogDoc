# CogDoc 统一开发命令，覆盖原生扩展构建、健康检查、测试与启动。

PYTHON  ?= python
MATURIN ?= maturin
SHELL   := /bin/bash

# src-layout：包源码在 src/，入口经 PYTHONPATH 注入，无需先安装即可 run/serve/test。
export PYTHONPATH := src

.PHONY: help install native check test smoke-api run debug backup eval eval-coverage eval-quality eval-quality-coverage eval-suite eval-suite-run-retrieval eval-suite-report eval-suite-baseline eval-suite-update-baseline serve frontend docs docs-install

help:
	@echo "make install - 可编辑安装含开发依赖 (pip install -e '.[dev]')"
	@echo "make native  - 构建 rust_core 原生扩展 (cd rust_core && maturin develop --release)"
	@echo "make check   - 校验原生扩展是否就绪 (scripts/check_native.py)"
	@echo "make test    - 运行 pytest 全量测试"
	@echo "make smoke-api - 运行不依赖真实模型/索引的 API E2E smoke"
	@echo "make backup  - 备份 data/ 与 logs/traces/ 到 backups/"
	@echo "make eval    - 离线检索评测 recall@k/MRR (scripts/eval_retrieval.py)"
	@echo "make eval-coverage - 只检查检索评测集覆盖面，不执行真实检索"
	@echo "make eval-quality - 离线质量评测 router/citation/faithfulness (scripts/eval_quality.py)"
	@echo "make eval-quality-coverage - 检查质量评测集覆盖面"
	@echo "make eval-suite - 运行组合评测门禁（覆盖审计 + 质量评测）"
	@echo "make eval-suite-run-retrieval - 运行组合评测并执行真实检索"
	@echo "make eval-suite-report - 写入 eval/eval_suite_report.json"
	@echo "make eval-suite-baseline - 对比 eval/eval_suite_baseline.json"
	@echo "make eval-suite-update-baseline - 更新 eval/eval_suite_baseline.json"
	@echo "make run     - 启动多库多对话控制台 (python -m cogdoc.cli)"
	@echo "make debug   - 启动独立 Debug 控制台 (python -m cogdoc.debug)"
	@echo "make serve   - 启动 FastAPI 服务 (uvicorn cogdoc.api.app:app)"
	@echo "make frontend - 启动 Streamlit 前端 (src/cogdoc/frontend/app.py)"
	@echo "make docs     - 本地渲染 Mermaid 图 (docs/images/*.mmd → .png)"
	@echo "make docs-install - 安装 mermaid-cli 依赖 (首次使用 make docs 前运行)"

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

backup:
	$(PYTHON) scripts/backup_state.py

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

eval-suite-run-retrieval:
	$(PYTHON) scripts/eval_suite.py --run-retrieval

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

docs-install:
	PUPPETEER_SKIP_DOWNLOAD=true npm install --prefix docs/images

docs:
	@test -x docs/images/node_modules/.bin/mmdc || (echo ">> 先运行 make docs-install"; exit 1)
	@CHROME=$$(which google-chrome-stable 2>/dev/null || which chromium-browser 2>/dev/null || which chromium 2>/dev/null); \
	if [ -z "$$CHROME" ]; then echo ">> 未找到 Chrome/Chromium，请先安装"; exit 1; fi; \
	set -euo pipefail; \
	MMDC=docs/images/node_modules/.bin/mmdc; \
	for mmd in docs/images/*.mmd; do \
		echo ">> 渲染 $$mmd"; \
		svg="$${mmd%.mmd}.svg"; \
		rm -f "$$svg"; \
		if [[ "$$(basename "$$mmd")" == architecture*.mmd ]]; then \
			PUPPETEER_EXECUTABLE_PATH=$$CHROME "$$MMDC" -i "$$mmd" -o "$$svg" \
				-b white --width 1600 --height 1400 \
				-c docs/images/mermaid-config.json -p .github/puppeteer-config.json; \
		else \
			PUPPETEER_EXECUTABLE_PATH=$$CHROME "$$MMDC" -i "$$mmd" -o "$$svg" \
				-b white \
				-c docs/images/mermaid-config.json -p .github/puppeteer-config.json; \
		fi; \
	done
