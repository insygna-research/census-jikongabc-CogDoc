# CogDoc

**A RAG console for papers and technical docs — LangGraph agents on top, deterministic Rust kernels underneath.**

[English](README.md) · [简体中文](README_zh-CN.md)

---

## What Is This

CogDoc answers questions over your own PDF library. A [LangGraph](https://langchain-ai.github.io/langgraph/) workflow routes intent, rewrites the query, retrieves and reranks evidence, then generates an answer where **every fact carries a `[source:Pn]` citation**. Retrieval fusion, citation rules, and manifest hashing live in a small [Rust](https://www.rust-lang.org/) core exposed through [PyO3](https://pyo3.rs/); chunk identity is a versioned Python-side contract shared across the whole pipeline.

## Why It Exists

RAG over technical documents fails in quiet, specific ways: query rewrites hallucinate new entities, models fabricate file names and page numbers, and indexes silently drift when a PDF changes. CogDoc treats those as contract problems, not prompt problems:

- **Rewrites are guarded** — semantically drifting rewrites are filtered by cosine before they ever touch retrieval.
- **Citations are checked, not trusted** — a Rust validator matches every tag against the retrieved `source`/`page`; fabricated ones bounce the answer back into a self-heal loop.
- **Indexing is content-addressed** — a per-file SHA-256 manifest plus a versioned chunk-identity contract decides exactly when to reuse vs. rebuild.

The result: the deterministic, verifiable behavior stays in tested kernels and survives agent/prompt churn.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make native     # build the Rust extension: cd rust_core && maturin develop --release
make check      # verify the extension and its native symbols
make run        # build/reuse the index, warm up models, start the console
```

Python dependencies are split into [requirements.txt](requirements.txt) for runtime packages and [requirements-dev.txt](requirements-dev.txt) for build/test tooling.

Put PDFs in `测试论文/` (or set `COGDOC_DOC_DIR`). In the console, `/local` uses Ollama, `/cloud` uses the cloud backend, `exit` quits. `make native` must be re-run after any change under `rust_core/src/` — the `.so` is not auto-rebuilt and not committed.

## What You Get

- **Grounded answers with verified citations.** Generation is constrained to retrieved `<Document>` blocks; fabricated file/page tags are caught by the Rust checker and re-generated up to `max_iteration_count`.
- **Hybrid retrieval with deterministic fusion.** Vector (Chroma + BGE) and BM25 (jieba) recall are merged by a Rust RRF kernel with stable, reproducible tie-breaks.
- **Indexing that rebuilds only when it must.** Unchanged corpora skip rebuilds; changed PDFs or a bumped chunking contract force a clean reindex.

## Status

| Area | Status |
| --- | --- |
| Indexing pipeline (parse → chunk → index → manifest) | Implemented |
| Intent router | Implemented |
| Query rewrite + semantic drift guard | Implemented |
| Hybrid retrieval (Vector + BM25 + Rust RRF) | Implemented |
| Cross-encoder rerank | Implemented |
| Citation validation + self-heal loop | Implemented |
| Rust native core | Implemented |
| Summary / Compare subgraphs | Placeholder nodes |
| `api/`, `frontend/` | Empty placeholder packages |

## Architecture

```text
user question
  -> intent_router (qa | summary | compare | unknown)
  -> QA subgraph
       rewrite_node          QueryRewriteAgent: 1–3 keyword queries
    -> verify_rewrite_node   RewriteVerifyAgent: cosine drift filter
    -> retrieve_node         HybridRetriever: Vector + BM25 per query, Rust RRF fuse
    -> rerank_node           BGEReranker: cross-encoder top_n
    -> generate_node         Generator: grounded answer with [source:Pn] tags
    -> citation_node         CitationValidatorAgent: Rust checker + Python critique
       (loop generate <-> citation up to max_iteration_count, else END)
```

The Python layer owns orchestration, prompts, model clients, indexing, and the console. The Rust layer (`rust_core`) owns deterministic kernels that stay stable across agent changes and are unit-tested independently.

## Indexing Pipeline

Driven by `run.py` (`build_index`) before the console starts:

1. **Scan** — `scan_pdf_manifest_native` (Rust) hashes every PDF with rayon-parallel, 1 MiB-buffered SHA-256 and returns `{doc_id, documents: [{name, size, sha256}]}`, sorted by filename.
2. **Compare** — `manifests_match` reuses the index only if `doc_id`, `chunk_identity_version`, and every `{name, sha256}` match the saved manifest; any mismatch forces a rebuild.
3. **Parse** — `smart_parse` (PyMuPDF) extracts page text, reflows two-column layouts by block center-x, and flags likely scanned pages (`is_ocr_fallback`).
4. **Chunk** — `chunk_paper` slides a 600-char / 60-overlap window (30-char min) over the page stream, maps each chunk to its page span via `bisect`, and assigns a stable `chunk_id`.
5. **Index** — chunks land in Chroma (vector) and a pickled BM25Okapi store; `save_index_manifest` persists the manifest.

**Chunk identity contract:**

```
chunk_id = sha256:{source_sha256}:p{page_start}-p{page_end}:c{local_chunk_index}
```

`chunk_id` is the single stable identity key across chunker, index, retriever, RRF, and evidence — dedup and fusion never rely on array position. It is versioned (`chunk_identity_version = source_sha256_page_span_local_v1_cs600_ov60_min30`); changing chunk boundaries must bump `CHUNK_IDENTITY_BASE_VERSION` so stale indexes rebuild instead of mixing schemes.

## Query Pipeline

- **Intent routing** — `RouterAgent` asks the LLM for structured `task_type ∈ {qa, summary, compare, unknown}` and falls back to `qa` on any parse error. Only `qa` is wired to a real subgraph today.
- **Rewrite + drift guard** — `QueryRewriteAgent` emits 1–3 keyword queries (pydantic structured output). `RewriteVerifyAgent` embeds `[original] + rewrites` in one batch, keeps those with `cosine >= rewrite_similarity_threshold` (default `0.5`), logs kept/dropped to `steps_trace`, and falls back to the original query alone if all are dropped.
- **Hybrid retrieval + RRF** — per query, both channels over-recall `top_k * 3` (QA uses `top_k = 9` → 27/channel); `rrf_fusion_native` (Rust, `k = 60`) computes `score(d) = Σ_c 1 / (k + rank_c(d))`, merges hits sharing a `chunk_id`, and sorts by score desc then identity key asc for determinism.
- **Rerank** — `BGEReranker` (`bge-reranker-v2-m3`) scores `(original_query, doc)` and keeps `top_n = 3`; rewrites never bias the final ranking.
- **Generation + citation self-heal** — `Generator` (OpenAI-compatible; cloud `deepseek-chat` or local `qwen2.5:7b`, `temperature = 0.2`) wraps docs as `<Document source=… page=… chunk_id=…>` and forces `[source:Pn]` tags. `validate_citations_native` (Rust) returns structured `missing_citations` / `invalid_sources` / `invalid_pages`; `citation_node` turns failures into a critique and loops `generate → citation` up to `max_iteration_count` (default `2`). Only validated answers are printed.

## Usage Example

```text
[本地Ollama] 请输入您的问题 >>> 参加 AI 智能体开发应用赛时，团队需要重点关注哪些提交要求？

参赛团队应先确认赛事规程中的参赛对象、报名方式和作品提交节点，避免因流程性要求遗漏影响评审资格。[AI智能体开发应用赛赛事规程260428.pdf:P2]

作品材料需要围绕智能体应用场景、核心功能、技术方案和运行效果展开说明，提交内容应能支撑评委复现或理解系统能力。[AI智能体开发应用赛赛事规程260428.pdf:P4]

如果团队选择本地部署或调用外部模型服务，还应在说明中交代运行环境、依赖组件和接口配置，降低评审过程中的复现成本。[大模型开发应用赛.pdf:P6]
```

The exact wording depends on the selected model and the local PDF corpus. The important contract is that every factual claim should end with a citation whose file name and page exist in the retrieved context.

## Native Core

`rust_core` is a PyO3/maturin extension, loaded via `tools.rust_core_loader.ensure_rust_core`, which fails fast with a `maturin develop` hint if the build is missing or a symbol is stale.

| Function | Module | Purpose |
| --- | --- | --- |
| `scan_pdf_manifest_native` | `scanner.rs` | Rayon-parallel, buffered SHA-256 of every PDF; size + hash manifest, stably sorted |
| `rrf_fusion_native` | `rrf.rs` | Deterministic RRF (`k=60`) merge of vector + BM25 results, keyed on `chunk_id` |
| `validate_citations_native` | `citation.rs` | Structured citation check → `invalid_sources` / `invalid_pages` / `missing_citations` |

## Project Layout

```text
CogDoc/
├── agents/                  # router, query_rewriter, rewrite_verifier, generator, citation_validator
├── graph/
│   ├── state.py             # GraphState / RetrievedDoc / DocMeta / Evidence + list reducers
│   ├── workflow.py          # top-level graph: intent_router -> qa | summary | compare
│   └── subgraphs/qa.py      # QA subgraph wiring
├── tools/
│   ├── parser.py            # PyMuPDF parsing, two-column reflow + OCR-fallback flag
│   ├── chunker.py           # char-window chunking with page-span mapping
│   ├── chunk_identity.py    # chunk_id format + versioned identity contract
│   ├── manifest.py          # manifest IO + match logic for index reuse
│   ├── embedder.py          # bge-small-zh-v1.5 (normalized)
│   ├── reranker.py          # bge-reranker-v2-m3 cross-encoder
│   ├── rust_core_loader.py  # ensure_rust_core: load + symbol check
│   └── retriever/           # vector (Chroma), bm25 (jieba), hybrid (Rust RRF)
├── rust_core/src/           # lib.rs, scanner.rs, rrf.rs, citation.rs (+ #[cfg(test)] units)
├── scripts/check_native.py  # native extension health check
├── tests/                   # Python regression tests
├── data/                    # generated chroma_db / bm25_db / manifests (runtime state)
├── 测试论文/                 # default PDF corpus dir (override with COGDOC_DOC_DIR)
└── run.py                   # interactive console entry point
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `COGDOC_DOC_DIR` | `测试论文` | PDF corpus directory scanned by `run.py` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local OpenAI-compatible Ollama endpoint |
| `OLLAMA_MODEL_NAME` | `qwen2.5:7b` | Local model name |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | Cloud OpenAI-compatible endpoint |
| `LLM_MODEL_NAME` | `deepseek-chat` | Cloud model name |
| `LLM_API_KEY` | `your-cloud-api-key-here` | Cloud API key |
| `HF_TOKEN` | unset | Optional Hugging Face Hub token |

Requirements: Python 3.13 (the extension targets 3.8+), a Rust toolchain with `cargo` (edition 2024, via [rustup](https://rustup.rs/)), and [maturin](https://www.maturin.rs/). Optional: [Ollama](https://ollama.com/) for local models.

## Development & Testing

| Command | Description |
| --- | --- |
| `make native` | Build / rebuild `rust_core` (required after editing `.rs`) |
| `make check` | Verify the extension is importable and all native symbols exist |
| `make test` | Run the Python test suite |
| `make run` | Start the interactive console |
| `cd rust_core && cargo test` | Run Rust unit tests |
| `cd rust_core && cargo fmt --check` | Check Rust formatting |

Test layering: business logic and the Python↔native API contract are tested in Python (`tests/`); pure-Rust logic uses Rust `#[test]` in `rust_core/src/`. Native-dependent Python tests `importorskip` when `rust_core` is not built, so run `make native` before a full regression.

## Roadmap

- **Phase 1: QA baseline hardening** — keep retrieval, reranking, citation validation, native checks, and tests stable.
- **Phase 2: Summary MVP** — replace the placeholder with a single-document structured summary subgraph.
- **Phase 3: Compare MVP** — replace the placeholder with a multi-document comparison subgraph.
- **Phase 4: API and frontend** — turn the current console workflow into service and UI entry points.
- **Phase 5: Evaluation and native expansion** — add regression datasets, quality metrics, and move more deterministic kernels into `rust_core` only when justified.

## Known Limitations

- The primary interface is still the CLI console; `api/` and `frontend/` are placeholders.
- Summary and Compare routing exist, but their real subgraphs are not implemented yet.
- Citation validation checks physical citation legality (`source` and `page`), not whether the surrounding sentence is semantically perfect.
- The rewrite similarity threshold defaults to `0.5` and should be calibrated on real project data.
- Local model downloads may require network access or a pre-populated Hugging Face cache.

## Troubleshooting

- `Rust 扩展 rust_core 未安装` / `缺少: …` — run `make native`, then `make check`.
- Rust edits don't change behavior — you didn't rebuild; the old `.so` is still loaded. Run `make native`.
- `Model Mismatch!` — the index's embedding model differs from `Embedder.MODEL_NAME`; rebuild the index (clear the `doc_id`'s Chroma collection or change `doc_id`).
- Hugging Face anonymous-rate warning — set `HF_TOKEN` for higher Hub limits; public models usually download without it.

## License

No license has been declared yet. Until a `LICENSE` file is added, default copyright applies and reuse rights are not granted.
