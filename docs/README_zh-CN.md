# CogDoc

**面向论文与技术文档的 RAG 控制台:LangGraph 多 Agent 在上,确定性 Rust kernel 在下。**

[English](../README.md) · [简体中文](README_zh-CN.md)

---

## 这是什么

CogDoc 可以在你自己的 PDF 库上做问答、总结单篇文档、对比多篇文档。一条 [LangGraph](https://langchain-ai.github.io/langgraph/) 工作流先把意图路由到 QA、Summary 或 Compare,再让每条生成结论都绑定回 `[source:Pn]` 引用。分词、BM25 打分、检索融合、引用规则、manifest 哈希放在通过 [PyO3](https://pyo3.rs/) 暴露的小型 [Rust](https://www.rust-lang.org/) 核心里;chunk 身份是 Python 侧带版本的共享契约,贯穿整条链路。

## 为什么需要它

技术文档上的 RAG 会以安静而具体的方式翻车:改写会幻觉出新实体、模型会捏造文件名和页码、PDF 一变索引就悄悄漂移。CogDoc 把这些当作契约问题而非 Prompt 问题来解:

- **改写有守卫** — 语义漂移的改写在进入检索前就被 cosine 过滤掉。
- **引用要校验,不靠信任** — Rust 校验器把每个标签对照召回文档的 `source`/`page`;捏造的引用会把回答打回自愈循环。
- **索引是内容寻址的** — 逐文件 SHA-256 manifest 加上带版本的 chunk 身份契约,精确决定何时复用、何时重建。

结果:确定、可验证的行为留在有测试的 kernel 里,不随 Agent / Prompt 变动而漂移。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # 运行时 + 构建/测试依赖(可编辑安装,src-layout)
make native     # 构建 Rust 扩展: cd rust_core && maturin develop --release
make check      # 校验扩展及其 native 符号
make run        # 构建/复用索引、预热模型、启动控制台
```

依赖统一在 [pyproject.toml](../pyproject.toml):运行时依赖在 `[project.dependencies]`,`dev`(构建/测试)与 `frontend`(Streamlit 客户端)为可选 extras,用 `.[dev]` / `.[frontend]` 安装。包采用 `src/` 布局(`src/cogdoc/`);`make` 目标会把 `src/` 加入 `PYTHONPATH`,因此跑测试无需先安装。

把 PDF 放进收件箱 `your_documents/`(或设置 `COGDOC_DOC_DIR`)。控制台对齐云端 API，采用仿 Claude Code 的斜杠命令:`/kb new <名称>` 建知识库,`/add <文件.pdf>` 把收件箱里的 PDF 加入当前库(同步重建索引),`/new` 开新对话,`/chats` / `/open` 浏览持久化历史,`/rmchat` 与 `/kb rm` 删除(需确认)。`/local` 用 Ollama,`/cloud` 用云端后端,`/help` 列出全部命令,`exit` 退出。`make debug`(`python -m cogdoc.debug`)打开检索可视化控制台,针对单个库打印路由判别、多路改写、召回+精排切块(含 RRF 分)与引证审计。每次修改 `rust_core/src/` 下的代码后都必须重跑 `make native`——`.so` 不会自动重建,也不纳入版本控制。

## 你会得到什么

- **带可验证引用的有据回答。** 生成被约束在召回的 `<Document>` 块内;捏造的文件/页码标签会被 Rust 校验器抓出并重新生成,直到 `max_iteration_count`。Summary 与 Compare 复用同一套校验器,不开豁免。
- **结构化摘要与多文档对比。** Summary 加载一篇点名文档,按固定章节生成带确定性引用的摘要;Compare 加载两篇或更多点名文档,先建逐文档 profile,再按维度输出带引用的对比块。
- **打分路径全 native 的混合检索。** 向量(Chroma + BGE)与 BM25 两路召回由 Rust RRF kernel 以稳定 tie-break 合并;分词(`jieba-rs`)与 BM25 打分(`Bm25Index`)均为 native,且与各自的旧 Python 实现逐位对齐。
- **只在必要时重建的索引。** 未变化的语料跳过重建;PDF 变化或 chunk 契约 bump 才触发干净重索引。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| 索引链路(parse → chunk → index → manifest) | 已实现 |
| 意图路由 | 已实现 |
| 问题改写 + 语义漂移守卫 | 已实现 |
| 混合检索(Vector + native BM25 + Rust RRF) | 已实现 |
| 交叉编码器重排 | 已实现 |
| 引用校验 + 自愈循环 | 已实现 |
| Rust 原生核心(manifest、RRF、citation、分词、BM25) | 已实现 |
| Summary 子图(单文档结构化摘要) | 已实现 |
| Compare 子图(多文档对比) | 已实现 |
| `api/`、`frontend/` | 空占位包 |

## 架构

```text
user question
  -> intent_router (qa | summary | compare | unknown)
  -> QA subgraph
       rewrite_node          QueryRewriteAgent: 1–3 条关键词查询
    -> verify_rewrite_node   RewriteVerifyAgent: cosine 漂移过滤
    -> retrieve_node         HybridRetriever: 每条 query 走 Vector + native BM25,Rust RRF 融合
    -> rerank_node           BGEReranker: cross-encoder 取 top_n
    -> generate_node         Generator: 带 [source:Pn] 标签的受约束回答
    -> citation_node         CitationValidatorAgent: Rust 校验 + Python critique
       (generate <-> citation 循环至 max_iteration_count,否则 END)
  -> Summary 子图            document_loader -> section_planner -> section_summary -> global_summary
  -> Compare 子图            document_loader -> document_profile -> compare_table -> citation
```

Summary 为单个点名文档生成固定章节结构化摘要;Compare 为每篇文档在固定维度上建 profile,再按维度渲染带引用的 Markdown 对比块。两者都从 chunk 元数据确定性地绑定 `[source:Pn]` 引用,并跑与 QA 同一套 `validate_citations_native` 校验——任何子图都不豁免。

Python 层负责图编排、Prompt、模型客户端、索引和控制台。Rust 层(`rust_core`)负责确定性 kernel,不随 Agent 逻辑漂移,并独立做单元测试。

## 索引链路

由 `build_kb_index_transactional` 在某个库的文件变更时驱动(`/add`、`/rm` 或云端上传/删除接口):

1. **扫描** — `scan_pdf_manifest_native`(Rust)用 rayon 并行、1 MiB 缓冲的 SHA-256 计算每个 PDF,返回 `{doc_id, documents: [{name, size, sha256}]}`,按文件名排序。
2. **比对** — `manifests_match` 仅当 `doc_id`、`chunk_identity_version` 及每个 `{name, sha256}` 都与已存 manifest 一致时才复用索引;任一不匹配都强制重建。
3. **解析** — `smart_parse`(PyMuPDF)抽取页文本,按文本块中心 x 坐标重排双栏布局,对疑似扫描页打 `is_ocr_fallback` 标记。
4. **切块** — `chunk_paper` 以 600 字符 / 60 重叠(最小 30)滑窗切过页文本流,通过 `bisect` 把每个 chunk 映射回页跨度,并赋予稳定的 `chunk_id`。
5. **建索引** — chunk 写入 Chroma(向量)和 pickle 持久化的分词语料;加载时由 native `Bm25Index` 从该语料重建。`save_index_manifest` 落盘 manifest。分词走 `tokenize_mixed_text_native`(`jieba-rs`)。

**Chunk 身份契约:**

```
chunk_id = sha256:{source_sha256}:p{page_start}-p{page_end}:c{local_chunk_index}
```

`chunk_id` 是贯穿 chunker、index、retriever、RRF、evidence 的唯一稳定身份键——去重和融合从不依赖数组下标。它带版本(`chunk_identity_version = source_sha256_page_span_local_v1_cs600_ov60_min30`);改动切块边界必须 bump `CHUNK_IDENTITY_BASE_VERSION`,让旧索引重建而非混用两套方案。

## 查询链路

- **意图路由** — `RouterAgent` 要求 LLM 返回结构化 `task_type ∈ {qa, summary, compare, unknown}`,任何解析异常都按关键词规则回退。`qa`、`summary`、`compare` 都已接到真实子图。
- **改写 + 漂移守卫** — `QueryRewriteAgent` 生成 1–3 条关键词查询(pydantic 结构化输出)。`RewriteVerifyAgent` 一次批量 embed `[原问题] + 改写`,保留 `cosine >= rewrite_similarity_threshold`(默认 `0.5`)的改写,把保留/丢弃写入 `steps_trace`;若全被丢弃则只用原问题。
- **混合检索 + RRF** — 每条 query 下两路各超召 `top_k * 3`(QA 用 `top_k = 9` → 每路 27);`rrf_fusion_native`(Rust,`k = 60`)计算 `score(d) = Σ_c 1 / (k + rank_c(d))`,合并共享同一 `chunk_id` 的命中,并按分数降序、身份键升序排序保证确定性。
- **重排** — `BGEReranker`(`bge-reranker-v2-m3`)对 `(原问题, doc)` 打分并取 `top_n = 3`;改写不会影响最终排序。
- **生成 + 引用自愈** — `Generator`(OpenAI 兼容;云端 `deepseek-chat` 或本地 `qwen2.5:7b`,`temperature = 0.2`)把文档包装为 `<Document source=… page=… chunk_id=…>` 并强制 `[source:Pn]` 标签。`validate_citations_native`(Rust)返回结构化的 `missing_citations` / `invalid_sources` / `invalid_pages`;`citation_node` 把失败转成 critique,循环 `generate → citation` 至 `max_iteration_count`(默认 `2`)。只有通过校验的回答才会打印。

**Summary 子图** — `document_loader` 选定一个点名文档(若语料库只有一篇则可自动选中;多文档歧义 query 返回可操作提示),`section_planner` 默认固定为背景与目标、方案与流程、规则与要求、价值与产出、限制与注意事项五个章节(也可由 state 传入自定义标题),`section_summary` 逐章节生成一段短摘要(模型只写正文,`[source:Pn]` 由程序按所用 chunk 确定性绑定),`global_summary` 整合答案并复跑引用校验。无依据章节不带引用、不带 evidence。

**Compare 子图** — `document_loader` 要求显式点名至少 2 篇文档;本地 Ollama 模式最多同时对比 2 篇。`document_profile` 在固定维度上逐文档建 profile(云端:方法/数据/指标/优点/限制/适用场景;本地:方法/数据/指标/限制),并复用 Summary 的 cell 原语。`compare_table` 渲染 Markdown 对比块;云端模式会额外生成一段受控短结论,本地模式跳过这次额外调用以降低内存压力。`compare_citation_node` 先单独校验结论,再校验对比块;任一失败都降级为纯对比块并附警告。全无依据的对比不会被误判为缺引用。

## 使用示例

问答:

```text
[本地Ollama] 请输入您的问题 >>> 参加 AI 智能体开发应用赛时，团队需要重点关注哪些提交要求？

参赛团队应先确认赛事规程中的参赛对象、报名方式和作品提交节点，避免因流程性要求遗漏影响评审资格。[AI智能体开发应用赛赛事规程260428.pdf:P2]

作品材料需要围绕智能体应用场景、核心功能、技术方案和运行效果展开说明，提交内容应能支撑评委复现或理解系统能力。[AI智能体开发应用赛赛事规程260428.pdf:P4]

如果团队选择本地部署或调用外部模型服务，还应在说明中交代运行环境、依赖组件和接口配置，降低评审过程中的复现成本。[大模型开发应用赛.pdf:P6]
```

具体措辞取决于所选模型和本地 PDF 语料。关键契约是:每条事实性陈述都应带有引用,且引用中的文件名和页码必须存在于本轮检索上下文中。

摘要:

```text
[本地Ollama] 请输入您的问题 >>> 请总结 AI智能体开发应用赛赛事规程260428.pdf

# AI智能体开发应用赛赛事规程260428.pdf 结构化摘要

## 背景与目标
...
```

对比:

```text
[云端API] 请输入您的问题 >>> 请对比 paper-a.pdf 和 paper-b.pdf 的方法和指标

# 多文档对比

## 方法
- **paper-a.pdf**：...
- **paper-b.pdf**：...
```

Summary 在多 PDF 语料库中需要点名目标文件。Compare 需要点名至少两篇文件,可使用完整文件名或明确的 stem;本地 Ollama 模式一次只支持两篇。

## Rust 原生核心

`rust_core` 是 PyO3/maturin 扩展,通过 `tools.rust_core_loader.ensure_rust_core` 加载;若构建缺失或符号过期,会尽早失败并给出 `maturin develop` 提示。共暴露五个 native 符号,全部登记在 `scripts/check_native.py`,使 `make check` 能对旧构建报错。

| 符号 | 模块 | 用途 |
| --- | --- | --- |
| `scan_pdf_manifest_native` | `scanner.rs` | rayon 并行、缓冲式 SHA-256 计算所有 PDF;size + 哈希 manifest,稳定排序 |
| `rrf_fusion_native` | `rrf.rs` | 对 vector + BM25 结果做确定性 RRF(`k=60`)融合,以 `chunk_id` 为键 |
| `validate_citations_native` | `citation.rs` | 结构化引用校验 → `invalid_sources` / `invalid_pages` / `missing_citations` |
| `tokenize_mixed_text_native` | `tokenizer.rs` | `jieba-rs` 中英混合分词,与旧 Python `jieba` 路径逐 token 对齐 |
| `Bm25Index`(类) | `bm25.rs` | BM25 索引 + `score_topk`,与 `rank_bm25.BM25Okapi` 逐位对齐,top-k 在 native 端选出 |

## 项目结构

```text
CogDoc/
├── src/cogdoc/              # 可导入的发行包(src-layout)
│   ├── cli.py               # 多库/多对话控制台(python -m cogdoc.cli / `cogdoc`)
│   ├── debug.py             # 检索可视化控制台(python -m cogdoc.debug / `cogdoc-debug`)
│   ├── agents/              # router、query_rewriter、rewrite_verifier、qa_generator、
│   │                        # citation_validator、structured_output、summary_*、compare_*
│   ├── api/                 # FastAPI app、routes、持久化、访问控制、metrics
│   ├── config/              # pydantic-settings 配置
│   ├── frontend/            # Streamlit 瘦客户端
│   ├── graph/               # state.py、workflow.py、subgraphs/(qa、summary、compare)
│   ├── observability/       # 结构化日志 + trace 导出
│   ├── service/             # chat/ingest 服务、KB 生命周期、事务化索引
│   └── tools/               # parser、chunker、manifest、tokenizer、embedder、reranker、
│                            # rust_core_loader、retriever/(vector、native bm25、hybrid)
├── rust_core/src/           # lib.rs、scanner.rs、rrf.rs、citation.rs、tokenizer.rs、bm25.rs
├── scripts/check_native.py  # 原生扩展健康检查(6 个必需符号)
├── tests/                   # Python 回归测试
├── eval/                    # 离线评测示例数据集
├── docs/                    # 中文 README 及其他文档
└── pyproject.toml           # 项目元数据、依赖、构建、pytest 配置
```

## 配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `COGDOC_DOC_DIR` | `your_documents` | 收件箱目录，`/add` 从这里把 PDF 选入知识库 |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | 本地 OpenAI 兼容 Ollama endpoint |
| `OLLAMA_MODEL_NAME` | `qwen2.5:7b` | 本地模型名 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 云端 OpenAI 兼容 endpoint |
| `LLM_MODEL_NAME` | `deepseek-chat` | 云端模型名 |
| `LLM_API_KEY` | `your-cloud-api-key-here` | 云端 API key |
| `HF_TOKEN` | 未设置 | 可选 Hugging Face Hub token |

环境要求:Python 3.13(扩展目标 3.8+)、带 `cargo` 的 Rust 工具链(edition 2024,经 [rustup](https://rustup.rs/))、[maturin](https://www.maturin.rs/)。可选:[Ollama](https://ollama.com/) 用于本地模型。

## 开发与测试

| 命令 | 说明 |
| --- | --- |
| `make native` | 构建 / 重建 `rust_core`(改过 `.rs` 必跑) |
| `make check` | 校验扩展可导入且 native 符号齐全 |
| `make test` | 运行 Python 测试 |
| `make run` | 启动交互式控制台 |
| `cd rust_core && cargo test` | 运行 Rust 单元测试 |
| `cd rust_core && cargo fmt --check` | 检查 Rust 代码格式 |

测试分层:业务逻辑与 Python↔native API 契约用 Python 覆盖(`tests/`);纯 Rust 逻辑用 `rust_core/src/` 里的 Rust `#[test]`。依赖 native 的 Python 测试在未构建时会 `importorskip` 跳过,完整回归前请先 `make native`。

## 已知限制

- 当前主要入口仍是 CLI 控制台;`api/` 和 `frontend/` 还是占位包。
- Summary 与 Compare 是单遍 MVP:逐章节/逐维度的 LLM 调用串行执行(尚未并发),默认章节/维度集合固定,除非通过 graph state 传入自定义配置。
- 本地 Compare 有意限制为 2 篇文档、4 个核心维度,并跳过额外结论生成,以降低 Ollama 内存压力。
- Citation 校验只证明引用的 `source` 和 `page` 物理合法,不证明整句话语义完全正确,也不强制每句都带引用。
- Rewrite 相似度阈值默认 `0.5`,后续应基于真实数据标定。
- 暂无离线评测框架,检索/回答质量改动尚不能量化衡量。
- 本地模型下载依赖网络或已有 Hugging Face 缓存。

## 故障排查

- `Rust 扩展 rust_core 未安装` / `缺少: …` — 运行 `make native`,再 `make check`。
- 改了 Rust 但行为没变 — 没有重新构建,旧 `.so` 仍在被加载。运行 `make native`。
- `Model Mismatch!` — 索引的 embedding 模型与 `Embedder.MODEL_NAME` 不一致;重建索引(清空该 `doc_id` 的 Chroma collection 或更换 `doc_id`)。
- Hugging Face 匿名限额提示 — 设置 `HF_TOKEN` 提高 Hub 限额;公开模型通常不设置也能下载。

## 许可证

尚未声明许可证。在添加 `LICENSE` 文件之前,默认版权归作者所有,未授予复用权利。
