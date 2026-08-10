# CogDoc 备份与恢复说明

本文档记录本地知识库状态如何备份、恢复，以及哪些索引变化需要重建。

## 备份与恢复

需要备份的状态：

- `data/kb/`：知识库 registry、源 PDF、generation state、入库 journal。
- `data/chroma_db/`：向量集合。
- `data/bm25_db/`：BM25 registry 与 native index bytes。
- `data/manifests/`：manifest 与索引契约快照。
- `data/state.db`：sessions 与 index jobs。
- `data/feedback/`：feedback 与 bad cases。
- `logs/traces/`：请求 trace，可按保留策略裁剪。

恢复顺序：

1. 停止 API/前端进程。
2. 恢复 `data/` 目录和需要保留的 `logs/traces/`。
3. 运行 `make check` 确认 native extension 符号匹配。
4. 运行 `make smoke-api` 验证 API 骨架可用。
5. 启动服务后检查 `/readyz`、`/v1/knowledge-bases` 和目标 KB 的 sources/chunks。

没有做过恢复演练的备份不能视为有效备份。每次索引格式或 chunk identity 变化后，都应执行一次小规模 restore drill。

本地备份命令：

```bash
make backup
```

默认会把 `data/` 和 `logs/traces/` 打成 `backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz`，**不会**包含 `.env`。归档内版本化的 `backup_manifest.json` 会记录每个文件的相对路径、字节数、SHA-256、备份创建时间，以及不含密钥的源根目录配置元数据。为保持兼容，备份命令默认仍输出人类可读文本；自动化场景显式传入 `--json` 才输出单个 JSON 对象。

`v2` 归档执行逐文件完整性校验。恢复工具也兼容旧 `v1` 归档，并检查安全路径、成员类型、声明根目录、汇总大小及已有的顶层文件哈希。由于 `v1` 没有目录内逐文件哈希，结果会明确标记为 `verification_level: "degraded"` 并包含警告，不能将其视为全部恢复内容的密码学完整性证明。

如需同时备份 `.env`：

```bash
python scripts/backup_state.py --include-env
```

`.env` 可能包含 API key，只应保存到受控位置，不要提交或共享；优先从密钥管理系统独立恢复密钥。

仅校验归档、不修改运行状态：

```bash
python scripts/restore_state.py backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz --verify-only
```

先恢复到空的演练目录，再检查其中的 `data/` 与 trace 根目录：

```bash
mkdir -p /srv/cogdoc-restore-drill
python scripts/restore_state.py backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz \
  --target /srv/cogdoc-restore-drill
```

原地恢复前必须停止所有写入进程，再使用 `--target . --force`。非空目标在没有 `--force` 时会被拒绝；强制恢复只替换归档声明的顶层路径，不影响项目中的其他文件。恢复程序会拒绝路径穿越和非普通成员，先在目标同级临时目录解包并全量校验 manifest，成功后才以原子移动提交；提交失败会回滚原路径。

每个发布版本以及每次索引契约变化后至少执行一次恢复演练，并记录归档大小、校验耗时、恢复耗时、`/readyz`、KB/source 数量和代表性检索结果。本地归档是文件级崩溃一致副本，不是跨存储协调快照；要求零丢失恢复点时必须先停止写入。因此可实现的 RPO 等于距离最近一次已完成、静默备份的时间，之后的变更无法恢复。RTO 包括归档传输、全量 SHA-256 校验、解包、native/index 兼容性检查，以及必要时的索引重建；大型 Chroma/BM25 状态通常主导恢复时间。只有使用生产规模数据完成演练后，才能承诺具体 RPO/RTO。

## 统一 SQLite 状态迁移

默认后端仍为 `COGDOC_STATE_BACKEND=jsonl`。迁移完成并通过校验前不要切换后端。先停止 API、worker，以及所有可能写入 sessions、jobs、research plans、feedback、analysis、derived knowledge 或 retrieval feedback 的进程，再针对同一实例依次执行：

```bash
python scripts/migrate_state.py
python scripts/migrate_state.py --apply
python scripts/migrate_state.py --verify-only
```

第一条命令是 dry-run，不应修改任何状态。`--apply` 会获取同一实例的迁移锁，复制现有 JSONL 状态并保留 sessions/jobs，在临时库中构建统一 SQLite，完成全部 canonical record 对比后才原子替换 `state.db`。`--verify-only` 会独立比较已提交 SQLite 与 canonical 源记录。三步全部成功后，才设置：

```bash
COGDOC_STATE_BACKEND=sqlite
```

随后启动服务，检查 `/readyz`、会话历史、未完成/已完成索引任务、反馈数量、派生知识，以及一条代表性的检索反馈查询。在整个回滚窗口内保留 `state.db.pre-unified-*.bak` 和原始 JSONL；它们是恢复工件，不能在迁移后立即清理。

Research 证据执行以章节为恢复粒度。若服务在任务处于 `running` 时退出，启动过程会把执行中的章节重置为 `pending`，并把任务协调到 `paused`，必须由运维或用户显式恢复。报告生成会把每个原子需求重新送入闭集 Evidence Unit 校验，只有 `supported` 的 grounding ID 能进入章节生成；生成后的声明只依据本章精确证据接受审计，独立的需求覆盖审计还要求每个原子需求都由已支持且有引用的声明回答。声明与覆盖失败共享最多一次有界修复，修复后必须重新通过引用、声明和覆盖三道门。无证据、冲突、遗漏需求、校验失败、语义审计失败和生成失败都会成为报告中的显式缺口。若服务在 `generating` 时退出，任务会回到 `evidence_ready` 等待显式重试，并保留选择性重生成范围。状态库只保存有界证据预览、定位、公开引用账本、声明/覆盖审计摘要和渲染后的 Markdown 报告，不保存完整来源 chunk 或模型声明文本。

每次证据/报告 attempt 都持久化 attempt ID、可轮换 lease、阶段截止时间，以及检索查询、候选文档、模型调用和模型输入累计字符的原子预算。恢复执行必定轮换 lease，因此正在排空或迟到的 worker 不能继续预扣资源，也不能提交旧输出。已准入的排队/运行总量受 `COGDOC_RESEARCH_MAX_PENDING` 限制；超过上限的启动/生成请求返回带 `Retry-After` 的 `503`。暂停和取消会原子作废证据与报告 lease、通知活动 worker，并取消尚未开始的 future。截止时间或预算耗尽会持久化并 fail-closed。

自动规划的来源读取与模型工作使用独立的有界 daemon executor（`COGDOC_RESEARCH_PLANNING_WORKERS` / `COGDOC_RESEARCH_PLANNING_MAX_PENDING`），不占用共享 API offload 池；前后的短状态库操作仍使用共享池。绝对截止时间覆盖排队、来源读取和模型执行。进入 lifespan 关闭后，服务会通知所有已注册规划控制器、取消排队任务；若不透明的进程内来源读取仍未排空，则延后关闭 runtime 和释放进程锁。`make serve` 还通过 `UVICORN_GRACEFUL_SHUTDOWN_SECONDS`（默认 `15`）设置 Uvicorn 活动请求的优雅关闭上限；使用其他启动器时必须配置等价的有限上限，否则 Uvicorn 可能在进入 lifespan 关闭前无限等待活动 HTTP handler。原始 socket 断开不是所有 ASGI 服务器都会转成 handler 取消信号，因此这种情况下仍以专用容量和规划绝对截止时间作为外层边界。

自动规划及证据/报告生成中的标准工厂 `ChatOpenAI` 调用，会在全新的 spawn 子进程中重建，并关闭传输层重试。监督器会用规划或持久阶段的剩余截止时间收紧单次调用时限；子进程存活时轮询进程内停止信号与单调截止时间，在准入前和子进程回收后执行权威持久检查，并始终 join 和回收子进程。超时、暂停、取消或关闭会先发送 terminate，超过 `COGDOC_RESEARCH_PROVIDER_KILL_GRACE_SECONDS` 后升级为 kill。应按后台 Research attempt 可用的 provider 容量设置 `COGDOC_RESEARCH_PROVIDER_WORKERS` 与 `COGDOC_RESEARCH_PROVIDER_MAX_PENDING`，让 `COGDOC_RESEARCH_PROVIDER_CALL_TIMEOUT_SECONDS` 小于上游负载均衡器超时，并把 `COGDOC_RESEARCH_PROVIDER_IPC_MAX_BYTES` 视为 fail-closed 的响应信封上限。若已识别的 `ChatOpenAI` 客户端无法转换为安全子进程调用，在 `COGDOC_RESEARCH_LLM_PROCESS_ISOLATION_ENABLED=true` 时会 fail-closed；不透明或非标准客户端仍走有界 daemon 兼容路径，只能在检查点协作式停止。优雅关闭会在结束应用 lifespan 前作废全部活动 lease，兼容路径的迟到结果无法提交。

超时计时从 spawn 前开始，包含 provider 槽位等待与子进程生命周期。Python 本地 spawn bootstrap 和有界 IPC 信封解码属于可信的准入/序列化边界：其耗时会计入 deadline，但解释器无法异步抢占这些短同步操作本身。工厂调用会先转成有大小上限的纯字节配方，再进入 spawn，以保持该边界可预测。

这层隔离只终止本地 HTTP 客户端进程。已经收到请求的远端 API 或 Ollama 服务可能继续计算和计费，因此仍需 provider 侧请求 ID、预算及账单告警。检索、重排、嵌入、Hugging Face 模型加载、Torch kernel 与 native/Rust 调用仍在进程内；Research 控制器会在这些调用前后检查截止时间，但无法强制抢占阻塞中的调用。不得把本版本描述为任意 provider 沙箱或全流水线隔离。

集合视图应轮询 `GET /v1/research-jobs/summaries`，而不是兼容保留的完整列表接口。摘要接口使用有界 keyset 分页（`limit` 与 opaque `cursor`），返回 ETag、支持 `If-None-Match` 命中后的 `304`，且不包含章节、证据、报告和历史正文。只有用户显式选中后才获取一个任务详情及其报告。应监控 `cogdoc_research_lifecycle_total`、`cogdoc_research_background_total`、`cogdoc_research_background_in_progress`、`cogdoc_research_terminations_total`、章节候选/证据直方图以及覆盖/声明审计计数器；指标标签均为低基数闭集，任务 ID 只进入结构化日志字段。

Research 发布是受独立 `COGDOC_EVAL_REVIEW_API_KEYS` 审核凭据保护的乐观并发状态转换；服务端只保存审核者/发布者的 key 指纹。每次证据执行都会冻结索引 generation/build/chunk identity、来源 SHA-256、已批准派生知识版本、检索调权版本以及检索/校验契约版本；任何漂移都会把证据标为 stale，并阻止生成、审阅与发布。显式刷新会先归档旧报告，清空所有章节的证据与审计结果，再基于新快照执行全量检索。已生成章节必须标记为 `approved`，被阻断章节必须显式标记为 `accepted_gap` 并填写非空理由；任何 `changes_requested` 决定都必须附带修订要求，且只能通过同一检索与校验链路重新生成。系统最多归档十个完整报告版本，审阅历史最多保留 100 个事件。只有退回或旧版未审计章节会消耗检索、校验和生成资源；保留章节与新章节的局部账本会重新编号、换算偏移并合成为经过校验的全局账本。v2 artifact SHA-256 精确绑定 Markdown、严格引用账本、可追踪 provenance、有界聚合/逐章声明与需求覆盖审计、证据身份/文本哈希承诺、报告版本和生成时间；独立 publication SHA-256 再把该 artifact 与精确审阅历史、逐章决定、发布时间和审核 key 指纹绑定。确定性 ZIP 包含 `report.md`、`citation-ledger.json`、`provenance.json`、`verification.json` 及逐文件哈希 manifest。旧版已发布 Markdown 仍可通过 `X-CogDoc-Integrity: legacy-unverified` 下载，但不能生成验证包；任何畸形或被篡改的 artifact 都不会返回正文。

dry-run、apply 或 verify 任一步失败时，保持服务停止且不要切换后端。保存命令输出的 JSON 错误，确认没有遗留迁移进程占用实例锁，检查数据目录的剩余空间和权限，并修复 malformed/duplicate canonical records 后重新从 dry-run 开始。禁止手工提升临时数据库。

SQLite 启动失败或迁移后检查失败时，按以下步骤回滚：

1. 停止 API 和所有状态写入进程。
2. 设置 `COGDOC_STATE_BACKEND=jsonl`，或删除 SQLite 覆盖配置。
3. 保留失败的 `state.db` 用于排障，不要覆盖留存的 JSONL。
4. 如果统一数据库替换了既有 `state.db`，仅在仍依赖旧数据库的组件需要时恢复对应的 `state.db.pre-unified-*.bak`。
5. 重启服务，从 JSONL 验证 sessions/jobs 和反馈状态；修复根因后重新从 dry-run 开始迁移。

迁移锁只能串行化同一实例中遵循该锁的迁移进程，不能保证在线应用写入安全，因此停止全部写入进程是强制运维前提。

## 索引格式与迁移

以下变化必须视为索引契约变化：

- `CHUNK_IDENTITY_BASE_VERSION` 或 chunk 参数变化。
- `INDEX_BUILD_VERSION` 变化。
- parser、tokenizer、embedding model、BM25 artifact 格式变化。
- Chroma collection 命名或 generation layout 变化。

规则：

- 可复用变化：只改 API/前端/Prompt，不改 chunk/index/native artifact，可不强制重建。
- 强制重建变化：chunk identity、parser/tokenizer、embedding model、BM25 bytes 格式变化。
- 迁移说明必须写清楚：是否强制重建、是否兼容旧 generation、失败如何回滚。
