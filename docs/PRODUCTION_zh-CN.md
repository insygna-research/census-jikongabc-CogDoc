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
