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

默认会把 `data/` 和 `logs/traces/` 打成 `backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz`，并在归档里写入 `backup_manifest.json`。`backups/` 已被 Git 忽略。

如需同时备份 `.env`：

```bash
python scripts/backup_state.py --include-env
```

`.env` 可能包含 API key，只应保存到受控位置，不要提交或共享。恢复时在项目根目录解压对应归档即可。

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
