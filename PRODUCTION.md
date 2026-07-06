# CogDoc Backup and Restore

This document covers how to back up local CogDoc runtime state, restore it, and decide when an index rebuild is required.

## Backup and Restore

State worth backing up:

- `data/kb/`: knowledge-base registry, source PDFs, generation state, and ingest journal.
- `data/chroma_db/`: vector collections.
- `data/bm25_db/`: BM25 registry and native index bytes.
- `data/manifests/`: manifests and index contract snapshots.
- `data/state.db`: sessions and index jobs.
- `data/feedback/`: feedback and bad cases.
- `logs/traces/`: request traces, if you need debugging or audit history.

Restore order:

1. Stop the API and frontend processes.
2. Restore `data/` and any retained `logs/traces/`.
3. Run `make check` to verify the native extension symbols.
4. Run `make smoke-api` to verify the API skeleton.
5. Start the service and check `/readyz`, `/v1/knowledge-bases`, and the target KB's sources/chunks.

A backup is not proven until you have tested a restore. After index-format or chunk-identity changes, run a small restore drill.

Create a local backup:

```bash
make backup
```

By default this archives `data/` and `logs/traces/` into `backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz` and includes `backup_manifest.json` inside the archive. `backups/` is ignored by Git.

To include `.env`:

```bash
python scripts/backup_state.py --include-env
```

`.env` may contain API keys. Store it only in a controlled location and do not commit or share it. To restore, extract the archive at the project root after stopping the service.

## Index Format and Migration

Treat these changes as index-contract changes:

- `CHUNK_IDENTITY_BASE_VERSION` or chunking parameter changes.
- `INDEX_BUILD_VERSION` changes.
- Parser, tokenizer, embedding model, or BM25 artifact format changes.
- Chroma collection naming or generation layout changes.

Rules:

- Reusable changes: API, frontend, and prompt-only changes usually do not require an index rebuild.
- Rebuild-required changes: chunk identity, parser/tokenizer, embedding model, or BM25 bytes format changes.
- If a migration is needed, state whether a rebuild is required, whether old generations remain compatible, and how to roll back after failure.
