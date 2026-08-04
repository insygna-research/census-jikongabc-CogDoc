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

By default this archives `data/` and `logs/traces/` into `backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz`. It does **not** include `.env`. The archive's versioned `backup_manifest.json` records every file's relative path, byte size, SHA-256, and backup creation time, plus non-secret source-root configuration metadata. Backup output remains human-readable for compatibility; pass `--json` when automation needs one JSON object.

`v2` archives receive full per-file integrity verification. The restore tool also accepts legacy `v1` archives after validating safe paths, member types, declared roots, aggregate sizes, and hashes that exist for top-level files. Because `v1` has no per-file hashes inside directory roots, its result is explicitly reported as `verification_level: "degraded"` with a warning; it must not be treated as cryptographic proof of all restored content.

To include `.env`:

```bash
python scripts/backup_state.py --include-env
```

`.env` may contain API keys. Store it only in a controlled location and do not commit or share it. Prefer restoring secrets independently from a secret manager.

Verify an archive without changing runtime state:

```bash
python scripts/restore_state.py backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz --verify-only
```

Restore into an empty drill directory, then inspect the restored `data/` and trace roots:

```bash
mkdir -p /srv/cogdoc-restore-drill
python scripts/restore_state.py backups/cogdoc-backup-YYYYMMDD-HHMMSS.tar.gz \
  --target /srv/cogdoc-restore-drill
```

For an in-place recovery, stop every writer first and use `--target . --force`. A non-empty target is rejected without `--force`. Forced recovery replaces only top-level paths present in the archive; unrelated project files remain in place. The restore validates member types and paths, extracts into a sibling temporary directory, verifies the complete manifest, and only then promotes state with atomic moves and rollback on promotion failure.

Run a restore drill at least once per release and after every index-contract change. Record archive size, verification duration, restore duration, `/readyz` result, KB/source counts, and a representative retrieval result. The local archive is a crash-consistent file copy, not a coordinated database snapshot: stop writers before backup when a zero-loss restore point matters. Therefore the achievable RPO is the time since the last completed, quiesced backup; changes after it are not recoverable. RTO includes archive transfer, full SHA-256 verification, extraction, native/index compatibility checks, and any required index rebuild, so large Chroma/BM25 stores can dominate recovery time. Set operational RPO/RTO targets only after measuring them with production-sized restore drills.

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
