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

## Unified SQLite State Migration

The default remains `COGDOC_STATE_BACKEND=jsonl`. Do not change the backend before the migration has completed and passed verification. Stop the API, workers, and every other process that can write sessions, jobs, feedback, analysis, derived knowledge, or retrieval feedback, then run the migration against the same instance in this order:

```bash
python scripts/migrate_state.py
python scripts/migrate_state.py --apply
python scripts/migrate_state.py --verify-only
```

The first command is a dry run and must not modify state. `--apply` acquires the same-instance migration lock, copies the existing JSONL state while preserving sessions and jobs, builds a temporary unified database, performs a full canonical-record comparison, and atomically replaces `state.db` only after every store matches. `--verify-only` independently compares the committed SQLite state with the canonical source records. Only after all three commands succeed should you set:

```bash
COGDOC_STATE_BACKEND=sqlite
```

Start the service and check `/readyz`, session history, outstanding/completed index jobs, feedback counts, derived knowledge, and a representative retrieval-feedback query. Keep `state.db.pre-unified-*.bak` and the original JSONL files for the entire rollback window; they are recovery artifacts, not files to clean up immediately.

If dry-run, apply, or verification fails, keep the service stopped and do not switch the backend. Capture the command's JSON error, confirm that no stale migration process owns the instance lock, check free disk space and permissions for the data directory, and resolve malformed or duplicate canonical records before rerunning the dry run. Never promote a temporary database manually.

To roll back after a failed SQLite startup or post-migration check:

1. Stop the API and all state writers.
2. Set `COGDOC_STATE_BACKEND=jsonl` (or remove the SQLite override).
3. Preserve the failed `state.db` for diagnosis; do not overwrite the retained JSONL files.
4. If the unified database replaced a pre-existing `state.db`, restore the matching `state.db.pre-unified-*.bak` only for components that still require that legacy database.
5. Restart the service, verify sessions/jobs and feedback state from JSONL, and repeat the migration from dry-run after the cause is fixed.

The migration lock only serializes cooperating migration processes for one instance; it does not make live application writes safe. Stopping all writers is therefore a required operational precondition.

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
