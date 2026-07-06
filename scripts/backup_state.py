import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cogdoc.config.settings import get_settings


DEFAULT_BACKUP_DIR = ROOT / "backups"
MANIFEST_NAME = "backup_manifest.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _arcname(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_paths(
    *,
    include_traces: bool,
    include_env: bool,
    extra_paths: Iterable[Path],
) -> list[Path]:
    settings = get_settings()
    paths = [Path(settings.cogdoc_data_dir)]
    if include_traces:
        paths.append(Path(settings.cogdoc_trace_dir))
    if include_env:
        paths.append(ROOT / ".env")
    paths.extend(extra_paths)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = path if path.is_absolute() else ROOT / path
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def build_manifest(paths: list[Path], archive_name: str, include_env: bool) -> dict:
    settings = get_settings()
    items = []
    for path in paths:
        item = {
            "path": _arcname(path),
            "type": "dir" if path.is_dir() else "file",
            "size_bytes": _path_size(path),
        }
        if path.is_file():
            item["sha256"] = _sha256(path)
        items.append(item)

    return {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_name,
        "project_root": str(ROOT),
        "data_dir": str(Path(settings.cogdoc_data_dir)),
        "trace_dir": str(Path(settings.cogdoc_trace_dir)),
        "includes_env": include_env,
        "items": items,
        "restore_hint": (
            "Stop CogDoc processes, extract this archive at the project root, "
            "then run make check and make smoke-api."
        ),
    }


def create_backup(
    output_dir: Path,
    *,
    name: str | None,
    include_traces: bool,
    include_env: bool,
    extra_paths: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = name or f"cogdoc-backup-{_timestamp()}.tar.gz"
    if not archive_name.endswith(".tar.gz"):
        archive_name = f"{archive_name}.tar.gz"
    archive_path = (output_dir / archive_name).resolve()
    if archive_path.exists():
        raise FileExistsError(f"备份文件已存在: {archive_path}")

    paths = collect_paths(
        include_traces=include_traces,
        include_env=include_env,
        extra_paths=extra_paths,
    )
    if not paths:
        raise FileNotFoundError("没有找到可备份的路径")

    manifest = build_manifest(paths, archive_path.name, include_env)
    with tempfile.TemporaryDirectory(prefix="cogdoc-backup-") as tmp:
        manifest_path = Path(tmp) / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(manifest_path, arcname=MANIFEST_NAME)
            for path in paths:
                archive.add(path, arcname=_arcname(path), recursive=True)
    return archive_path


def print_summary(archive_path: Path) -> None:
    size_mb = archive_path.stat().st_size / 1024 / 1024
    print(f"备份完成: {archive_path}")
    print(f"大小: {size_mb:.2f} MB")
    print("恢复提示: 停止服务后在项目根目录解压，再运行 make check 和 make smoke-api")


def main() -> int:
    parser = argparse.ArgumentParser(description="备份 CogDoc 本地运行状态")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help="备份输出目录，默认 backups/",
    )
    parser.add_argument("--name", default=None, help="备份文件名，可省略 .tar.gz")
    parser.add_argument(
        "--include-traces",
        action="store_true",
        default=True,
        help="包含 logs/traces（默认包含）",
    )
    parser.add_argument(
        "--no-traces",
        action="store_false",
        dest="include_traces",
        help="不包含 logs/traces",
    )
    parser.add_argument(
        "--include-env",
        action="store_true",
        help="包含 .env；注意其中可能有 API key",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="额外加入备份的文件或目录，可重复传入",
    )
    args = parser.parse_args()

    archive_path = create_backup(
        args.output_dir,
        name=args.name,
        include_traces=args.include_traces,
        include_env=args.include_env,
        extra_paths=args.extra,
    )
    print_summary(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
