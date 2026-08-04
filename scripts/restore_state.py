import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "backup_manifest.json"
MANIFEST_VERSION = "v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RestoreError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _safe_name(name: str) -> str:
    candidate = name[2:] if name.startswith("./") else name
    pure = PurePosixPath(candidate)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or pure.is_absolute()
        or ".." in pure.parts
        or (name != "." and (not candidate or pure.as_posix() != candidate))
    ):
        raise RestoreError("UNSAFE_ARCHIVE", f"不安全的归档路径: {name!r}")
    return name


def _valid_time(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _validate_manifest(manifest: object) -> tuple[dict[str, dict], set[str], list[str]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise RestoreError("INVALID_MANIFEST", "仅支持 v2 备份 manifest")
    if not _valid_time(manifest.get("created_at")):
        raise RestoreError("INVALID_MANIFEST", "manifest.created_at 无效")
    raw_files = manifest.get("files")
    raw_directories = manifest.get("directories")
    source = manifest.get("source")
    roots = source.get("roots") if isinstance(source, dict) else None
    if not isinstance(raw_files, list) or not isinstance(raw_directories, list):
        raise RestoreError("INVALID_MANIFEST", "manifest 文件清单无效")
    if not isinstance(roots, list) or not roots:
        raise RestoreError("INVALID_MANIFEST", "manifest 根路径清单无效")

    files: dict[str, dict] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise RestoreError("INVALID_MANIFEST", "manifest 文件条目无效")
        name = _safe_name(item.get("path", ""))
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            name == MANIFEST_NAME
            or name in files
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or not _valid_time(item.get("created_at"))
        ):
            raise RestoreError("INVALID_MANIFEST", f"manifest 文件条目无效: {name}")
        files[name] = item
    directories = {_safe_name(item) for item in raw_directories if isinstance(item, str)}
    if len(directories) != len(raw_directories) or directories.intersection(files):
        raise RestoreError("INVALID_MANIFEST", "manifest 目录清单无效")
    root_names = []
    for root in roots:
        if not isinstance(root, dict):
            raise RestoreError("INVALID_MANIFEST", "manifest 根路径条目无效")
        root_names.append(_safe_name(root.get("archive_path", "")))
    declared = set(files).union(directories)
    if any(root not in declared for root in root_names) or any(
        not any(name == root or name.startswith(f"{root}/") for root in root_names)
        for name in declared
    ):
        raise RestoreError("INVALID_MANIFEST", "manifest 路径不属于声明的状态根")
    if manifest.get("file_count") != len(files) or manifest.get(
        "total_size_bytes"
    ) != sum(item["size_bytes"] for item in files.values()):
        raise RestoreError("INVALID_MANIFEST", "manifest 汇总字段不匹配")
    return files, directories, root_names


def _validate_v1_manifest(
    manifest: object, members: dict[str, tarfile.TarInfo]
) -> tuple[dict[str, dict], set[str], list[str]]:
    error = "V1_COMPATIBILITY_ERROR"
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "v1":
        raise RestoreError(error, "旧 v1 manifest 结构无效")
    if not _valid_time(manifest.get("created_at")) or not isinstance(
        manifest.get("items"), list
    ):
        raise RestoreError(error, "旧 v1 manifest 缺少有效 created_at/items")
    roots: list[str] = []
    root_items: dict[str, dict] = {}
    for item in manifest["items"]:
        if not isinstance(item, dict):
            raise RestoreError(error, "旧 v1 manifest 包含无效条目")
        name = _safe_name(item.get("path", ""))
        kind = item.get("type")
        size = item.get("size_bytes")
        if (
            name == MANIFEST_NAME
            or name in root_items
            or kind not in {"dir", "file"}
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise RestoreError(error, f"旧 v1 manifest 条目无法安全解释: {name}")
        if kind == "file" and (
            not isinstance(item.get("sha256"), str)
            or not SHA256_RE.fullmatch(item["sha256"])
        ):
            raise RestoreError(error, f"旧 v1 文件缺少有效哈希: {name}")
        roots.append(name)
        root_items[name] = item
    if not roots:
        raise RestoreError(error, "旧 v1 manifest 没有状态根")
    for name, member in members.items():
        if not any(name == root or name.startswith(f"{root}/") for root in roots):
            raise RestoreError(error, f"旧 v1 归档存在未声明成员: {name}")
    for root, item in root_items.items():
        member = members.get(root)
        if member is None or (item["type"] == "dir") != member.isdir():
            raise RestoreError(error, f"旧 v1 状态根缺失或类型不匹配: {root}")
        if item["type"] == "file":
            if member.size != item["size_bytes"]:
                raise RestoreError(error, f"旧 v1 文件大小不匹配: {root}")
        else:
            actual_size = sum(
                candidate.size
                for name, candidate in members.items()
                if candidate.isfile() and name.startswith(f"{root}/")
            )
            if actual_size != item["size_bytes"]:
                raise RestoreError(error, f"旧 v1 目录汇总大小不匹配: {root}")
    file_expectations = {
        name: item for name, item in root_items.items() if item["type"] == "file"
    }
    directories = {name for name, member in members.items() if member.isdir()}
    return file_expectations, directories, roots


def _extract_verified(archive_path: Path, payload: Path) -> tuple[dict, dict]:
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            members: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                name = _safe_name(member.name)
                if name in members:
                    raise RestoreError("UNSAFE_ARCHIVE", f"重复归档成员: {name}")
                if not member.isfile() and not member.isdir():
                    raise RestoreError("UNSAFE_ARCHIVE", f"不支持的归档成员: {name}")
                members[name] = member
            manifest_member = members.pop(MANIFEST_NAME, None)
            if manifest_member is None or not manifest_member.isfile():
                raise RestoreError("INVALID_MANIFEST", "归档缺少 backup_manifest.json")
            stream = archive.extractfile(manifest_member)
            if stream is None:
                raise RestoreError("INVALID_MANIFEST", "无法读取 manifest")
            try:
                manifest = json.loads(stream.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RestoreError("INVALID_MANIFEST", f"manifest 解析失败: {exc}") from exc
            archive_files = {name for name, member in members.items() if member.isfile()}
            archive_directories = {
                name for name, member in members.items() if member.isdir()
            }
            version = manifest.get("schema_version") if isinstance(manifest, dict) else None
            if version == MANIFEST_VERSION:
                files, directories, _ = _validate_manifest(manifest)
                if archive_files != set(files) or archive_directories != directories:
                    raise RestoreError("INTEGRITY_ERROR", "归档成员与 manifest 不一致")
                verification_level = "full"
                warning = None
            elif version == "v1":
                files, directories, _ = _validate_v1_manifest(manifest, members)
                verification_level = "degraded"
                warning = (
                    "v1 manifest 没有目录内逐文件哈希；已验证安全路径、成员类型、"
                    "汇总大小及顶层文件哈希，但无法证明目录内同大小内容未被修改"
                )
            else:
                raise RestoreError(
                    "UNSUPPORTED_MANIFEST_VERSION",
                    f"无法安全恢复 manifest 版本 {version!r}；支持 v1（降级校验）和 v2",
                )

            for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
                (payload / directory).mkdir(parents=True, exist_ok=True)
            for name, expected in sorted(files.items()):
                source = archive.extractfile(members[name])
                if source is None:
                    raise RestoreError("INTEGRITY_ERROR", f"无法读取归档文件: {name}")
                destination = payload / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with destination.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                if size != members[name].size:
                    raise RestoreError("INTEGRITY_ERROR", f"文件校验失败: {name}")
                if name in files and (
                    size != files[name]["size_bytes"]
                    or digest.hexdigest() != files[name]["sha256"]
                ):
                    raise RestoreError("INTEGRITY_ERROR", f"文件校验失败: {name}")
            return manifest, {
                "verification_level": verification_level,
                "degraded": verification_level == "degraded",
                "warning": warning,
                "file_count": len(archive_files),
                "total_size_bytes": sum(members[name].size for name in archive_files),
            }
    except RestoreError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise RestoreError("INVALID_ARCHIVE", str(exc)) from exc


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _target_nonempty(target: Path) -> bool:
    if not _exists(target):
        return False
    if target.is_symlink() or not target.is_dir():
        return True
    return next(target.iterdir(), None) is not None


def _promote(payload: Path, target: Path, force: bool, work: Path) -> bool:
    existed = _exists(target)
    if not existed:
        os.replace(payload, target)
        return False
    nonempty = _target_nonempty(target)
    if nonempty and not force:
        raise RestoreError("TARGET_NOT_EMPTY", f"目标非空，需显式 --force: {target}")
    rollback = work / "rollback"
    rollback.mkdir()
    if not target.is_dir() or target.is_symlink() or not nonempty:
        old = rollback / "target"
        os.replace(target, old)
        try:
            os.replace(payload, target)
        except Exception:
            os.replace(old, target)
            raise
        shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink(missing_ok=True)
        return nonempty

    moved_old: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        for source in sorted(payload.iterdir(), key=lambda path: path.name):
            destination = target / source.name
            old = rollback / source.name
            if _exists(destination):
                os.replace(destination, old)
                moved_old.append((old, destination))
            os.replace(source, destination)
            installed.append((destination, source))
    except Exception:
        for destination, source in reversed(installed):
            if _exists(destination):
                os.replace(destination, source)
        for old, destination in reversed(moved_old):
            if _exists(old):
                os.replace(old, destination)
        raise
    shutil.rmtree(rollback, ignore_errors=True)
    return True


def restore_archive(
    archive_path: Path,
    target: Path | None = None,
    *,
    force: bool = False,
    verify_only: bool = False,
) -> dict:
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise RestoreError("ARCHIVE_NOT_FOUND", f"备份文件不存在: {archive_path}")
    if verify_only:
        with tempfile.TemporaryDirectory(prefix="cogdoc-verify-") as tmp:
            manifest, verification = _extract_verified(
                archive_path, Path(tmp) / "payload"
            )
    else:
        target = (target or ROOT).expanduser().absolute()
        if not target.parent.is_dir():
            raise RestoreError("TARGET_PARENT_MISSING", f"目标父目录不存在: {target.parent}")
        if _target_nonempty(target) and not force:
            raise RestoreError("TARGET_NOT_EMPTY", f"目标非空，需显式 --force: {target}")
        work = Path(tempfile.mkdtemp(prefix=".cogdoc-restore-", dir=target.parent))
        try:
            payload = work / "payload"
            payload.mkdir()
            manifest, verification = _extract_verified(archive_path, payload)
            replaced = _promote(payload, target, force, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {
        "ok": True,
        "operation": "verify" if verify_only else "restore",
        "archive": str(archive_path),
        "target": None if verify_only else str(target),
        "schema_version": manifest["schema_version"],
        "created_at": manifest["created_at"],
        "file_count": verification["file_count"],
        "total_size_bytes": verification["total_size_bytes"],
        "verification_level": verification["verification_level"],
        "degraded": verification["degraded"],
        **({"warning": verification["warning"]} if verification["warning"] else {}),
        **({} if verify_only else {"replaced_existing": replaced}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="校验或恢复 CogDoc 状态备份")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--target", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    operation = "verify" if args.verify_only else "restore"
    try:
        result = restore_archive(
            args.archive,
            args.target,
            force=args.force,
            verify_only=args.verify_only,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except RestoreError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "operation": operation,
                    "error": {"code": exc.code, "message": str(exc)},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "operation": operation,
                    "error": {"code": "RESTORE_FAILED", "message": str(exc)},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
