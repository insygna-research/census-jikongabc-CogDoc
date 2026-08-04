#!/usr/bin/env python3
"""Run a command with a hard deadline and persist timeout diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class TailBuffer:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()

    def append(self, chunk: bytes) -> None:
        self.data.extend(chunk)
        if len(self.data) > self.limit:
            del self.data[: len(self.data) - self.limit]

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def _relay(stream, fd: int, tail: TailBuffer) -> None:
    while True:
        chunk = stream.read(8192)
        if not chunk:
            return
        tail.append(chunk)
        try:
            os.write(fd, chunk)
        except OSError:
            pass


def _read(path: Path, limit: int = 65536) -> str | None:
    try:
        return path.read_bytes()[:limit].decode("utf-8", errors="replace")
    except OSError:
        return None


def _status(pid: int) -> dict[str, str]:
    raw = _read(Path("/proc") / str(pid) / "status") or ""
    wanted = {"Name", "State", "Pid", "PPid", "Tgid", "Threads", "VmRSS", "VmSize"}
    values = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            values[key.lower()] = value.strip()
    return values


def _proc_snapshot(root_pid: int) -> dict:
    if not Path("/proc").is_dir():
        return {"available": False, "reason": "/proc is unavailable"}
    queue = [root_pid]
    seen: set[int] = set()
    processes = []
    while queue and len(seen) < 256:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        base = Path("/proc") / str(pid)
        status = _status(pid)
        if not status:
            continue
        cmdline = _read(base / "cmdline")
        tasks = []
        try:
            task_dirs = sorted((base / "task").iterdir(), key=lambda item: int(item.name))
        except OSError:
            task_dirs = []
        children: set[int] = set()
        for task_dir in task_dirs[:256]:
            task_status = _status(int(task_dir.name))
            child_text = _read(task_dir / "children") or ""
            children.update(int(value) for value in child_text.split() if value.isdigit())
            tasks.append(
                {
                    "tid": int(task_dir.name),
                    "name": task_status.get("name"),
                    "state": task_status.get("state"),
                    "wchan": (_read(task_dir / "wchan") or "").strip() or None,
                }
            )
        queue.extend(sorted(children - seen))
        processes.append(
            {
                "pid": pid,
                "status": status,
                "cmdline": (cmdline or "").replace("\x00", " ").strip(),
                "children": sorted(children),
                "threads": tasks,
            }
        )
    return {"available": True, "root_pid": root_pid, "processes": processes, "truncated": bool(queue)}


def _signal_group(process: subprocess.Popen, sig: signal.Signals) -> dict:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
        return {"signal": sig.name, "sent": True}
    except (OSError, ProcessLookupError) as exc:
        return {"signal": sig.name, "sent": False, "error": str(exc)}


def _group_alive(process: subprocess.Popen) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop(process: subprocess.Popen, grace: float) -> list[dict]:
    attempts = [_signal_group(process, signal.SIGTERM)]
    deadline = time.monotonic() + grace
    while _group_alive(process) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _group_alive(process):
        attempts.append(_signal_group(process, signal.SIGKILL))
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    return attempts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _default_diagnostic(command: list[str]) -> Path:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(command[0]).name).strip("-") or "command"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("artifacts/reliability") / f"{stamp}-{name}-timeout.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--grace", type=float, default=5.0)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--tail-bytes", type=int, default=65536)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.timeout <= 0 or args.grace < 0 or args.tail_bytes <= 0:
        parser.error("timeout and tail-bytes must be positive; grace cannot be negative")

    stdout_tail = TailBuffer(args.tail_bytes)
    stderr_tail = TailBuffer(args.tail_bytes)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        print(f"run_guarded: could not start command: {exc}", file=sys.stderr)
        return 127

    threads = [
        threading.Thread(target=_relay, args=(process.stdout, 1, stdout_tail), daemon=True),
        threading.Thread(target=_relay, args=(process.stderr, 2, stderr_tail), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timeout_elapsed = time.monotonic() - started
        snapshot = _proc_snapshot(process.pid)
        signals = _stop(process, args.grace)
        for thread in threads:
            thread.join(timeout=1.0)
        diagnostic = args.diagnostic or _default_diagnostic(command)
        payload = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "cwd": str((args.cwd or Path.cwd()).resolve()),
            "exit_reason": "timeout",
            "timeout_seconds": args.timeout,
            "grace_seconds": args.grace,
            "timeout_elapsed_seconds": round(timeout_elapsed, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "root_pid": process.pid,
            "final_returncode": process.poll(),
            "signal_attempts": signals,
            "process_snapshot": snapshot,
            "stdout_tail": stdout_tail.text(),
            "stderr_tail": stderr_tail.text(),
        }
        try:
            _write_json(diagnostic, payload)
            print(f"run_guarded: timeout after {args.timeout}s; diagnostic={diagnostic}", file=sys.stderr)
        except OSError as exc:
            print(f"run_guarded: timeout after {args.timeout}s; diagnostic write failed: {exc}", file=sys.stderr)
        return 124
    for thread in threads:
        thread.join(timeout=1.0)
    return returncode if returncode >= 0 else 128 + abs(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
