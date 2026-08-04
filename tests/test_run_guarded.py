import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_guarded.py"


def _run(tmp_path, code, *, timeout=2.0, grace=0.1):
    diagnostic = tmp_path / "timeout.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--timeout",
            str(timeout),
            "--grace",
            str(grace),
            "--diagnostic",
            str(diagnostic),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed, diagnostic


def test_normal_exit_is_transparent(tmp_path):
    completed, diagnostic = _run(tmp_path, "print('normal-output')")
    assert completed.returncode == 0
    assert "normal-output" in completed.stdout
    assert not diagnostic.exists()


def test_failure_exit_code_is_transparent(tmp_path):
    completed, diagnostic = _run(tmp_path, "import sys; sys.exit(7)")
    assert completed.returncode == 7
    assert not diagnostic.exists()


def test_timeout_kills_group_and_writes_diagnostic(tmp_path):
    code = "import subprocess,sys,time; print('before-timeout', flush=True); subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)"
    completed, diagnostic = _run(tmp_path, code, timeout=0.2)
    assert completed.returncode == 124
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["exit_reason"] == "timeout"
    assert payload["timeout_elapsed_seconds"] >= 0.2
    assert "before-timeout" in payload["stdout_tail"]
    assert payload["signal_attempts"][0]["signal"] == "SIGTERM"
    assert payload["process_snapshot"]["root_pid"] == payload["root_pid"]
