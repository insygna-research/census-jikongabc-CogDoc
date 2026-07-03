import json
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from cogdoc.config.settings import get_settings

_BAD_CASE_TYPES = {"thumbs_down", "correction"}


# 返回当前 UTC 时间字符串。
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 反馈追加落 jsonl；点踩/纠错另写 bad_cases，供评测集自我进化。
class FeedbackStore:
    # 反馈追加落 jsonl；点踩/纠错另写 bad_cases，供评测集自我进化。
    def __init__(
        self,
        feedback_path: str | None = None,
        bad_cases_path: str | None = None,
    ):
        settings = get_settings()
        self._feedback_path = feedback_path or settings.feedback_log_path
        self._bad_cases_path = bad_cases_path or settings.bad_cases_path
        self._lock = RLock()
        for path in (self._feedback_path, self._bad_cases_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # 记录结果。
    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback_id = uuid4().hex
        entry = {"feedback_id": feedback_id, "created_at": _now_iso(), **payload}
        is_bad_case = payload.get("feedback") in _BAD_CASE_TYPES
        with self._lock:
            self._append(self._feedback_path, entry)
            if is_bad_case:
                self._append(self._bad_cases_path, entry)
        return {"feedback_id": feedback_id, "is_bad_case": is_bad_case}

    # 追加。
    def _append(self, path: str, entry: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
