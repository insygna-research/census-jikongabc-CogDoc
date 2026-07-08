from datetime import datetime, timezone


# 返回当前协调世界时时间字符串。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
