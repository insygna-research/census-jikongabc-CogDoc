import re

from cogdoc.agents.router import FORCED_TASK_TYPES


FORCED_MODE_PATTERN = re.compile(
    rf"^/({'|'.join(re.escape(task) for task in FORCED_TASK_TYPES)})(?:\s+(.*))?$",
    re.I,
)


def parse_forced_mode(user_input: str) -> tuple[str | None, str]:
    match = FORCED_MODE_PATTERN.match(user_input.strip())
    if not match:
        return None, user_input
    return match.group(1).lower(), (match.group(2) or "").strip()
