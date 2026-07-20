from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


# 定义记忆容量策略。
@dataclass(frozen=True)
class MemoryPolicy:
    short_term_message_limit: int = 12
    short_term_char_limit: int = 6000
    mid_term_char_limit: int = 4000
    long_term_fact_limit: int = 64
    context_long_term_limit: int = 8
    message_preview_chars: int = 300
    memory_retrieval_enabled: bool = True
    memory_semantic_enabled: bool = True
    memory_retrieval_short_limit: int = 8
    memory_retrieval_mid_limit: int = 4
    memory_retrieval_recent_pin: int = 4
    memory_semantic_include_short: bool = False
    memory_rrf_k: float = 60.0
    memory_recency_weight: float = 1.0
    memory_lexical_weight: float = 1.4
    memory_semantic_weight: float = 1.6
    memory_importance_weight: float = 0.8
    memory_mid_priority_weight: float = 0.8


# 压缩单段记忆文本。
def _compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# 规范化有效消息。
def _normalise_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised = []
    for message in messages:
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        normalised.append(dict(message, content=content))
    return normalised


# 按双预算裁剪短期记忆。
def compact_short_term(
    messages: Sequence[Mapping[str, Any]], policy: MemoryPolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _normalise_messages(messages)
    kept_reversed: list[dict[str, Any]] = []
    used_chars = 0
    for message in reversed(source):
        content_chars = len(str(message.get("content", "")))
        over_count = len(kept_reversed) >= max(1, policy.short_term_message_limit)
        over_chars = bool(kept_reversed) and (
            used_chars + content_chars > max(1, policy.short_term_char_limit)
        )
        if over_count or over_chars:
            break
        kept_reversed.append(message)
        used_chars += content_chars

    kept = list(reversed(kept_reversed))
    return kept, source[: len(source) - len(kept)]


# 创建空中期记忆。
def _empty_mid_term() -> dict[str, Any]:
    return {"summary": [], "goals": [], "decisions": [], "archived_messages": 0}


# 追加去重后的限长条目。
def _append_unique(values: list[str], value: str, limit: int = 12) -> None:
    if value and value not in values:
        values.append(value)
    if len(values) > limit:
        del values[: len(values) - limit]


_GOAL_RE = re.compile(
    r"(?:当前目标(?:是|为)?|目标是|我要|需要完成|接下来(?:要|准备))\s*[:：]?\s*(.+)"
)
_DECISION_RE = re.compile(r"(?:决定|选择|改用|采用|不再使用)\s*[:：]?\s*(.+)")


# 清理目标后的决策描述。
def _clean_goal(value: str) -> str:
    return re.split(r"[，,；;]\s*(?:决定|选择|改用|采用|不再使用)", value, maxsplit=1)[
        0
    ]


# 归档淘汰消息到中期记忆。
def update_mid_term(
    current: Mapping[str, Any] | None,
    archived_messages: Sequence[Mapping[str, Any]],
    policy: MemoryPolicy,
) -> dict[str, Any]:
    state = _empty_mid_term()
    if current:
        for key in ("summary", "goals", "decisions"):
            state[key] = list(current.get(key, []) or [])
        state["archived_messages"] = int(current.get("archived_messages", 0) or 0)

    for message in archived_messages:
        content = _compact_text(message.get("content"), policy.message_preview_chars)
        if not content:
            continue
        role = str(message.get("role", ""))
        label = "用户" if role == "user" else "助手" if role == "assistant" else role
        _append_unique(state["summary"], f"{label}: {content}", limit=24)
        if role == "user":
            goal = _GOAL_RE.search(content)
            decision = _DECISION_RE.search(content)
            if goal:
                _append_unique(
                    state["goals"], _compact_text(_clean_goal(goal.group(1)), 200)
                )
            if decision:
                _append_unique(
                    state["decisions"], _compact_text(decision.group(1), 200)
                )
    state["archived_messages"] += len(archived_messages)

    while (
        state["summary"]
        and sum(len(x) for x in state["summary"]) > policy.mid_term_char_limit
    ):
        state["summary"].pop(0)
    return state


# 提取当前消息中的目标和决策。
def _update_mid_term_signals(
    state: dict[str, Any], messages: Sequence[Mapping[str, Any]]
) -> None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = _compact_text(message.get("content"), 300)
        goal = _GOAL_RE.search(content)
        decision = _DECISION_RE.search(content)
        if goal:
            _append_unique(
                state["goals"], _compact_text(_clean_goal(goal.group(1)), 200)
            )
        if decision:
            _append_unique(state["decisions"], _compact_text(decision.group(1), 200))


_LONG_TERM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explicit", re.compile(r"(?:请记住|记住)\s*[:：]?\s*(.+)")),
    ("preference", re.compile(r"我(?:更)?(?:喜欢|偏好|习惯于?)\s*[:：]?\s*(.+)")),
    ("policy", re.compile(r"以后(?:都|一律|默认)?\s*[:：]?\s*(.+)")),
    (
        "project_fact",
        re.compile(
            r"(?:(?:这个|本|我们的)项目|项目)(?:使用|采用|基于|后端是|前端是)\s*[:：]?\s*(.+)"
        ),
    ),
)


# 提取带强信号的长期事实。
def extract_long_term_facts(
    messages: Sequence[Mapping[str, Any]], policy: MemoryPolicy
) -> list[dict[str, Any]]:
    facts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = _compact_text(message.get("content"), policy.message_preview_chars)
        for fact_type, pattern in _LONG_TERM_PATTERNS:
            match = pattern.search(content)
            if not match:
                continue
            fact_content = _compact_text(match.group(1), policy.message_preview_chars)
            if len(fact_content) < 2:
                continue
            fact_id = hashlib.sha256(
                f"{fact_type}:{fact_content.casefold()}".encode("utf-8")
            ).hexdigest()[:24]
            facts.append(
                {
                    "id": fact_id,
                    "type": fact_type,
                    "content": fact_content,
                    "importance": 1.0 if fact_type == "explicit" else 0.8,
                }
            )
            break
    return facts


# 按重要性和新近性选择长期记忆。
def rank_long_term_facts(
    facts: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    indexed = enumerate(facts)
    ranked = sorted(
        indexed,
        key=lambda item: (
            float(item[1].get("importance", 0.0) or 0.0),
            float(item[1].get("updated_at", 0.0) or 0.0),
            item[0],
        ),
        reverse=True,
    )
    return [dict(fact) for _, fact in ranked[:limit]]


# 更新三层记忆。
def update_memory(
    short_term: Sequence[Mapping[str, Any]],
    mid_term: Mapping[str, Any] | None,
    new_memory_messages: Sequence[Mapping[str, Any]],
    display_messages: Sequence[Mapping[str, Any]],
    policy: MemoryPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    short, archived = compact_short_term([*short_term, *new_memory_messages], policy)
    mid = update_mid_term(mid_term, archived, policy)
    _update_mid_term_signals(mid, new_memory_messages)
    facts = extract_long_term_facts(display_messages, policy)
    return short, mid, facts


# 构造中期记忆上下文消息。
def _mid_term_message(mid_term: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not mid_term:
        return None
    sections = []
    goals = list(mid_term.get("goals", []) or [])
    decisions = list(mid_term.get("decisions", []) or [])
    summary = list(mid_term.get("summary", []) or [])
    if goals:
        sections.append("当前/历史目标：" + "；".join(goals[-4:]))
    if decisions:
        sections.append("已做决策：" + "；".join(decisions[-4:]))
    if summary:
        sections.append("较早经历：\n" + "\n".join(summary[-12:]))
    if not sections:
        return None
    return {
        "role": "memory",
        "content": "【中期记忆】\n" + "\n".join(sections),
    }


# 构造长期记忆上下文消息。
def _long_term_message(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not facts:
        return None
    lines = [f"- {_compact_text(f.get('content'), 300)}" for f in facts]
    return {
        "role": "memory",
        "content": "【长期记忆】\n稳定偏好/事实（仅在与当前问题相关时使用）：\n"
        + "\n".join(lines),
    }


# 合成分层记忆上下文。
def assemble_memory_context(
    short_term: Sequence[Mapping[str, Any]],
    mid_term: Mapping[str, Any] | None,
    long_term_facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    long_message = _long_term_message(long_term_facts)
    mid_message = _mid_term_message(mid_term)
    if long_message:
        context.append(long_message)
    if mid_message:
        context.append(mid_message)
    context.extend(dict(message) for message in short_term)
    return context


# 选择静态记忆并组装上下文。
def build_memory_context(
    short_term: Sequence[Mapping[str, Any]],
    mid_term: Mapping[str, Any] | None,
    long_term_facts: Sequence[Mapping[str, Any]],
    policy: MemoryPolicy,
) -> list[dict[str, Any]]:
    selected_facts = rank_long_term_facts(
        long_term_facts, policy.context_long_term_limit
    )
    memory_count = int(bool(selected_facts)) + int(bool(_mid_term_message(mid_term)))
    short_limit = max(0, policy.short_term_message_limit - memory_count)
    recent_messages = short_term[-short_limit:] if short_limit else []
    return assemble_memory_context(recent_messages, mid_term, selected_facts)
