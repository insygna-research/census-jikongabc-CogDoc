from typing import List, Optional, Sequence, Mapping, Any
from pydantic import BaseModel, Field
from cogdoc.agents.conversation_memory import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    format_recent_chat_history,
)
from cogdoc.agents.qa_generator import Generator
from cogdoc.agents.structured_output import invoke_structured


class SourceResolution(BaseModel):
    # 允许为空：无法确定指代时返回空数组，由调用方回落原早退。
    sources: List[str] = Field(
        default_factory=list,
        description="用户实际指向的文件名，逐字取自可用文件清单；无法确定时为空数组。",
    )


_COMPARE_SYSTEM = (
    "你是文件指代消解助手。用户想对比若干文件，但当前提问里可能用了“这个文件”、"
    "“上面那篇”、“刚才那个”等指代或省略表达。请结合【近期对话】，判断用户实际想对比"
    "【可用文件清单】中的哪些文件。\n\n"
    "【硬性约束】\n"
    "1. 只能从【可用文件清单】里选择，输出的文件名必须与清单逐字一致（含扩展名）。\n"
    "2. 禁止编造清单之外的文件名，禁止臆测。\n"
    "3. 若无法确定用户具体指向哪些文件，输出空数组。\n\n"
    "【输出格式】\n"
    '只输出 JSON 对象，不要 Markdown，不要解释。示例：{"sources":["a.pdf","b.pdf"]}'
)

_SUMMARY_SYSTEM = (
    "你是文件指代消解助手。用户想总结某个文件，但当前提问里可能用了“这个文件”、"
    "“上面那篇”、“刚才那个”等指代或省略表达。请结合【近期对话】，判断用户实际想总结"
    "【可用文件清单】中的哪一个文件。\n\n"
    "【硬性约束】\n"
    "1. 只能从【可用文件清单】里选择，输出的文件名必须与清单逐字一致（含扩展名）。\n"
    "2. 禁止编造清单之外的文件名，禁止臆测。\n"
    "3. 若无法确定用户具体指向哪个文件，输出空数组。\n\n"
    "【输出格式】\n"
    '只输出 JSON 对象，不要 Markdown，不要解释。示例：{"sources":["a.pdf"]}'
)


def _filter_known_sources(names: Sequence[str], sources: Sequence[str]) -> List[str]:
    # 闭集校验 + 去重，只保留清单中真实存在的文件名，并按 sources 下标排序保证列序确定。
    order = {source: index for index, source in enumerate(sources)}
    seen = set()
    resolved: List[str] = []
    for name in names:
        name = (name or "").strip()
        if name in order and name not in seen:
            seen.add(name)
            resolved.append(name)
    resolved.sort(key=lambda name: order[name])
    return resolved


def _resolve_sources(
    query: str,
    sources: Sequence[str],
    chat_history: Sequence[Mapping[str, Any]] | None,
    is_local: bool,
    system_prompt: str,
) -> List[str]:
    # 用近期对话消解“这个文件/上面那篇”等多轮指代；任何异常回落空，不打断管道。
    history_text = format_recent_chat_history(
        chat_history, limit=CHAT_HISTORY_MESSAGE_LIMIT
    )
    if not history_text:
        return []

    source_list = "\n".join(f"- {source}" for source in sources)
    try:
        llm = Generator._get_client(is_local=is_local)
        output = invoke_structured(
            llm,
            SourceResolution,
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"【可用文件清单】\n{source_list}\n\n"
                        f"【近期对话】\n{history_text}\n\n"
                        f"【当前请求】\n{query}\n\n"
                        "请判断用户指向清单中的哪些文件。"
                    ),
                },
            ],
        )
    except Exception:
        return []

    return _filter_known_sources(output.sources, sources)


def resolve_compare_sources(
    query: str,
    sources: Sequence[str],
    chat_history: Sequence[Mapping[str, Any]] | None,
    is_local: bool = False,
) -> List[str]:
    # compare 需要 ≥2 篇；不足则回落空，交由调用方早退。
    if not query or len(sources) < 2:
        return []
    resolved = _resolve_sources(query, sources, chat_history, is_local, _COMPARE_SYSTEM)
    return resolved if len(resolved) >= 2 else []


def resolve_summary_source(
    query: str,
    sources: Sequence[str],
    chat_history: Sequence[Mapping[str, Any]] | None,
    is_local: bool = False,
) -> Optional[str]:
    # summary 只需 1 篇；取消解结果中的第一篇，无法确定返回 None。
    if not query or not sources:
        return None
    resolved = _resolve_sources(query, sources, chat_history, is_local, _SUMMARY_SYSTEM)
    return resolved[0] if resolved else None
