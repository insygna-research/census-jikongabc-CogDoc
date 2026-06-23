import re
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from agents.conversation_memory import format_recent_chat_history
from agents.qa_generator import Generator
from agents.structured_output import invoke_structured
from observability.logger import log_event
from typing import Literal


FORCED_TASK_TYPES = ("qa", "summary", "compare")
FORCED_TASK_TYPE_SET = frozenset(FORCED_TASK_TYPES)


class RouteDecision(BaseModel):
    task_type: Literal["qa", "summary", "compare", "unknown"] = Field(
        description="任务类型: 'qa', 'summary', 'compare', 'unknown'"
    )
    reason: str = Field(description="做出该路由决策的清晰理由。")


def classify_intent_by_rule(query: str) -> RouteDecision:
    # LLM 结构化输出不可用时的确定性兜底，不能把明确摘要/对比请求打到 QA。
    normalized = (query or "").strip().lower()
    if not normalized:
        return RouteDecision(task_type="unknown", reason="空问题")

    compare_patterns = [
        r"(对比|比较).+(和|与|跟|及|以及|vs|versus).+",
        r".+(和|与|跟|及|以及|vs|versus).+(对比|比较|区别|差异|异同)",
        r"(区别|差异|异同).+(是什么|有哪些|在哪|如何|怎么)",
        r"compare .+ (and|with|vs|versus) .+",
    ]
    if any(re.search(pattern, normalized) for pattern in compare_patterns):
        return RouteDecision(task_type="compare", reason="命中对比关键词")

    summary_patterns = [
        r"^(请|帮我|给我|麻烦)?(总结|摘要|概括)(一下|下)?[：:\s]?.+",
        r"^.+(全文|整篇|这篇|该文档|这个文档).*(总结|摘要|概括)$",
        r"^(summarize|summary)\b.+",
    ]
    if any(re.search(pattern, normalized) for pattern in summary_patterns):
        return RouteDecision(task_type="summary", reason="命中摘要关键词")

    return RouteDecision(task_type="qa", reason="规则兜底为问答")


class RouterAgent:
    @staticmethod
    def route_intent(state: dict, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        configurable = config.get("configurable", {})

        query = configurable.get("query", "")
        doc_id = configurable.get("doc_id", "arch_blueprint_2026")
        request_id = configurable.get("request_id", state.get("request_id", ""))
        trace_id = configurable.get("trace_id", state.get("trace_id", ""))
        is_local = configurable.get("is_local", False)
        forced_task = configurable.get("forced_task")

        if not query and messages:
            for msg in reversed(messages):
                if msg.type == "user":
                    query = msg.content
                    break

        if forced_task in FORCED_TASK_TYPE_SET:
            result = {
                "query": query,
                "doc_id": doc_id,
                "request_id": request_id,
                "trace_id": trace_id,
                "is_local": is_local,
                "task_type": forced_task,
                "router_reason": "用户显式指定",
            }
            log_event(
                "router",
                "router_decision",
                result,
                task_type=forced_task,
                forced=True,
            )
            return result

        system_prompt = (
            "你是一位任务路由专家，负责将用户提问分配至对应的处理子图。\n"
            "知识库中存放了多种领域的文档，用户提问的范围可能很广。\n\n"
            "【分配规则】\n"
            "1. qa      — 默认选项。用户想了解某项知识、技能、准备方法、注意事项、技术细节，或就某一主题寻求信息，一律分配至 qa。\n"
            "2. summary — 用户明确要求'总结'、'摘要'、'概括'某篇或某段文档的全文内容。\n"
            "3. compare — 用户明确要求'对比'、'比较'两个或多个方案、论文、技术之间的异同。\n"
            "4. unknown — 极窄兜底。仅当提问是纯闲聊（如'你好'、'今天天气怎么样'）、与本地知识库问答无关的闲谈，或完全无法理解的乱码输入时才使用。\n\n"
            "【决策要点】\n"
            "- 只要提问中含有实质性的信息诉求，无论措辞是'需要什么'、'如何准备'、'怎么学'、'是什么'，均选 qa。\n"
            "- 若用户用'那总结一下呢'、'再对比一下'这类省略追问，结合近期对话判断 summary 或 compare。\n"
            "- 不确定时，选 qa，让检索系统判断能否找到相关内容。\n"
            "- reason 字段用一句话简要说明分配依据，不超过 30 字。\n\n"
            "【输出格式】\n"
            "只输出 JSON 对象，不要 Markdown，不要解释。JSON 示例：\n"
            '{"task_type":"qa","reason":"用户询问信息"}'
        )

        history_text = format_recent_chat_history(state.get("chat_history"))
        if history_text:
            user_content = (
                "请结合近期对话分析以下用户提问的路由意图。\n\n"
                f"【近期对话】\n{history_text}\n\n"
                f"【当前提问】\n{query}"
            )
        else:
            user_content = f"请分析以下用户提问的路由意图：\n{query}"

        try:
            llm = Generator._get_client(is_local=is_local)
            decision = invoke_structured(
                llm,
                RouteDecision,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )

            result = {
                "query": query,
                "doc_id": doc_id,
                "request_id": request_id,
                "trace_id": trace_id,
                "is_local": is_local,
                "task_type": decision.task_type,
                "router_reason": decision.reason,
            }
            log_event(
                "router",
                "router_decision",
                result,
                task_type=decision.task_type,
                forced=False,
                query_length=len(query),
            )
            return result

        except Exception as e:
            fallback = classify_intent_by_rule(query)
            result = {
                "query": query,
                "doc_id": doc_id,
                "request_id": request_id,
                "trace_id": trace_id,
                "is_local": is_local,
                "task_type": fallback.task_type,
                "router_reason": (
                    f"结构化路由解析异常，已按规则 fallback 到 {fallback.task_type}。"
                    f"规则原因：{fallback.reason}。错误详情: {str(e)}"
                ),
            }
            log_event(
                "router",
                "router_fallback",
                result,
                task_type=fallback.task_type,
                error_class=type(e).__name__,
                query_length=len(query),
            )
            return result
