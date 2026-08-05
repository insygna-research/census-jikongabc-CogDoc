from typing import Any, List, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from cogdoc.config.settings import get_settings
from cogdoc.graph.state import RetrievedDoc
from cogdoc.agents.answer_markers import NO_RELEVANT_CONTENT_ANSWER
from cogdoc.agents.conversation_memory import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    format_recent_chat_history,
)
from cogdoc.tools.evidence_rendering import render_evidence_context
from cogdoc.tools.citation_ledger import ensure_evidence_ids


QA_SYSTEM_PROMPT_TEMPLATE = (
    "你是一位严谨的学术文献问答专家，专职根据检索到的本地知识库文档回答用户提问。\n\n"
    "【任务定义】\n阅读下方 <Document> 标签中的参考文档，针对用户提问给出准确、有据可查的书面回答。\n\n"
    "【约束规则】\n1. 仅基于 <Document> 标签内的文本作答，禁止引入任何标签外的知识。\n"
    "2. 近期对话只用于理解当前提问里的指代或省略，不能作为事实来源。\n"
    "3. 每陈述一处来自文档的事实，须在该句句尾附加对应 Evidence ID（格式见下方）。\n"
    "4. 同一句话涉及多处证据时，可连续附加多个 Evidence ID，例如：[E001][E003]。\n"
    "5. 回答语言须与用户提问保持一致。\n\n【引用格式】\n"
    '统一格式：[evidence_id属性值]，例如标签的 evidence_id="E001" 时写作 [E001]。\n'
    "Evidence ID 必须逐字取自支持该事实的 <Document> 或 <Knowledge> 标签。\n"
    "禁止：不得输出文件页码引用、knowledge 引用、小写/全角/带空格的 ID，"
    "也不得引用未出现在参考资料标签中的 Evidence ID。\n\n"
    "【兜底规则】\n若参考文档中找不到与提问相关的内容，请直接回复：\n"
    "「{no_relevant_content_answer}」\n不得凭空推断或捏造答案。"
)
QA_USER_PROMPT_TEMPLATE = (
    "【近期对话】\n{history_text}\n\n【参考资料开始】\n{context}\n"
    "【参考资料结束】\n\n【用户提问】：{query}\n\n"
    "请根据上述参考资料作答，每处事实须在句尾附加对应的 Evidence ID。"
)


# 不同后端和模型使用独立客户端缓存。
class Generator:
    _clients: dict[str, Any] = {}

    # 清理客户端。
    @classmethod
    def clear_clients(cls) -> None:
        # 改了模型配置后清缓存，避免继续使用旧客户端。
        cls._clients.clear()

    # 获取客户端。
    @classmethod
    def _get_client(
        cls,
        is_local: bool = False,
        custom_model_name: str | None = None,
    ) -> ChatOpenAI:
        # 客户端键必须包含后端、地址和模型名。
        settings = get_settings()
        if is_local:
            base_url = settings.ollama_base_url
            api_key = settings.ollama_api_key
            model_name = custom_model_name or settings.ollama_model_name
            client_key = f"local_{base_url}_{model_name}"
            # 本地超时更长、重试更少。
            timeout = settings.ollama_timeout_seconds
            max_retries = settings.ollama_max_retries
        else:
            base_url = settings.llm_base_url
            api_key = settings.llm_api_key
            model_name = custom_model_name or settings.llm_model_name
            client_key = f"cloud_{base_url}_{model_name}"
            # 云端超时与重试按后端服务级别可配。
            timeout = settings.llm_timeout_seconds
            max_retries = settings.llm_max_retries
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY is not configured. Set it in your shell environment "
                    "or create a local .env file from .env.example."
                )

        if client_key not in cls._clients:
            cls._clients[client_key] = ChatOpenAI(
                model=model_name,
                api_key=cast(Any, api_key),
                base_url=base_url,
                temperature=0.2,
                timeout=timeout,
                max_retries=max_retries,
            )
        return cls._clients[client_key]

    # 获取节点客户端；这是其他 agent 复用模型路由的公开入口。
    @classmethod
    def get_client_for_node(
        cls, node_name: str, *, is_local: bool = False
    ) -> ChatOpenAI:
        settings = get_settings()
        is_local = settings.is_local_for_node(node_name, request_is_local=is_local)
        model_name = settings.model_name_for_node(node_name, is_local=is_local)
        default_model = (
            settings.ollama_model_name if is_local else settings.llm_model_name
        )
        if model_name == default_model:
            return cls._get_client(is_local=is_local)
        return cls._get_client(is_local=is_local, custom_model_name=model_name)

    # 保留旧入口，避免破坏现有节点和测试替身。
    @classmethod
    def _get_client_for_node(
        cls, node_name: str, *, is_local: bool = False
    ) -> ChatOpenAI:
        return cls.get_client_for_node(node_name, is_local=is_local)

    # 构建上下文文本。
    @classmethod
    def _build_context_string(cls, docs: List[RetrievedDoc]) -> str:
        # 共享 renderer 也是 Evidence Pack 的精确字符计数口径。
        prompt_docs, _ = ensure_evidence_ids(docs)
        return render_evidence_context(prompt_docs)

    # 格式化提示。
    @classmethod
    def format_prompt(
        cls, query: str, docs: List[RetrievedDoc], chat_history: list | None = None
    ) -> List:
        # 提示明确约束模型只使用参考标签内的信息。
        context_str = cls._build_context_string(docs)
        history_text = format_recent_chat_history(
            chat_history, limit=CHAT_HISTORY_MESSAGE_LIMIT
        )

        system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(
            no_relevant_content_answer=NO_RELEVANT_CONTENT_ANSWER
        )

        user_content = QA_USER_PROMPT_TEMPLATE.format(
            history_text=history_text or "（无）",
            context=context_str,
            query=query,
        )

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
