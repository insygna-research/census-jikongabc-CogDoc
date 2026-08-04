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


QA_SYSTEM_PROMPT_TEMPLATE = (
    "你是一位严谨的学术文献问答专家，专职根据检索到的本地知识库文档回答用户提问。\n\n"
    "【任务定义】\n阅读下方 <Document> 标签中的参考文档，针对用户提问给出准确、有据可查的书面回答。\n\n"
    "【约束规则】\n1. 仅基于 <Document> 标签内的文本作答，禁止引入任何标签外的知识。\n"
    "2. 近期对话只用于理解当前提问里的指代或省略，不能作为事实来源。\n"
    "3. 每陈述一处来自文档的事实，须在该句句尾附加引用标签（格式见下方）。\n"
    "4. 同一句话涉及多处来源时，可连续附加多个引用标签，例如：[a.pdf:P3][b.pdf:P7]。\n"
    "5. 回答语言须与用户提问保持一致。\n\n【引用格式】\n"
    "原始文档格式：[source属性值:P+page属性值]\n派生知识格式：[knowledge:knowledge_id属性值]\n"
    "说明：原始文档引用值直接取自对应 <Document> 标签属性；派生知识引用值直接取自 <Knowledge> 标签属性。\n"
    '示例：若标签为 <Document source="大模型开发应用赛.pdf" page="5">，\n'
    "      则该文档的引用写作：[大模型开发应用赛.pdf:P5]\n"
    '示例：若标签为 <Knowledge knowledge_id="K123">，则该知识的引用写作：[knowledge:K123]\n'
    "禁止：不能使用中文括号（ ）；不能写 [文件名:页码] 等占位形式；不能引用未出现在参考资料标签中的文件名、页码或知识标识。\n\n"
    "【兜底规则】\n若参考文档中找不到与提问相关的内容，请直接回复：\n"
    "「{no_relevant_content_answer}」\n不得凭空推断或捏造答案。"
)
QA_USER_PROMPT_TEMPLATE = (
    "【近期对话】\n{history_text}\n\n【参考资料开始】\n{context}\n"
    "【参考资料结束】\n\n【用户提问】：{query}\n\n"
    "请根据上述参考资料作答，每处事实须在句尾附加对应的引用标签。"
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

    # 获取节点客户端；未配置覆盖时保持旧调用路径和测试替身兼容性。
    @classmethod
    def _get_client_for_node(cls, node_name: str, *, is_local: bool = False):
        settings = get_settings()
        is_local = settings.is_local_for_node(node_name, request_is_local=is_local)
        model_name = settings.model_name_for_node(node_name, is_local=is_local)
        default_model = (
            settings.ollama_model_name if is_local else settings.llm_model_name
        )
        if model_name == default_model:
            return cls._get_client(is_local=is_local)
        return cls._get_client(is_local=is_local, custom_model_name=model_name)

    # 构建上下文文本。
    @classmethod
    def _build_context_string(cls, docs: List[RetrievedDoc]) -> str:
        # 共享 renderer 也是 Evidence Pack 的精确字符计数口径。
        return render_evidence_context(docs)

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
