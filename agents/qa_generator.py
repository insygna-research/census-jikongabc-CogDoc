from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings
from graph.state import RetrievedDoc
from agents.answer_markers import NO_RELEVANT_CONTENT_ANSWER
from agents.conversation_memory import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    format_recent_chat_history,
)


class Generator:
    # 不同后端和模型使用独立客户端缓存。
    _clients = {}

    @classmethod
    def _get_client(
        cls, is_local: bool = False, custom_model_name: str = None
    ) -> ChatOpenAI:
        # client_key 必须包含后端、地址和模型名。
        settings = get_settings()
        if is_local:
            base_url = settings.ollama_base_url
            api_key = settings.ollama_api_key
            model_name = custom_model_name if custom_model_name else settings.ollama_model_name
            client_key = f"local_{base_url}_{model_name}"
        else:
            base_url = settings.llm_base_url
            api_key = settings.llm_api_key
            model_name = custom_model_name if custom_model_name else settings.llm_model_name
            client_key = f"cloud_{base_url}_{model_name}"
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY is not configured. Set it in your shell environment "
                    "or create a local .env file from .env.example."
                )

        if client_key not in cls._clients:
            cls._clients[client_key] = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.2,
                timeout=90.0,
                max_retries=2,
            )
        return cls._clients[client_key]

    @classmethod
    def _build_context_string(cls, docs: List[RetrievedDoc]) -> str:
        # chunk_id 只进入 Document 属性，引用格式仍按 source/page。
        if not docs:
            return "（未检索到任何相关的参考本地知识库内容。）"

        context_blocks = []
        for doc in docs:
            meta = doc["meta"]
            source = meta.get("source", "未知文件")
            page = meta.get("page", 1)
            chunk_id = meta.get("chunk_id", meta.get("chunk_index", 0))

            block = (
                f'<Document source="{source}" page="{page}" chunk_id="{chunk_id}">\n'
                f"{doc['text'].strip()}\n"
                f"</Document>"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    @classmethod
    def format_prompt(
        cls, query: str, docs: List[RetrievedDoc], chat_history: list | None = None
    ) -> List:
        # Prompt 明确约束模型只使用 Document 标签内的信息。
        context_str = cls._build_context_string(docs)
        history_text = format_recent_chat_history(
            chat_history, limit=CHAT_HISTORY_MESSAGE_LIMIT
        )

        system_prompt = (
            "你是一位严谨的学术文献问答专家，专职根据检索到的本地知识库文档回答用户提问。\n\n"
            "【任务定义】\n"
            "阅读下方 <Document> 标签中的参考文档，针对用户提问给出准确、有据可查的书面回答。\n\n"
            "【约束规则】\n"
            "1. 仅基于 <Document> 标签内的文本作答，禁止引入任何标签外的知识。\n"
            "2. 近期对话只用于理解当前提问里的指代或省略，不能作为事实来源。\n"
            "3. 每陈述一处来自文档的事实，须在该句句尾附加引用标签（格式见下方）。\n"
            "4. 同一句话涉及多处来源时，可连续附加多个引用标签，例如：[a.pdf:P3][b.pdf:P7]。\n"
            "5. 回答语言须与用户提问保持一致。\n\n"
            "【引用格式】\n"
            "格式：[source属性值:P+page属性值]\n"
            "说明：source 和 page 的值直接取自对应 <Document> 标签的属性，不得使用任何占位词。\n"
            '示例：若标签为 <Document source="大模型开发应用赛.pdf" page="5">，\n'
            "      则该文档的引用写作：[大模型开发应用赛.pdf:P5]\n"
            "禁止：不能使用中文括号（ ）；不能写 [文件名:页码] 等占位形式；"
            "不能引用未出现在 <Document> 标签中的文件名或页码。\n\n"
            "【兜底规则】\n"
            "若参考文档中找不到与提问相关的内容，请直接回复：\n"
            f"「{NO_RELEVANT_CONTENT_ANSWER}」\n"
            "不得凭空推断或捏造答案。"
        )

        user_content = (
            f"【近期对话】\n"
            f"{history_text or '（无）'}\n\n"
            f"【参考资料开始】\n"
            f"{context_str}\n"
            f"【参考资料结束】\n\n"
            f"【用户提问】：{query}\n\n"
            f"请根据上述参考资料作答，每处事实须在句尾附加对应的引用标签。"
        )

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
