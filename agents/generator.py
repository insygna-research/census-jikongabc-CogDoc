import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import RetrievedDoc

class Generator:
    _clients = {}  # 缓存不同后端对应的 OpenAI 客户端实例

    CLOUD_MODEL = os.getenv("LLM_MODEL_NAME", "deepseek-chat")  # 云端模型名称
    CLOUD_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")  # 云端接口地址
    CLOUD_API_KEY = os.getenv("LLM_API_KEY", "your-cloud-api-key-here")  # 云端 API Key

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")  # 本地 Ollama 模型名称
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")  # Ollama OpenAI兼容接口地址
    OLLAMA_API_KEY = "ollama"

    @classmethod
    def _get_client(cls, is_local: bool = False, custom_model_name: str = None) -> ChatOpenAI:
        if is_local:
            base_url = cls.OLLAMA_BASE_URL
            api_key = cls.OLLAMA_API_KEY
            model_name = custom_model_name if custom_model_name else cls.OLLAMA_MODEL  # 优先使用传入模型
            client_key = f"local_{base_url}_{model_name}"  # 本地客户端缓存键
        else:
            base_url = cls.CLOUD_BASE_URL
            api_key = cls.CLOUD_API_KEY
            model_name = custom_model_name if custom_model_name else cls.CLOUD_MODEL  # 优先使用传入模型
            client_key = f"cloud_{base_url}_{model_name}"  # 云端客户端缓存键

        if client_key not in cls._clients:
            cls._clients[client_key] = ChatOpenAI(
                model = model_name,
                openai_api_key = api_key,
                openai_api_base = base_url,  # 接口地址
                temperature = 0.2,  # 回答随机性
                timeout = 90.0,     # 请求超时时间
                max_retries = 2     # 请求失败重试次数
            )
        return cls._clients[client_key]

    @classmethod
    def _build_context_string(cls, docs: List[RetrievedDoc]) -> str:
        if not docs:
            return "（未检索到任何相关的参考本地知识库内容。）"

        context_blocks = []  # 存储格式化后的文档块
        for doc in docs:
            meta = doc["meta"]
            source = meta.get("source", "未知文件")  # 文件名
            page = meta.get("page", 1)              # 页码
            chunk_idx = meta.get("chunk_index", 0)  # chunk编号

            block = (
                f'<Document source="{source}" page="{page}" chunk_id="{chunk_idx}">\n'
                f'{doc["text"].strip()}\n'  # 文本内容
                f'</Document>'
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)  # 拼接全部文档块

    @classmethod
    def format_prompt(cls, query: str, docs: List[RetrievedDoc]) -> List:
        context_str = cls._build_context_string(docs)

        system_prompt = (
            "你是一位严谨的学术文献问答专家，专职根据检索到的本地知识库文档回答用户提问。\n\n"
            "【任务定义】\n"
            "阅读下方 <Document> 标签中的参考文档，针对用户提问给出准确、有据可查的书面回答。\n\n"
            "【约束规则】\n"
            "1. 仅基于 <Document> 标签内的文本作答，禁止引入任何标签外的知识。\n"
            "2. 每陈述一处来自文档的事实，须在该句句尾附加引用标签（格式见下方）。\n"
            "3. 同一句话涉及多处来源时，可连续附加多个引用标签，例如：[a.pdf:P3][b.pdf:P7]。\n"
            "4. 回答语言须与用户提问保持一致。\n\n"
            "【引用格式】\n"
            "格式：[source属性值:P+page属性值]\n"
            "说明：source 和 page 的值直接取自对应 <Document> 标签的属性，不得使用任何占位词。\n"
            "示例：若标签为 <Document source=\"大模型开发应用赛.pdf\" page=\"5\">，\n"
            "      则该文档的引用写作：[大模型开发应用赛.pdf:P5]\n"
            "禁止：不能使用中文括号（ ）；不能写 [文件名:页码] 等占位形式；"
            "不能引用未出现在 <Document> 标签中的文件名或页码。\n\n"
            "【兜底规则】\n"
            "若参考文档中找不到与提问相关的内容，请直接回复：\n"
            "「在所提供的参考资料中未找到与该问题相关的内容，建议查阅更多资料。」\n"
            "不得凭空推断或捏造答案。"
        )

        user_content = (
            f"【参考资料开始】\n"
            f"{context_str}\n"
            f"【参考资料结束】\n\n"
            f"【用户提问】：{query}\n\n"
            f"请根据上述参考资料作答，每处事实须在句尾附加对应的引用标签。"
        )

        return [
            SystemMessage(content = system_prompt),  # 系统消息
            HumanMessage(content = user_content)     # 用户消息
        ]
