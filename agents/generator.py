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
            "你是一个极其专业、严谨的本地知识库问答助手。你的任务是根据给定的参考资料回答用户的问题。\n\n"
            "【行为准则】:\n"
            "1. 必须完全基于下文中包裹在 <Document> 标签内的本地知识进行回答。\n"
            "2. 如果参考资料中的内容与用户的提问不相关、或不足以推导回答，请直接坦诚告知用户‘在参考资料中没有找到相关依据’，严禁胡编乱造。\n"
            "3. 当你引用了某段文档的内容时，请在句尾或段落末尾以 [文件名:页码] 的格式进行精确的物理溯源标注（例如: [架构设计手册.pdf:P12]）。"
        )

        user_content = (
            f"【参考资料开始】\n"
            f"{context_str}\n"
            f"【参考资料结束】\n\n"
            f"【用户提问】: {query}\n"
            f"请根据上述参考资料，给出严谨且带有引用标记的回答。"
        )

        return [
            SystemMessage(content = system_prompt),  # 系统消息
            HumanMessage(content = user_content)     # 用户消息
        ]
