import os
from typing import List, Iterator
from openai import OpenAI
from graph.state import RetrievedDoc

class Generator:
    _clients = {}  # 缓存不同后端对应的 OpenAI 客户端实例

    CLOUD_MODEL = os.getenv("LLM_MODEL_NAME", "deepseek-chat")  # 云端模型名称
    CLOUD_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")  # 云端接口地址
    CLOUD_API_KEY = os.getenv("LLM_API_KEY", "your-cloud-api-key-here")  # 云端 API Key

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")  # 本地 Ollama 模型名称
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")  # Ollama OpenAI兼容接口地址
    OLLAMA_API_KEY = "ollama"  # Ollama 占位 API Key

    @classmethod
    def _get_client_and_model(cls, is_local: bool = False, custom_model_name: str = None) -> tuple[OpenAI, str]:
        if is_local:
            base_url = cls.OLLAMA_BASE_URL
            api_key = cls.OLLAMA_API_KEY
            model_name = custom_model_name if custom_model_name else cls.OLLAMA_MODEL  # 优先使用传入模型
            client_key = f"local_{base_url}"  # 本地客户端缓存键
        else:
            base_url = cls.CLOUD_BASE_URL
            api_key = cls.CLOUD_API_KEY
            model_name = custom_model_name if custom_model_name else cls.CLOUD_MODEL  # 优先使用传入模型
            client_key = f"cloud_{base_url}"  # 云端客户端缓存键

        if client_key not in cls._clients:
            cls._clients[client_key] = OpenAI(
                api_key = api_key,  # 接口认证信息
                base_url = base_url,  # 接口地址
                timeout = 90.0  # 请求超时时间
            )
        return cls._clients[client_key], model_name  # 返回客户端和最终模型名

    @classmethod
    def _build_context_string(cls, docs: List[RetrievedDoc]) -> str:
        if not docs:
            return "（未检索到任何相关的参考本地知识库内容。）"

        context_blocks = []  # 存储格式化后的文档块
        for doc in docs:
            meta = doc["meta"]
            source = meta.get("source", "未知文件")  # 文件名
            page = meta.get("page", 1)  # 页码
            chunk_idx = meta.get("chunk_index", 0)  # chunk编号

            block = (
                f'<Document source="{source}" page="{page}" chunk_id="{chunk_idx}">\n'
                f'{doc["text"].strip()}\n'  # 文本内容
                f'</Document>'
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)  # 拼接全部文档块


    @classmethod
    def generate_stream(cls, query: str, docs: List[RetrievedDoc], is_local: bool = False, model_name: str = None, temperature: float = 0.2) -> Iterator[str]:
        client, model = cls._get_client_and_model(is_local = is_local, custom_model = model_name)  # 获取客户端和模型
        context_str = cls._build_context_string(docs)  # 构造知识库上下文

        system_prompt = (
            "你是一个极其专业、严谨的本地知识库问答助手。你的任务是根据给定的参考资料回答用户的问题。\n\n"
            "【行为准则】:\n"
            "1. 必须完全基于下文中包裹在 <Document> 标签内的本地知识进行回答。\n"
            "2. 如果参考资料中的内容与用户的提问不相关、或不足以推导回答，请直接坦诚告知用户‘在参考资料中没有找到相关依据’，严禁胡编乱造。\n"
            "3. 当你引用了某段文档的内容时，请在句尾或段落末尾以 [文件名:页码] 的格式进行精确的物理溯源标注（例如: [架构设计手册.pdf:P12]）。"
        )  # 系统提示词

        user_content = (
            f"【参考资料开始】\n"
            f"{context_str}\n"  # 检索出的参考资料
            f"【参考资料结束】\n\n"
            f"【用户提问】: {query}\n"  # 用户问题
            f"请根据上述参考资料，给出严谨且带有引用标记的回答。"
        )

        messages = [
            {"role": "system", "content": system_prompt},  # 系统消息
            {"role": "user", "content": user_content}  # 用户消息
        ]

        try:
            response = client.chat.completions.create(
                model = model,  # 模型名称
                messages = messages,  # 对话消息
                temperature = temperature,  # 采样温度
                stream = True  # 开启流式输出
            )

            for chunk in response:  # 遍历流式响应
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content  # 逐Token返回内容
                    
        except Exception as e:
            yield f"\n[Generator 运行时异常]: 与模型后端通信中断。当前模式: {'本地Ollama' if is_local else '云端API'}，目标模型: {target_model}。错误详情: {str(e)}"  # 返回异常信息