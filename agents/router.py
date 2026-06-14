from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from agents.generator import Generator
from typing import Literal

class RouteDecision(BaseModel):
    task_type: Literal["qa", "summary", "compare", "unknown"] = Field(description = "任务类型: 'qa', 'summary', 'compare', 'unknown'")
    reason: str = Field(description = "做出该路由决策的清晰理由。")

class RouterAgent:
    @staticmethod
    def route_intent(state: dict, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])  # 读取消息历史
        configurable = config.get("configurable", {})  # 读取运行配置
        
        query = configurable.get("query", "")  # 获取用户问题
        doc_id = configurable.get("doc_id", "arch_blueprint_2026")
        is_local = configurable.get("is_local", False)

        if not query and messages:  # query为空时兜底
            for msg in reversed(messages):  # 从最新消息开始查找
                if msg.type == "user":  # 查找用户消息
                    query = msg.content  # 提取消息内容
                    break

        llm = Generator._get_client(is_local = is_local)
        structured_llm = llm.with_structured_output(RouteDecision)  # 开启结构化输出

        system_prompt = (
            "你是一位任务路由专家，负责将用户提问分配至对应的处理子图。\n"
            "知识库中存放了多种领域的文档，用户提问的范围可能很广。\n\n"
            "【分配规则】\n"
            "1. qa      — 默认选项。用户想了解某项知识、技能、准备方法、注意事项、技术细节，或就某一主题寻求信息，一律分配至 qa。\n"
            "2. summary — 用户明确要求'总结'、'摘要'、'概括'某篇或某段文档的全文内容。\n"
            "3. compare — 用户明确要求'对比'、'比较'两个或多个方案、论文、技术之间的异同。\n"
            "4. unknown — 仅当提问是纯闲聊（如'你好'、'今天天气怎么样'）或完全无法理解的乱码输入时才使用。\n\n"
            "【决策要点】\n"
            "- 只要提问中含有实质性的信息诉求，无论措辞是'需要什么'、'如何准备'、'怎么学'、'是什么'，均选 qa。\n"
            "- 不确定时，选 qa，让检索系统判断能否找到相关内容。\n"
            "- reason 字段用一句话简要说明分配依据，不超过 30 字。"
        )

        user_content = f"请分析以下用户提问的路由意图：\n{query}"
        
        try:
            decision = structured_llm.invoke([
                {
                    "role": "system",
                    "content": system_prompt
                },  # 系统消息
                {
                    "role": "user",
                    "content": user_content
                }  # 用户消息
            ])  # 调用模型进行路由判断

            return {
                "query": query,
                "doc_id": doc_id,
                "is_local": is_local,
                "task_type": decision.task_type,
                "router_reason": decision.reason
            }  # 返回路由结果

        except Exception as e:
            return {
                "query": query,
                "doc_id": doc_id,
                "is_local": is_local,
                "task_type": "qa",
                "router_reason": f"结构化路由解析异常，安全 fallback 到标准 QA 子图。错误详情: {str(e)}"
            }  # 异常时默认走QA