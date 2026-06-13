from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from agents.generator import Generator
from typing import Literal

class RouteDecision(BaseModel):
    task_type: Literal["qa", "summary", "compare", "unknown"] = Field(description="任务类型: 'qa', 'summary', 'compare', 'unknown'")
    reason: str = Field(description="做出该路由决策的清晰理由。")

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
            "你是一个高级任务路由专家。你的职责是深度分析用户针对本地论文知识库输入的提问意图，并将其精准分发到对应的处理子图中。\n\n"
            "分配规则：\n"
            "1. 如果用户是针对具体事实、概念、某项技术细节、实验数据进行提问或寻根问底，分发至 'qa'。\n"
            "2. 如果用户要求总结整篇论文、抽取全篇核心大意、概述章节结构或摘要，分发至 'summary'。\n"
            "3. 如果用户明确提到了多篇文档/论文、要求做技术方案异同点对比、跨论文对比，分发至 'compare'。\n"
            "请严格按照规则输出结构化 JSON。"
        )  # Router系统提示词

        user_content = f"请评估以下用户输入的路由意图 >>> \n{query}"  # 构造用户提示词
        
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