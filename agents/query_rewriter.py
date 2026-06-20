from typing import List
from pydantic import BaseModel, Field
from agents.generator import Generator

class QueryRewriteOutput(BaseModel):
    # 改写结果限定在少量高价值检索查询内。
    queries: List[str] = Field(
        min_length = 1,
        max_length = 3,
        description = "针对用户原始问题，裂变、改写出的 1-3 个最适合在本地知识库中检索的差异化查询语句。"
    )

class QueryRewriteAgent:
    @staticmethod
    def rewrite_query(state: dict) -> dict:
        # 改写失败时回退到原始 query，保证检索链路可继续。
        query = state.get("query", "")
        is_local = state.get("is_local", False)
        
        if not query:
            return {"rewritten_queries": []}

        llm = Generator._get_client(is_local = is_local)
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        system_prompt = (
            "你是一位 RAG 检索优化专家，负责将用户原始提问改写为更适合在向量数据库（Vector Search）和关键词引擎（BM25）中精确召回的检索语句。\n\n"
            "【任务定义】\n"
            "输出 1 至 3 条改写查询，每条聚焦原问题的某一具体技术侧面或命名实体，覆盖不同检索角度。\n\n"
            "【改写规则】\n"
            "1. 去除语气词和问句结构（如'请问'、'是什么'、'为什么'、'如何'），保留核心名词、动词和专有名词。\n"
            "2. 每条改写语句之间须有明显差异，不得重复或近义替换。\n"
            "3. 不得引入原问题中不存在的概念或实体。\n"
            "4. 若原问题已是简洁的关键词形式，可将其作为第一条直接输出，再补充 1-2 条不同角度的改写。\n\n"
            "【示例】\n"
            "原问题：大模型在医疗影像诊断中是怎么应用的？\n"
            "改写输出：\n"
            "  1. 大模型 医疗影像诊断 应用方法\n"
            "  2. 医学图像分析 深度学习 临床部署\n"
            "  3. 医疗AI 影像识别 模型推理"
        )

        try:
            output = structured_llm.invoke([
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"请对以下用户提问执行检索改写：\n{query}"
                }
            ])
            return {"rewritten_queries": output.queries}
        except Exception as e:
            return {"rewritten_queries": [query]}
