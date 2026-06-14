from typing import List
from pydantic import BaseModel, Field
from agents.generator import Generator

class QueryRewriteOutput(BaseModel):
    queries: List[str] = Field(
        min_items = 1,
        max_items = 3,
        description = "针对用户原始问题，裂变、改写出的 1-3 个最适合在本地知识库中检索的差异化查询语句。"
    )

class QueryRewriteAgent:
    @staticmethod
    def rewrite_query(state: dict) -> dict:
        query = state.get("query", "")
        is_local = state.get("is_local", False)
        
        if not query:
            return {"rewritten_queries": []}

        llm = Generator._get_client(is_local = is_local)
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        system_prompt = (
            "你是一个高级分布式 RAG 系统中的查询重写专家。\n"
            "你的任务是分析用户的提问，将其拆解并改写为 1 到 3 个最适合在本地知识库（Vector + BM25）中检索的差异化查询串。\n\n"
            "【改写准则】:\n"
            "1. 每一个改写句应当聚焦于用户问题中的某一个具体技术侧面、名词或潜在实体。\n"
            "2. 强力去除非检索词（如‘请问’、‘是什么’、‘为什么’），将其转化为高密度的术语关键词组合。\n"
            "3. 保持与原文核心意图的一致性，绝对不要凭空捏造与用户提问无关的概念。"
        )

        try:
            output = structured_llm.invoke([
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"请针对该用户提问执行检索改写 >>> \n{query}"
                }
            ])
            return {"rewritten_queries": output.queries}
        except Exception as e:
            return {"rewritten_queries": [query]}