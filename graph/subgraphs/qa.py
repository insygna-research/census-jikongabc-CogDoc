import copy
from functools import lru_cache
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from graph.state import GraphState, Evidence
from tools.retriever.vector_retriever import VectorRetriever
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.hybrid import HybridRetriever
from tools.reranker import BGEReranker
from agents.generator import Generator
from agents.query_rewriter import QueryRewriteAgent
from agents.citation_validator import CitationValidatorAgent

class RetrieverFactory:
    @staticmethod
    @lru_cache(maxsize = 32) # 缓存 Retriever 实例
    def get_engine(doc_id: str) -> HybridRetriever:
        return HybridRetriever(
            vector_retriever = VectorRetriever(collection_id = doc_id),
            bm25_retriever = BM25Retriever(collection_id = doc_id)
        )
    
def rewrite_node(state: GraphState) -> dict:
    return QueryRewriteAgent.rewrite_query(state)

def retrieve_node(state: GraphState) -> dict:
    original_query = state.get("query", "")
    doc_id = state.get("doc_id", "default") # 知识库ID

    # 原始 query 始终作为第一检索分支，保证不被改写轻微跑偏时漏掉强相关文档
    rewritten = state.get("rewritten_queries", [])
    queries = [original_query] if original_query else []
    for q in rewritten:
        if q not in queries:
            queries.append(q)

    engine = RetrieverFactory.get_engine(doc_id) # 获取检索器

    retrieved_docs = []
    seen_chunk_keys = set()

    for query in queries:
        docs = engine.search(query = query, top_k = 9)  # 混合检索
        for doc in docs:
            meta = doc["meta"]
            chunk_key = (meta["source"], meta["chunk_index"])
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)

                if query != original_query:
                    # 改写来源写入 retrieval 字段
                    doc_copy = copy.deepcopy(doc)
                    doc_copy.setdefault("retrieval", {})["rewrite_query"] = query
                else:
                    doc_copy = doc
                retrieved_docs.append(doc_copy)

    return {"retrieved_docs": retrieved_docs}   # 写入状态
    

def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    is_local = state.get("is_local", False)

    if is_local:
        BGEReranker.set_device("cpu")  # 本地模式下 Ollama 已占用 GPU，Reranker 退到 CPU

    reranked_docs = BGEReranker.rerank(query = query, docs = docs, top_n = 3)
    return {"reranked_docs": reranked_docs}

def generate_node(state: GraphState) -> dict:
    query = state.get("query", "")
    is_local = state.get("is_local", False) # 是否使用本地模型
    final_docs = state.get("reranked_docs", [])

    critique = state.get("critique", "") # 提取上一轮可能存在的校验失败批注
    iteration_count = state.get("iteration_count", 0)

    llm = Generator._get_client(is_local = is_local) # 获取模型客户端
    base_prompt = Generator.format_prompt(query = query, docs = final_docs) # 构造 Prompt

    messages_payload = list(base_prompt)
    if critique and iteration_count > 0:
        # 将引证校验失败的批注追加到系统消息末尾，保持 base_prompt 其余部分不变
        correction_note = (
            f"\n\n【引用校验失败通知】\n"
            f"你上一轮的回答已被引用校验器拦截，错误详情如下：\n\n"
            f"{critique}\n\n"
            f"请严格按照上述修正要求重新生成答案，确保每处引用的文件名和页码与 <Document> 标签属性完全吻合。"
        )
        if messages_payload and isinstance(messages_payload[0], SystemMessage):
            messages_payload[0] = SystemMessage(content = messages_payload[0].content + correction_note)
        else:
            messages_payload.insert(0, SystemMessage(content = correction_note))

    response_message = llm.invoke(messages_payload) # 调用模型

    evidence: list[Evidence] = [
        Evidence(
            chunk_index = doc["meta"]["chunk_index"],
            source = doc["meta"]["source"],
            page = doc["meta"]["page"],
            rerank_score = doc.get("retrieval", {}).get("rerank_score"),
            rewrite_query = doc.get("retrieval", {}).get("rewrite_query"),
            text_preview = doc["text"][:100],
        )
        for doc in final_docs
    ]

    return {
        "messages": [response_message],
        "answer": response_message.content,
        "sources": [doc["meta"] for doc in final_docs],
        "evidence": evidence,
    } # 写入消息状态

def citation_node(state: GraphState) -> dict:
    answer = state.get("answer", "")
    final_docs = state.get("reranked_docs", [])
    iteration_count = state.get("iteration_count", 0)
    max_iteration_count = state.get("max_iteration_count", 2)

    check_res = CitationValidatorAgent.validate_citations(answer, final_docs)

    return {
        "critique": check_res["critique"],
        "iteration_count": iteration_count + 1,
        "max_iteration_count": max_iteration_count
    }

def citation_check(state: GraphState) -> str:
    critique = state.get("critique", "")
    iteration_count = state.get("iteration_count", 0)
    max_iteration_count = state.get("max_iteration_count", 2) # 默认最大允许自愈重试 2 次

    if not critique:   # 没有错误批注，流向终点
        return END

    if iteration_count <= max_iteration_count:   # 有错，没达到反思重试上限，打回给生成节点
        return "generate_node"

    return END   # 达到重试上限仍未修好，放弃抢救，直接向下流出，交由前台做警告标记

sub_graph = StateGraph(GraphState) # 创建状态图

sub_graph.add_node("rewrite_node", rewrite_node)   # 注册问题重写节点
sub_graph.add_node("retrieve_node", retrieve_node) # 注册检索节点
sub_graph.add_node("rerank_node", rerank_node)     # 注册重排节点
sub_graph.add_node("generate_node", generate_node) # 注册生成节点
sub_graph.add_node("citation_node", citation_node) # 注册校验节点

sub_graph.add_edge(START, "rewrite_node")       # START -> 问题重写
sub_graph.add_edge("rewrite_node", "retrieve_node")   # 问题重写 -> 检索      
sub_graph.add_edge("retrieve_node", "rerank_node") # 检索 -> 重排
sub_graph.add_edge("rerank_node", "generate_node")    # 重排 -> 生成
sub_graph.add_edge("generate_node", "citation_node")  # 生成 -> 引证校验

sub_graph.add_conditional_edges(
    "citation_node",
    citation_check,
    {
        "generate_node": "generate_node", # 校验失败，打回重写
        END: END                          # 校验通过或无力回天，安全退出
    }
)

qa_subgraph_node = sub_graph.compile()
