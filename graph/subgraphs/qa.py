import copy
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from graph.state import GraphState, Evidence
from tools.retriever.vector_retriever import VectorRetriever
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.hybrid import HybridRetriever
from tools.reranker import BGEReranker
from agents.generator import Generator
from agents.query_rewriter import QueryRewriteAgent

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
            chunk_key = (meta["source"], meta["chunk_index"])  # 与 HybridRetriever 保持一致
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)

                # 不污染 meta.origin；改写来源写入 retrieval 字段（原始 query 路径不写该字段）
                doc_copy = copy.deepcopy(doc)
                if query != original_query:
                    doc_copy.setdefault("retrieval", {})["rewrite_query"] = query
                retrieved_docs.append(doc_copy)

    return {"retrieved_docs": retrieved_docs}   # 写入状态
    

def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", []) # 读取检索结果
    
    reranked_docs = BGEReranker.rerank(query = query, docs = docs, top_n = 3) # 重排序
    return {"reranked_docs": reranked_docs}

def generate_node(state: GraphState) -> dict:
    query = state.get("query", "")
    is_local = state.get("is_local", False) # 是否使用本地模型
    final_docs = state.get("reranked_docs", []) # 获取最终上下文

    llm = Generator._get_client(is_local = is_local) # 获取模型客户端
    prompt = Generator.format_prompt(query = query, docs = final_docs) # 构造 Prompt
    response_message = llm.invoke(prompt) # 调用模型

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

sub_graph = StateGraph(GraphState) # 创建状态图

sub_graph.add_node("rewrite_node", rewrite_node)   # 注册问题重写节点
sub_graph.add_node("retrieve_node", retrieve_node) # 注册检索节点
sub_graph.add_node("rerank_node", rerank_node)     # 注册重排节点
sub_graph.add_node("generate_node", generate_node) # 注册生成节点

sub_graph.add_edge(START, "rewrite_node")       # START -> 问题重写
sub_graph.add_edge("rewrite_node", "retrieve_node")   # 问题重写 -> 检索      
sub_graph.add_edge("retrieve_node", "rerank_node") # 检索 -> 重排
sub_graph.add_edge("rerank_node", "generate_node") # 重排 -> 生成
sub_graph.add_edge("generate_node", END)           # 生成 -> END

qa_subgraph_node = sub_graph.compile()