from functools import lru_cache
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from graph.state import GraphState, RetrievedDoc
from tools.retriever.vector_retriever import VectorRetriever
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.hybrid import HybridRetriever
from tools.reranker import BGEReranker
from agents.generator import Generator

class RetrieverFactory:
    @staticmethod
    @lru_cache(maxsize = 32) # 缓存 Retriever 实例
    def get_engine(doc_id: str) -> HybridRetriever:
        return HybridRetriever(
            vector_retriever = VectorRetriever(collection_id = doc_id),
            bm25_retriever = BM25Retriever(collection_id = doc_id)
        )
    
def retrieve_node(state: GraphState) -> dict:
    query = state.get("query", "") # 用户问题
    doc_id = state.get("doc_id", "default") # 知识库ID

    engine = RetrieverFactory.get_engine(doc_id) # 获取检索器
    retrieved_docs = engine.search(query = query, top_k = 9) # 混合检索
    
    return {"retrieved_docs": retrieved_docs} # 写入状态

def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    current_docs = state.get("retrieved_docs", []) # 读取检索结果
    
    reranked_docs = BGEReranker.rerank(query = query, docs = current_docs, top_n = 3) # 重排序
    return {"reranked_docs": reranked_docs} # 覆盖 docs

def generate_node(state: GraphState) -> dict:
    query = state.get("query", "")
    is_local = state.get("is_local", False) # 是否使用本地模型
    final_docs = state.get("reranked_docs", []) # 获取最终上下文

    llm = Generator._get_client(is_local = is_local) # 获取模型客户端

    prompt = Generator.format_prompt(query = query, docs = final_docs) # 构造 Prompt

    response_message = llm.invoke(prompt) # 调用模型

    return {
        "messages": [response_message],
        "answer": response_message.content
    } # 写入消息状态

sub_graph = StateGraph(GraphState) # 创建状态图

sub_graph.add_node("retrieve_node", retrieve_node) # 注册检索节点
sub_graph.add_node("rerank_node", rerank_node)     # 注册重排节点
sub_graph.add_node("generate_node", generate_node) # 注册生成节点

sub_graph.add_edge(START, "retrieve_node")         # START -> 检索
sub_graph.add_edge("retrieve_node", "rerank_node") # 检索 -> 重排
sub_graph.add_edge("rerank_node", "generate_node") # 重排 -> 生成
sub_graph.add_edge("generate_node", END)           # 生成 -> END

qa_subgraph_node = sub_graph.compile()