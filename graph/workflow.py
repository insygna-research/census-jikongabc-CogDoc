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
    @staticmethod # 静态方法
    @lru_cache(maxsize = 32) # 缓存 Retriever 实例
    def get_engine(doc_id: str) -> HybridRetriever:
        return HybridRetriever(
            vector_retriever = VectorRetriever(collection_id = doc_id),
            bm25_retriever = BM25Retriever(collection_id = doc_id)
        )

def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {}) # 读取运行配置
    query = configurable.get("query", "") # 用户问题
    doc_id = configurable.get("doc_id", "default") # 知识库ID

    engine = RetrieverFactory.get_engine(doc_id) # 获取检索器
    retrieved_docs = engine.search(query = query, top_k = 9) # 混合检索
    
    return {"docs": retrieved_docs} # 写入状态

def rerank_node(state: GraphState, config: RunnableConfig) -> dict:

    configurable = config.get("configurable", {})
    query = configurable.get("query", "")
    current_docs = state.get("docs", []) # 读取检索结果
    
    reranked_docs = BGEReranker.rerank(query = query, docs = current_docs, top_n = 3) # 重排序
    return {"docs": reranked_docs} # 覆盖 docs

def generate_node(state: GraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    query = configurable.get("query", "")
    is_local = configurable.get("is_local", False) # 是否使用本地模型
    docs = state.get("docs", []) # 获取最终上下文

    llm = Generator._get_client(is_local = is_local) # 获取模型客户端

    prompt = Generator.format_prompt(query = query, docs = docs) # 构造 Prompt

    response_message = llm.invoke(prompt) # 调用模型

    return {"messages": [response_message]} # 写入消息状态
    

workflow = StateGraph(GraphState) # 创建状态图

workflow.add_node("retrieve_node", retrieve_node) # 注册检索节点
workflow.add_node("rerank_node", rerank_node)     # 注册重排节点
workflow.add_node("generate_node", generate_node) # 注册生成节点

workflow.add_edge(START, "retrieve_node")         # START -> 检索
workflow.add_edge("retrieve_node", "rerank_node") # 检索 -> 重排
workflow.add_edge("rerank_node", "generate_node") # 重排 -> 生成
workflow.add_edge("generate_node", END)           # 生成 -> END

app = workflow.compile() # 编译工作流
