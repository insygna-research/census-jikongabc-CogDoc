import copy
from collections import OrderedDict
from threading import RLock
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from config.settings import get_settings
from graph.state import GraphState, Evidence
from observability.logger import log_event
from tools.retriever.vector_retriever import VectorRetriever
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.hybrid import HybridRetriever
from tools.reranker import BGEReranker
from agents.qa_generator import Generator
from agents.query_rewriter import QueryRewriteAgent
from agents.rewrite_verifier import RewriteVerifyAgent
from agents.citation_validator import CitationValidatorAgent


class RetrieverFactory:
    # 进程内按 kb_id 缓存引擎，支持单库失效；线程安全且有界。
    _engines: "OrderedDict[str, HybridRetriever]" = OrderedDict()
    _lock = RLock()
    _max_engines = 32

    @classmethod
    def get_engine(cls, doc_id: str) -> HybridRetriever:
        with cls._lock:
            engine = cls._engines.get(doc_id)
            if engine is not None:
                cls._engines.move_to_end(doc_id)
                return engine

        # 在锁外构造，避免不同 kb 的冷启动相互阻塞。
        built = HybridRetriever(
            vector_retriever=VectorRetriever(collection_id=doc_id),
            bm25_retriever=BM25Retriever(collection_id=doc_id),
        )
        with cls._lock:
            engine = cls._engines.get(doc_id)
            if engine is None:
                cls._engines[doc_id] = built
                engine = built
                while len(cls._engines) > cls._max_engines:
                    cls._engines.popitem(last=False)
            cls._engines.move_to_end(doc_id)
            return engine

    @classmethod
    def invalidate(cls, doc_id: str) -> None:
        # 只失效重建过的 kb，不波及其他知识库的已缓存引擎。
        with cls._lock:
            cls._engines.pop(doc_id, None)


def rewrite_node(state: GraphState) -> dict:
    return QueryRewriteAgent.rewrite_query(state)


def verify_rewrite_node(state: GraphState) -> dict:
    # 在检索前过滤语义漂移的 query rewrite。
    return RewriteVerifyAgent.verify_rewrites(state)


def retrieve_node(state: GraphState) -> dict:
    original_query = state.get("query", "")
    doc_id = state.get("doc_id", "default")

    rewritten = state.get("rewritten_queries", [])
    queries = [original_query] if original_query else []
    for q in rewritten:
        if q not in queries:
            queries.append(q)

    engine = RetrieverFactory.get_engine(doc_id)

    retrieved_docs = []
    seen_chunk_keys = set()
    settings = get_settings()

    for query in queries:
        docs = engine.search(query=query, top_k=settings.qa_retrieval_top_k)
        for doc in docs:
            meta = doc["meta"]
            # 检索去重只认稳定 chunk_id。
            chunk_key = meta["chunk_id"]
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)

                if query != original_query:
                    doc_copy = copy.deepcopy(doc)
                    doc_copy.setdefault("retrieval", {})["rewrite_query"] = query
                else:
                    doc_copy = doc
                retrieved_docs.append(doc_copy)

    log_event(
        "qa",
        "qa_retrieve",
        state,
        query_count=len(queries),
        retrieved_count=len(retrieved_docs),
        retrieval_top_k=settings.qa_retrieval_top_k,
    )
    return {"retrieved_docs": retrieved_docs}


def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    is_local = state.get("is_local", False)

    target_device = "cpu" if is_local else BGEReranker.default_device()
    BGEReranker.set_device(target_device)

    reranked_docs = BGEReranker.rerank(
        query=query, docs=docs, top_n=get_settings().qa_rerank_top_n
    )
    log_event(
        "qa",
        "qa_rerank",
        state,
        candidate_count=len(docs),
        reranked_count=len(reranked_docs),
        device=target_device,
    )
    return {"reranked_docs": reranked_docs}


def generate_node(state: GraphState) -> dict:
    query = state.get("query", "")
    is_local = state.get("is_local", False)
    final_docs = state.get("reranked_docs", [])
    chat_history = state.get("chat_history", [])

    critique = state.get("critique", "")
    iteration_count = state.get("iteration_count", 0)

    llm = Generator._get_client(is_local=is_local)
    base_prompt = Generator.format_prompt(
        query=query, docs=final_docs, chat_history=chat_history
    )

    messages_payload = list(base_prompt)
    if critique and iteration_count > 0:
        correction_note = (
            f"\n\n【引用校验失败通知】\n"
            f"你上一轮的回答已被引用校验器拦截，错误详情如下：\n\n"
            f"{critique}\n\n"
            f"请严格按照上述修正要求重新生成答案，确保每处引用的文件名和页码与 <Document> 标签属性完全吻合。"
        )
        if messages_payload and isinstance(messages_payload[0], SystemMessage):
            messages_payload[0] = SystemMessage(
                content=messages_payload[0].content + correction_note
            )
        else:
            messages_payload.insert(0, SystemMessage(content=correction_note))

    response_message = llm.invoke(messages_payload)

    # Evidence 保留页跨度，引用校验仍按页级格式。
    evidence: list[Evidence] = [
        Evidence(
            chunk_id=doc.get("meta", {}).get("chunk_id", ""),
            chunk_index=doc.get("meta", {}).get("chunk_index", -1),
            source=doc.get("meta", {}).get("source", ""),
            page=doc.get("meta", {}).get("page", 0),
            page_start=doc.get("meta", {}).get(
                "page_start", doc.get("meta", {}).get("page", 0)
            ),
            page_end=doc.get("meta", {}).get(
                "page_end", doc.get("meta", {}).get("page", 0)
            ),
            rerank_score=doc.get("retrieval", {}).get("rerank_score"),
            rewrite_query=doc.get("retrieval", {}).get("rewrite_query"),
            text_preview=doc["text"][:100],
        )
        for doc in final_docs
    ]

    return {
        "messages": [response_message],
        "answer": response_message.content,
        "sources": [doc["meta"] for doc in final_docs],
        "evidence": evidence,
    }


def citation_node(state: GraphState) -> dict:
    answer = state.get("answer", "")
    final_docs = state.get("reranked_docs", [])
    iteration_count = state.get("iteration_count", 0)
    max_iteration_count = state.get("max_iteration_count", 2)

    check_res = CitationValidatorAgent.validate_citations(answer, final_docs)
    log_event(
        "qa",
        "qa_citation_check",
        state,
        is_valid=not bool(check_res["critique"]),
        iteration_count=iteration_count + 1,
        evidence_count=len(final_docs),
    )

    return {
        "critique": check_res["critique"],
        "iteration_count": iteration_count + 1,
        "max_iteration_count": max_iteration_count,
    }


def citation_check(state: GraphState) -> str:
    critique = state.get("critique", "")
    iteration_count = state.get("iteration_count", 0)
    max_iteration_count = state.get("max_iteration_count", 2)

    if not critique:
        return END

    if iteration_count <= max_iteration_count:
        return "generate_node"

    return END


sub_graph = StateGraph(GraphState)

sub_graph.add_node("rewrite_node", rewrite_node)
sub_graph.add_node("verify_rewrite_node", verify_rewrite_node)
sub_graph.add_node("retrieve_node", retrieve_node)
sub_graph.add_node("rerank_node", rerank_node)
sub_graph.add_node("generate_node", generate_node)
sub_graph.add_node("citation_node", citation_node)

sub_graph.add_edge(START, "rewrite_node")
sub_graph.add_edge("rewrite_node", "verify_rewrite_node")
sub_graph.add_edge("verify_rewrite_node", "retrieve_node")
sub_graph.add_edge("retrieve_node", "rerank_node")
sub_graph.add_edge("rerank_node", "generate_node")
sub_graph.add_edge("generate_node", "citation_node")

sub_graph.add_conditional_edges(
    "citation_node", citation_check, {"generate_node": "generate_node", END: END}
)

qa_subgraph_node = sub_graph.compile()
