import copy
import logging
from collections import OrderedDict
from threading import RLock
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from cogdoc.config.settings import get_settings
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore
from cogdoc.graph.state import GraphState, Evidence, RetrievedDoc
from cogdoc.observability.logger import log_event
from cogdoc.service.kb_lifecycle import LIFECYCLE_ACTIVE, shared_lifecycle_store
from cogdoc.service.kb_readers import kb_read_lease
from cogdoc.service.kb_state import KBState
from cogdoc.tools.retriever.base_retriever import NullRetriever
from cogdoc.tools.retriever.vector_retriever import (
    VectorRetriever,
    EmbeddingModelMismatchError,
)
from cogdoc.tools.retriever.bm25_retriever import BM25Retriever
from cogdoc.tools.retriever.derived_knowledge import DerivedKnowledgeRetriever
from cogdoc.tools.retriever.hybrid import HybridRetriever, IndexCorruptError
from cogdoc.tools.retriever.metadata import safe_retrieval_metadata
from cogdoc.tools.retriever.confidence import assess_retrieval_support
from cogdoc.tools.reranker import BGEReranker, skipped_cpu_rerank_docs
from cogdoc.agents.answer_markers import NO_RELEVANT_CONTENT_ANSWER
from cogdoc.agents.qa_generator import Generator
from cogdoc.agents.query_rewriter import QueryRewriteAgent
from cogdoc.agents.rewrite_verifier import RewriteVerifyAgent
from cogdoc.agents.citation_validator import CitationValidatorAgent


NEIGHBOR_CONTEXT_RADIUS = 1
_derived_knowledge_retriever = DerivedKnowledgeRetriever()
_retrieval_feedback_store = RetrievalFeedbackStore()


# 进程内按知识库和索引代缓存引擎，切代后失效缓存。
class RetrieverFactory:
    _engines: "OrderedDict[tuple, HybridRetriever]" = OrderedDict()
    _lock = RLock()
    _max_engines = 32

    # 返回检索引擎。
    @classmethod
    def get_engine(cls, kb_id: str) -> HybridRetriever:
        # 删库进行中/已删：禁读正在拆除的代，返回空引擎且不缓存。
        if shared_lifecycle_store().status(kb_id) != LIFECYCLE_ACTIVE:
            return HybridRetriever(NullRetriever(), NullRetriever())

        # 锁外解析活跃索引代，以知识库和索引代为缓存键查缓存。
        gen_id = cls._resolve_gen_id(kb_id)
        cache_key = (kb_id, gen_id)

        with cls._lock:
            engine = cls._engines.get(cache_key)
            if engine is not None:
                # 缓存命中前再查一次生命周期，避免删库中继续读旧引擎。
                if shared_lifecycle_store().status(kb_id) != LIFECYCLE_ACTIVE:
                    return HybridRetriever(NullRetriever(), NullRetriever())
                cls._engines.move_to_end(cache_key)
                return engine

        # 锁外构造，不同知识库冷启动互不阻塞。
        built = cls._build_engine(kb_id, gen_id)

        with cls._lock:
            # 构造期间被删库则丢弃，返回空引擎且不缓存。
            if shared_lifecycle_store().status(kb_id) != LIFECYCLE_ACTIVE:
                return HybridRetriever(NullRetriever(), NullRetriever())
            # 插入前重新解析，防止构造期间切代导致索引代已失效。
            current_gen_id = cls._resolve_gen_id(kb_id)
            if current_gen_id != gen_id:
                # 代已切换，丢弃刚构造的引擎不写回缓存；下次请求自然构造新代引擎。
                return built
            engine = cls._engines.get(cache_key)
            if engine is None:
                cls._engines[cache_key] = built
                engine = built
                while len(cls._engines) > cls._max_engines:
                    cls._engines.popitem(last=False)
            cls._engines.move_to_end(cache_key)
            return engine

    # 解析索引代标识。
    @classmethod
    def _resolve_gen_id(cls, kb_id: str) -> str | None:
        # 读取活跃索引代标识；无活跃代或合法空索引均返回空。
        active = KBState(kb_id).active()
        if active is None or active.get("expected_count") == 0:
            return None
        return active["id"]

    # 构建检索引擎。
    @classmethod
    def _build_engine(cls, kb_id: str, gen_id: str | None) -> HybridRetriever:
        if gen_id is None:
            # 无活跃代或合法空索引：返回空引擎，不报错。
            return HybridRetriever(NullRetriever(), NullRetriever())

        collection_id = get_settings().kb_collection_id(kb_id, gen_id)
        try:
            engine = HybridRetriever(
                vector_retriever=VectorRetriever(collection_id=collection_id),
                bm25_retriever=BM25Retriever(collection_id=collection_id),
            )
        except EmbeddingModelMismatchError:
            # 嵌入模型已更换，当前代向量集合不可用时返回空引擎。
            return HybridRetriever(NullRetriever(), NullRetriever())

        # 按索引代精确读取记录，避免切代时用新代计数校验旧代引擎。
        gen_state = KBState(kb_id).get(gen_id)
        if gen_state is None:
            # 该代在构造期间已被回收，返回空引擎且不缓存。
            return HybridRetriever(NullRetriever(), NullRetriever())
        expected = gen_state.get("expected_count")
        actual = engine.count()
        consistent = engine.is_consistent()
        if actual != expected or not consistent:
            raise IndexCorruptError(
                f"generation {gen_id}: expected_count={expected}, actual={actual}, "
                f"consistent={consistent}; rebuild required"
            )
        return engine

    # 使失效结果。
    @classmethod
    def invalidate(cls, kb_id: str) -> None:
        # 删除该知识库的全部代缓存，强制下次重解析活跃代。
        with cls._lock:
            stale_keys = [k for k in cls._engines if k[0] == kb_id]
            for k in stale_keys:
                del cls._engines[k]


# 处理问题改写节点。
def rewrite_node(state: GraphState) -> dict:
    return QueryRewriteAgent.rewrite_query(state)


# 校验问题改写节点。
def verify_rewrite_node(state: GraphState) -> dict:
    # 在检索前过滤语义漂移的问题改写。
    return RewriteVerifyAgent.verify_rewrites(state)


# 定位命中文本块在源文档文本块序列中的位置。
def _find_source_chunk_index(
    source_chunks: list[RetrievedDoc], target_doc: RetrievedDoc
) -> int:
    target_meta = target_doc.get("meta", {})
    target_id = str(target_meta.get("chunk_id", ""))
    if target_id:
        for idx, doc in enumerate(source_chunks):
            if str(doc.get("meta", {}).get("chunk_id", "") or "") == target_id:
                return idx

    target_local = target_meta.get("local_chunk_index")
    if target_local is None:
        return -1
    for idx, doc in enumerate(source_chunks):
        if doc.get("meta", {}).get("local_chunk_index") == target_local:
            return idx
    return -1


# 复制相邻文本块并标记上下文来源。
def _copy_neighbor_doc(doc: RetrievedDoc, parent_chunk_id: str) -> RetrievedDoc:
    copied = copy.deepcopy(doc)
    retrieval = copied.setdefault("retrieval", {})
    retrieval["search_channel"] = "neighbor"
    retrieval["parent_chunk_id"] = parent_chunk_id
    return copied


# 生成缺失文本块标识时的临时去重键。
def _missing_chunk_key(expanded: "OrderedDict[str, RetrievedDoc]") -> str:
    return f"__missing_chunk_id_{len(expanded)}"


# 为重排命中文本块补充前后相邻文本块，降低答案缺上下文的概率。
def _expand_with_neighbor_chunks(
    doc_id: str, reranked_docs: list[RetrievedDoc], state: GraphState | None = None
) -> list[RetrievedDoc]:
    if not reranked_docs:
        return []

    expanded: "OrderedDict[str, RetrievedDoc]" = OrderedDict()
    source_cache: dict[str, list[RetrievedDoc]] = {}
    try:
        with kb_read_lease(doc_id):
            engine = RetrieverFactory.get_engine(doc_id)
            for doc in reranked_docs:
                meta = doc.get("meta", {})
                if meta.get("source_type") == "derived_knowledge":
                    expanded[
                        str(meta.get("chunk_id", "")) or _missing_chunk_key(expanded)
                    ] = copy.deepcopy(doc)
                    continue
                source = str(meta.get("source", "") or "")
                parent_chunk_id = str(meta.get("chunk_id", ""))
                if not source or not parent_chunk_id:
                    expanded[parent_chunk_id or _missing_chunk_key(expanded)] = (
                        copy.deepcopy(doc)
                    )
                    continue

                if source not in source_cache:
                    source_cache[source] = engine.load_source_chunks(source)
                source_chunks = source_cache[source]
                hit_idx = _find_source_chunk_index(source_chunks, doc)
                if hit_idx < 0:
                    expanded[parent_chunk_id or _missing_chunk_key(expanded)] = (
                        copy.deepcopy(doc)
                    )
                    continue

                start = max(0, hit_idx - NEIGHBOR_CONTEXT_RADIUS)
                end = min(len(source_chunks), hit_idx + NEIGHBOR_CONTEXT_RADIUS + 1)
                for idx in range(start, end):
                    neighbor = source_chunks[idx]
                    neighbor_id = str(
                        neighbor.get("meta", {}).get("chunk_id", "") or ""
                    )
                    if not neighbor_id:
                        continue
                    if neighbor_id == parent_chunk_id:
                        expanded[neighbor_id] = copy.deepcopy(doc)
                    elif neighbor_id not in expanded:
                        expanded[neighbor_id] = _copy_neighbor_doc(
                            neighbor, parent_chunk_id
                        )
    except Exception as exc:
        log_event(
            "qa",
            "qa_neighbor_expand_failed",
            state,
            level=logging.WARNING,
            error_class=type(exc).__name__,
        )
        return reranked_docs

    return list(expanded.values())


# 应用检索反馈调权。
def _apply_retrieval_feedback(
    kb_id: str, query: str, docs: list[RetrievedDoc]
) -> list[RetrievedDoc]:
    if not docs or not query:
        return docs
    try:
        boosts = _retrieval_feedback_store.boosts_for_query(kb_id, query)
    except Exception as exc:
        log_event(
            "qa",
            "retrieval_feedback_boost_failed",
            {},
            level=logging.WARNING,
            kb_id=kb_id,
            error_class=type(exc).__name__,
        )
        return docs
    if not boosts:
        return docs
    adjusted = []
    for idx, doc in enumerate(docs):
        chunk_id = str(doc.get("meta", {}).get("chunk_id", "") or "")
        boost = boosts.get(chunk_id, 0.0)
        if boost:
            doc = copy.deepcopy(doc)
            retrieval = doc.setdefault("retrieval", {})
            retrieval["feedback_boost"] = boost
        adjusted.append((idx, boost, doc))
    adjusted.sort(key=lambda item: (-item[1], item[0]))
    return [doc for _, _, doc in adjusted]


# 检索节点。
def retrieve_node(state: GraphState) -> dict:
    original_query = state.get("query", "")
    doc_id = state.get("doc_id", "default")

    rewritten = state.get("rewritten_queries", [])
    queries = [original_query] if original_query else []
    for q in rewritten:
        if q not in queries:
            queries.append(q)

    retrieved_docs = []
    seen_chunk_keys = set()
    settings = get_settings()
    with kb_read_lease(doc_id):
        engine = RetrieverFactory.get_engine(doc_id)
        for query in queries:
            docs = engine.search(query=query, top_k=settings.qa_retrieval_top_k)
            for doc in docs:
                meta = doc["meta"]
                # 检索去重只认稳定分块标识。
                chunk_key = meta["chunk_id"]
                if chunk_key not in seen_chunk_keys:
                    seen_chunk_keys.add(chunk_key)

                    if query != original_query:
                        doc_copy = copy.deepcopy(doc)
                        doc_copy.setdefault("retrieval", {})["rewrite_query"] = query
                    else:
                        doc_copy = doc
                    retrieved_docs.append(doc_copy)
            knowledge_docs = _derived_knowledge_retriever.search(
                doc_id, query, top_k=settings.qa_retrieval_top_k
            )
            for doc in knowledge_docs:
                meta = doc["meta"]
                chunk_key = meta["chunk_id"]
                if chunk_key not in seen_chunk_keys:
                    seen_chunk_keys.add(chunk_key)
                    if query != original_query:
                        doc_copy = copy.deepcopy(doc)
                        doc_copy.setdefault("retrieval", {})["rewrite_query"] = query
                    else:
                        doc_copy = doc
                    retrieved_docs.append(doc_copy)

    retrieved_docs = _apply_retrieval_feedback(doc_id, original_query, retrieved_docs)

    log_event(
        "qa",
        "qa_retrieve",
        state,
        query_count=len(queries),
        retrieved_count=len(retrieved_docs),
        retrieval_top_k=settings.qa_retrieval_top_k,
    )
    return {"retrieved_docs": retrieved_docs}


# 重排节点。
def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    doc_id = state.get("doc_id", "default")
    settings = get_settings()

    target_device = BGEReranker.default_device()

    max_candidates = max(settings.qa_rerank_max_candidates, settings.qa_rerank_top_n)
    candidate_docs = docs[:max_candidates] if max_candidates > 0 else docs
    rerank_skipped_reason = ""
    if target_device == "cpu" and not settings.qa_rerank_on_cpu:
        rerank_skipped_reason = "cpu_disabled"
        reranked_docs = skipped_cpu_rerank_docs(
            candidate_docs, settings.qa_rerank_top_n, rerank_skipped_reason
        )
    else:
        reranked_docs = BGEReranker.rerank(
            query=query,
            docs=candidate_docs,
            top_n=settings.qa_rerank_top_n,
            device=target_device,
        )
    support = assess_retrieval_support(reranked_docs, settings)
    # 下游沿用重排结果字段名，实际内容已包含相邻上下文扩展。
    expanded_docs = _expand_with_neighbor_chunks(doc_id, reranked_docs, state)
    log_event(
        "qa",
        "qa_rerank",
        state,
        candidate_count=len(docs),
        rerank_candidate_count=len(candidate_docs),
        reranked_count=len(reranked_docs),
        expanded_count=len(expanded_docs),
        device=target_device,
        rerank_skipped_reason=rerank_skipped_reason,
        retrieval_confidence=round(support.score, 6),
        retrieval_abstained=not support.supported,
        retrieval_abstain_reason=support.reason,
    )
    return {
        "reranked_docs": expanded_docs,
        "retrieval_confidence": support.score,
        "retrieval_abstained": not support.supported,
        "retrieval_abstain_reason": support.reason,
        "retrieval_signals": support.signals,
    }


# 证据不足时不调用 LLM，返回稳定拒答并清空候选，避免无关证据进入引用和会话记忆。
def abstain_node(state: GraphState) -> dict:
    log_event(
        "qa",
        "qa_retrieval_abstained",
        state,
        retrieval_confidence=round(state.get("retrieval_confidence", 0.0), 6),
        retrieval_abstain_reason=state.get("retrieval_abstain_reason", ""),
    )
    return {
        "messages": [AIMessage(content=NO_RELEVANT_CONTENT_ANSWER)],
        "answer": NO_RELEVANT_CONTENT_ANSWER,
        "sources": [],
        "evidence": [],
        "reranked_docs": [],
        "critique": "",
        "retrieval_abstained": True,
    }


# 根据检索置信度选择生成或确定性拒答。
def retrieval_check(state: GraphState) -> str:
    return "abstain_node" if state.get("retrieval_abstained", False) else "generate_node"


# 生成节点。
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

    # 证据保留页跨度，引用校验仍按页级格式。
    evidence: list[Evidence] = [
        Evidence(
            chunk_id=doc.get("meta", {}).get("chunk_id", ""),
            source_type=doc.get("meta", {}).get("source_type", "document"),
            knowledge_id=doc.get("meta", {}).get("knowledge_id", ""),
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
            retrieval=safe_retrieval_metadata(doc.get("retrieval")),
        )
        for doc in final_docs
    ]

    return {
        "messages": [response_message],
        "answer": response_message.content,
        "sources": [doc["meta"] for doc in final_docs],
        "evidence": evidence,
    }


# 处理引用节点。
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


# 处理引用检查。
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
sub_graph.add_node("abstain_node", abstain_node)
sub_graph.add_node("generate_node", generate_node)
sub_graph.add_node("citation_node", citation_node)

sub_graph.add_edge(START, "rewrite_node")
sub_graph.add_edge("rewrite_node", "verify_rewrite_node")
sub_graph.add_edge("verify_rewrite_node", "retrieve_node")
sub_graph.add_edge("retrieve_node", "rerank_node")
sub_graph.add_conditional_edges(
    "rerank_node",
    retrieval_check,
    {"abstain_node": "abstain_node", "generate_node": "generate_node"},
)
sub_graph.add_edge("abstain_node", END)
sub_graph.add_edge("generate_node", "citation_node")

sub_graph.add_conditional_edges(
    "citation_node", citation_check, {"generate_node": "generate_node", END: END}
)

qa_subgraph_node = sub_graph.compile()
