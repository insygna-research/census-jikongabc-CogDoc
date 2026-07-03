from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from cogdoc.agents.compare_generator import CompareGeneratorAgent
from cogdoc.agents.compare_profile import DocumentProfileAgent, default_compare_dimensions
from cogdoc.agents.source_resolver import resolve_compare_sources
from cogdoc.graph.state import GraphState
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.observability.logger import log_event
from cogdoc.service.kb_readers import kb_read_lease
from cogdoc.tools.document_loader import select_sources_for_compare


LOCAL_COMPARE_MAX_SOURCES = 2


# 完成 documentloadernode 处理。
def document_loader_node(state: GraphState) -> dict:
    # Compare MVP 只处理用户显式点名的多文档对比。
    query = state.get("query", "")
    doc_id = state.get("doc_id", "default")
    is_local = state.get("is_local", False)
    with kb_read_lease(doc_id):
        engine = RetrieverFactory.get_engine(doc_id)
        sources = engine.list_sources()
    selected_sources = select_sources_for_compare(query, sources)

    resolution_trace = []
    if len(selected_sources) < 2 and sources and state.get("chat_history"):
        # 字面匹配不足时，用近期对话消解“这个文件/上面那篇”等多轮指代。
        resolved = resolve_compare_sources(
            query, sources, state.get("chat_history"), is_local
        )
        if len(resolved) >= 2:
            selected_sources = resolved
            resolution_trace = [
                {
                    "step_name": "compare_source_resolution",
                    "input_summary": query,
                    "output_summary": "，".join(resolved),
                }
            ]

    if len(selected_sources) < 2:
        source_list = "，".join(sources) if sources else "当前知识库没有可用文档"
        message = (
            "请在对比问题中点名至少 2 篇要对比的文件（可直接说出文件名）。"
            f"当前可用文档：{source_list}"
        )
        result = {
            "compare_sources": [],
            "compare_docs_by_source": {},
            "answer": message,
            "messages": [AIMessage(content=message)],
        }
        log_event(
            "compare",
            "compare_document_loader",
            state,
            selected_count=len(selected_sources),
            source_count=len(sources),
        )
        return result

    if is_local and len(selected_sources) > LOCAL_COMPARE_MAX_SOURCES:
        # 本地模式限制文档数，避免文档数乘维度数放大 LLM 调用。
        selected_list = "，".join(selected_sources)
        message = (
            f"本地 Ollama 模式最多支持同时对比 {LOCAL_COMPARE_MAX_SOURCES} 篇文档，"
            f"当前点名了 {len(selected_sources)} 篇：{selected_list}。"
            "请只保留 2 篇文档后重试，或切换到云端 API 模式。"
        )
        result = {
            "compare_sources": [],
            "compare_docs_by_source": {},
            "answer": message,
            "messages": [AIMessage(content=message)],
        }
        log_event(
            "compare",
            "compare_document_loader",
            state,
            selected_count=len(selected_sources),
            local_limit=LOCAL_COMPARE_MAX_SOURCES,
            limited=True,
        )
        return result

    docs_by_source = {}
    with kb_read_lease(doc_id):
        engine = RetrieverFactory.get_engine(doc_id)
        for source in selected_sources:
            docs = engine.load_source_chunks(source)
            if docs:
                docs_by_source[source] = docs

    if len(docs_by_source) < 2:
        loaded_sources = "，".join(docs_by_source.keys()) if docs_by_source else "无"
        message = (
            "未能从当前索引加载至少 2 篇对比文档。"
            f"已加载：{loaded_sources}。请重建索引后再试。"
        )
        result = {
            "compare_sources": list(docs_by_source.keys()),
            "compare_docs_by_source": docs_by_source,
            "answer": message,
            "messages": [AIMessage(content=message)],
        }
        log_event(
            "compare",
            "compare_document_loader",
            state,
            selected_count=len(selected_sources),
            loaded_count=len(docs_by_source),
        )
        return result

    compare_sources = list(docs_by_source.keys())
    result = {
        "compare_sources": compare_sources,
        "compare_docs_by_source": docs_by_source,
        "compare_dimensions": state.get("compare_dimensions")
        or default_compare_dimensions(is_local=is_local),
        "steps_trace": resolution_trace
        + [
            {
                "step_name": "compare_document_loader",
                "input_summary": query,
                "output_summary": "，".join(
                    f"{source}: {len(docs_by_source[source])} chunks"
                    for source in compare_sources
                ),
            }
        ],
    }
    log_event(
        "compare",
        "compare_document_loader",
        state,
        selected_count=len(compare_sources),
        loaded_count=len(docs_by_source),
        chunk_count=sum(len(docs) for docs in docs_by_source.values()),
    )
    return result


# 完成 document画像node 处理。
def document_profile_node(state: GraphState) -> dict:
    try:
        return DocumentProfileAgent.build_profiles(state)
    except Exception as exc:
        # 模型异常转为可打印答案，避免流式管道中断。
        if state.get("is_local", False):
            suggestion = "建议：释放内存或执行 `ollama stop <model>` 后重试；也可以切换到更小的本地模型或云端 API。"
        else:
            suggestion = "建议：稍后重试，或检查云端 API 配置与网络状态。"
        message = (
            f"模型生成对比画像失败。错误：{type(exc).__name__}: {exc}\n{suggestion}"
        )
        result = {
            "document_profiles": [],
            "answer": message,
            "messages": [AIMessage(content=message)],
            "error": str(exc),
            "steps_trace": [
                {
                    "step_name": "compare_document_profile_error",
                    "input_summary": state.get("query", ""),
                    "output_summary": message,
                }
            ],
        }
        log_event(
            "compare",
            "compare_document_profile_error",
            state,
            error_class=type(exc).__name__,
        )
        return result


# 生成对比 table node。
def compare_table_node(state: GraphState) -> dict:
    return CompareGeneratorAgent.build_compare_answer(state)


# 完成 引用node 处理。
def citation_node(state: GraphState) -> dict:
    return CompareGeneratorAgent.validate_compare_answer(state)


# 完成 documentloadercheck 处理。
def document_loader_check(state: GraphState) -> str:
    if len(state.get("compare_docs_by_source", {})) < 2:
        return END
    return "document_profile_node"


# 完成 document画像check 处理。
def document_profile_check(state: GraphState) -> str:
    if not state.get("document_profiles"):
        return END
    return "compare_table_node"


compare_graph = StateGraph(GraphState)

compare_graph.add_node("document_loader_node", document_loader_node)
compare_graph.add_node("document_profile_node", document_profile_node)
compare_graph.add_node("compare_table_node", compare_table_node)
# 避免 run.py 将 compare 校验误识别为 QA citation 节点。
compare_graph.add_node("compare_citation_node", citation_node)

compare_graph.add_edge(START, "document_loader_node")
compare_graph.add_conditional_edges(
    "document_loader_node",
    document_loader_check,
    {
        "document_profile_node": "document_profile_node",
        END: END,
    },
)
compare_graph.add_conditional_edges(
    "document_profile_node",
    document_profile_check,
    {
        "compare_table_node": "compare_table_node",
        END: END,
    },
)
compare_graph.add_edge("compare_table_node", "compare_citation_node")
compare_graph.add_edge("compare_citation_node", END)

compare_subgraph_node = compare_graph.compile()
