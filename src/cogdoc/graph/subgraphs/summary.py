from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from cogdoc.agents.summary_planner import SectionPlannerAgent
from cogdoc.agents.summary_generator import GlobalSummaryAgent, SectionSummaryAgent
from cogdoc.agents.source_resolver import resolve_summary_source
from cogdoc.graph.state import GraphState
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.observability.logger import log_event
from cogdoc.service.kb_readers import kb_read_lease
from cogdoc.tools.document_loader import select_source_for_summary


# 处理 document loader node 相关逻辑。
def document_loader_node(state: GraphState) -> dict:
    # Summary MVP 从当前索引直接加载单个 source 的全部 chunk。
    query = state.get("query", "")
    doc_id = state.get("doc_id", "default")
    is_local = state.get("is_local", False)
    with kb_read_lease(doc_id):
        engine = RetrieverFactory.get_engine(doc_id)
        sources = engine.list_sources()
    selected_source = select_source_for_summary(query, sources)

    resolution_trace = []
    if selected_source is None and sources and state.get("chat_history"):
        # 字面匹配不到时，用近期对话消解“总结这个文件/上面那篇”等多轮指代。
        resolved = resolve_summary_source(
            query, sources, state.get("chat_history"), is_local
        )
        if resolved:
            selected_source = resolved
            resolution_trace = [
                {
                    "step_name": "summary_source_resolution",
                    "input_summary": query,
                    "output_summary": resolved,
                }
            ]

    if selected_source is None:
        source_list = "，".join(sources) if sources else "当前知识库没有可用文档"
        message = (
            "请在摘要问题中明确指定要总结的文件名（可直接说出文件名）。"
            f"当前可用文档：{source_list}"
        )
        result = {
            "summary_source": "",
            "summary_docs": [],
            "answer": message,
            "messages": [AIMessage(content=message)],
        }
        log_event(
            "summary",
            "summary_document_loader",
            state,
            selected=False,
            source_count=len(sources),
        )
        return result

    with kb_read_lease(doc_id):
        docs = RetrieverFactory.get_engine(doc_id).load_source_chunks(selected_source)
    if not docs:
        message = f"未能从当前索引加载文档：{selected_source}。请重建索引后再试。"
        result = {
            "summary_source": selected_source,
            "summary_docs": [],
            "answer": message,
            "messages": [AIMessage(content=message)],
        }
        log_event(
            "summary",
            "summary_document_loader",
            state,
            selected=True,
            loaded=False,
            source=selected_source,
        )
        return result

    result = {
        "summary_source": selected_source,
        "summary_docs": docs,
        "steps_trace": resolution_trace
        + [
            {
                "step_name": "summary_document_loader",
                "input_summary": query,
                "output_summary": f"{selected_source}: {len(docs)} chunks",
            }
        ],
    }
    log_event(
        "summary",
        "summary_document_loader",
        state,
        selected=True,
        loaded=True,
        source=selected_source,
        chunk_count=len(docs),
    )
    return result


# 处理 section planner node 相关逻辑。
def section_planner_node(state: GraphState) -> dict:
    return SectionPlannerAgent.plan_sections(state)


# 处理 section summary node 相关逻辑。
def section_summary_node(state: GraphState) -> dict:
    return SectionSummaryAgent.summarize_sections(state)


# 处理 global summary node 相关逻辑。
def global_summary_node(state: GraphState) -> dict:
    return GlobalSummaryAgent.build_final_summary(state)


# 处理 document loader check 相关逻辑。
def document_loader_check(state: GraphState) -> str:
    # 文档加载失败时直接结束，避免下游节点在空 docs 上运行。
    if not state.get("summary_docs"):
        return END
    return "section_planner_node"


summary_graph = StateGraph(GraphState)

summary_graph.add_node("document_loader_node", document_loader_node)
summary_graph.add_node("section_planner_node", section_planner_node)
summary_graph.add_node("section_summary_node", section_summary_node)
summary_graph.add_node("global_summary_node", global_summary_node)

summary_graph.add_edge(START, "document_loader_node")
summary_graph.add_conditional_edges(
    "document_loader_node",
    document_loader_check,
    {
        "section_planner_node": "section_planner_node",
        END: END,
    },
)
summary_graph.add_edge("section_planner_node", "section_summary_node")
summary_graph.add_edge("section_summary_node", "global_summary_node")
summary_graph.add_edge("global_summary_node", END)

summary_subgraph_node = summary_graph.compile()
