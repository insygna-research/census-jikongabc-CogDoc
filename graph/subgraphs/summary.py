from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

from agents.summary_planner import SectionPlannerAgent
from agents.summary_generator import GlobalSummaryAgent, SectionSummaryAgent
from graph.state import GraphState
from graph.subgraphs.qa import RetrieverFactory
from tools.document_loader import select_source_for_summary


def document_loader_node(state: GraphState) -> dict:
    # Summary MVP 从当前索引直接加载单个 source 的全部 chunk。
    query = state.get("query", "")
    doc_id = state.get("doc_id", "default")
    engine = RetrieverFactory.get_engine(doc_id)
    sources = engine.list_sources()
    selected_source = select_source_for_summary(query, sources)

    if selected_source is None:
        source_list = "，".join(sources) if sources else "当前知识库没有可用文档"
        message = (
            "请在摘要问题中明确指定要总结的文件名。"
            f"当前可用文档：{source_list}"
        )
        return {
            "summary_source": "",
            "summary_docs": [],
            "answer": message,
            "messages": [AIMessage(content = message)],
        }

    docs = engine.load_source_chunks(selected_source)
    if not docs:
        message = f"未能从当前索引加载文档：{selected_source}。请重建索引后再试。"
        return {
            "summary_source": selected_source,
            "summary_docs": [],
            "answer": message,
            "messages": [AIMessage(content = message)],
        }

    return {
        "summary_source": selected_source,
        "summary_docs": docs,
        "steps_trace": [
            {
                "step_name": "summary_document_loader",
                "input_summary": query,
                "output_summary": f"{selected_source}: {len(docs)} chunks",
            }
        ],
    }


def section_planner_node(state: GraphState) -> dict:
    return SectionPlannerAgent.plan_sections(state)


def section_summary_node(state: GraphState) -> dict:
    return SectionSummaryAgent.summarize_sections(state)


def global_summary_node(state: GraphState) -> dict:
    return GlobalSummaryAgent.build_final_summary(state)


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
