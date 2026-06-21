from typing import Literal
from langgraph.graph import StateGraph, START, END
from graph.state import GraphState
from agents.router import RouterAgent
from graph.subgraphs.qa import qa_subgraph_node
from graph.subgraphs.summary import summary_subgraph_node
from graph.subgraphs.compare import compare_subgraph_node

def route_by_task(state: GraphState) -> Literal["qa_subgraph", "summary_subgraph", "compare_subgraph", "__end__"]:
    # 路由结果只允许落到已注册的子图节点。
    task = state.get("task_type", "qa")

    if task == "qa":
        return "qa_subgraph"
    elif task == "summary":
        return "summary_subgraph"
    elif task == "compare":
        return "compare_subgraph"
    else:
        return "__end__"

workflow = StateGraph(GraphState)

workflow.add_node("intent_router", RouterAgent.route_intent)
workflow.add_node("qa_subgraph", qa_subgraph_node)
workflow.add_node("summary_subgraph", summary_subgraph_node)
workflow.add_node("compare_subgraph", compare_subgraph_node)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    route_by_task,
    {
        "qa_subgraph": "qa_subgraph",
        "summary_subgraph": "summary_subgraph",
        "compare_subgraph": "compare_subgraph",
        "__end__": END
    }
)
workflow.add_edge("qa_subgraph", END)
workflow.add_edge("summary_subgraph", END)
workflow.add_edge("compare_subgraph", END)

app = workflow.compile()
