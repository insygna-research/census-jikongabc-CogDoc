from typing import Literal
from langgraph.graph import StateGraph, START, END
from graph.state import GraphState
from agents.router import RouterAgent
from graph.subgraphs.qa import qa_subgraph_node

def dummy_summary_subgraph(state: GraphState) -> dict:
    return {"messages": [{"role": "assistant", "content": "📝 [Summary Subgraph 提示]: 论文结构化摘要总结子图功能开发中，敬请期待。"}]}

def dummy_compare_subgraph(state: GraphState) -> dict:
    return {"messages": [{"role": "assistant", "content": "📊 [Compare Subgraph 提示]: 多文档跨技术方案对比子图功能开发中，敬请期待。"}]}

def route_by_task(state: GraphState) -> Literal["qa_subgraph", "summary_subgraph", "compare_subgraph", "__end__"]:
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

workflow.add_node("intent_router", RouterAgent.route_intent)  # 智能路由节点
workflow.add_node("qa_subgraph", qa_subgraph_node)            # 真实的QA子图实体
workflow.add_node("summary_subgraph", dummy_summary_subgraph)  # 摘要子图桩
workflow.add_node("compare_subgraph", dummy_compare_subgraph)  # 对比子图桩

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