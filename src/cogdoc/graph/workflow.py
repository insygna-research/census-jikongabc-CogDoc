from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from cogdoc.graph.state import GraphState
from cogdoc.agents.router import RouterAgent
from cogdoc.graph.subgraphs.qa import qa_subgraph_node
from cogdoc.graph.subgraphs.summary import summary_subgraph_node
from cogdoc.graph.subgraphs.compare import compare_subgraph_node


UNKNOWN_RESPONSE = (
    "我是面向本地知识库的文档问答助手，你这条更像闲聊或与库内文档无关。"
    "可以问我库里文档的内容，或用 /summary、/compare 指定模式。"
)


# 路由 by task。
def route_by_task(
    state: GraphState,
) -> Literal["qa_subgraph", "summary_subgraph", "compare_subgraph", "unknown_node"]:
    # 路由结果只允许落到已注册的子图节点。
    task = state.get("task_type", "qa")

    if task == "qa":
        return "qa_subgraph"
    elif task == "summary":
        return "summary_subgraph"
    elif task == "compare":
        return "compare_subgraph"
    else:
        return "unknown_node"


# 完成 未知意图node 处理。
def unknown_node(state: GraphState) -> dict:
    answer = UNKNOWN_RESPONSE
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


workflow = StateGraph(GraphState)

workflow.add_node("intent_router", RouterAgent.route_intent)
workflow.add_node("qa_subgraph", qa_subgraph_node)
workflow.add_node("summary_subgraph", summary_subgraph_node)
workflow.add_node("compare_subgraph", compare_subgraph_node)
workflow.add_node("unknown_node", unknown_node)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    route_by_task,
    {
        "qa_subgraph": "qa_subgraph",
        "summary_subgraph": "summary_subgraph",
        "compare_subgraph": "compare_subgraph",
        "unknown_node": "unknown_node",
    },
)
workflow.add_edge("qa_subgraph", END)
workflow.add_edge("summary_subgraph", END)
workflow.add_edge("compare_subgraph", END)
workflow.add_edge("unknown_node", END)

app = workflow.compile()
