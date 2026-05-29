from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    input: str
    count: int
    result: str


def step1(state: State) -> State:
    print(f"Step1执行第 {state['count']} 次")
    return {
        **state, 
        "count": state["count"] + 1,
        "result": f"处理了 {state['count'] + 1} 次"
    }

def check(state: State):
    print(f"检查节点：count={state['count']}")
    if state["count"] >= 3:
        return "finish"
    return "continue"

graph = StateGraph(State)
graph.add_node("step1", step1)
graph.set_entry_point("step1")

graph.add_conditional_edges(
    "step1",
    check,
    {
        "continue": "step1",
        "finish": END
    }
)

app = graph.compile()
result = app.invoke({"input": "你好", "count": 0, "result": ""})

print("最终结果:", result)
