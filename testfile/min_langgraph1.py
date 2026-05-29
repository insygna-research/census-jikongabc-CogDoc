from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    input: str
    step1_result: str
    step2_result: str


def step1(state: State) -> State:
    print(f"Step1收到: {state['input']}")
    return {
        **state, 
        "step1_result": f"处理后:{state['input']}"
            
    }

def step2(state: State) -> State:
    print(f"Step2收到: {state['step1_result']}")
    return {
        **state, 
        "step2_result": f"完成:{state['step1_result']}"
    }

graph = StateGraph(State)
graph.add_node("步骤1", step1)
graph.add_node("步骤2", step2)
graph.set_entry_point("步骤1")
graph.add_edge("步骤1", "步骤2")
graph.add_edge("步骤2", END)

app = graph.compile()
result = app.invoke({"input": "你好", "step1_result": "", "step2_result": ""})

print("最终结果:", result["step2_result"])
