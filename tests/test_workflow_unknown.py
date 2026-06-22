import importlib

from agents.router import RouterAgent
from graph import workflow
from graph.workflow import route_by_task, unknown_node


def test_route_by_task_sends_unknown_to_terminal_node():
    assert route_by_task({"task_type": "unknown"}) == "unknown_node"


def test_unknown_node_returns_readable_answer_and_message():
    result = unknown_node({"task_type": "unknown"})

    assert "本地知识库" in result["answer"]
    assert result["messages"]
    assert result["messages"][0].content == result["answer"]


def test_workflow_unknown_route_produces_answer(monkeypatch):
    def fake_route_intent(state, config):
        return {
            "query": "你好",
            "doc_id": "kb",
            "is_local": False,
            "task_type": "unknown",
            "router_reason": "纯闲聊",
        }

    with monkeypatch.context() as patcher:
        patcher.setattr(RouterAgent, "route_intent", staticmethod(fake_route_intent))
        reloaded = importlib.reload(workflow)
        result = reloaded.app.invoke(
            {
                "messages": [],
                "chat_history": [],
                "iteration_count": 0,
                "max_iteration_count": 2,
            },
            config={"configurable": {"query": "你好", "doc_id": "kb"}},
        )

    importlib.reload(workflow)

    assert result["task_type"] == "unknown"
    assert "本地知识库" in result["answer"]
