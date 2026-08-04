import importlib
from cogdoc.agents.router import RouterAgent
from cogdoc.config.settings import Settings
from cogdoc.graph import workflow
from cogdoc.graph.workflow import route_by_task, unknown_node


# 验证 route by task sends unknown to terminal node 场景。
def test_route_by_task_sends_unknown_to_terminal_node():
    assert route_by_task({"task_type": "unknown"}) == "unknown_node"


# 验证 unknown node returns readable answer and message 场景。
def test_unknown_node_returns_readable_answer_and_message():
    result = unknown_node({"task_type": "unknown"})

    assert "本地知识库" in result["answer"]
    assert result["messages"]
    assert result["messages"][0].content == result["answer"]


# 验证 workflow unknown route produces answer 场景。
def test_workflow_unknown_route_produces_answer(monkeypatch):
    # 构造route意图。
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


# 验证三个 RAG 任务都必须经过父图声明审计，unknown 响应则直接结束。
def test_workflow_places_claim_gate_after_all_rag_subgraphs():
    edges = {
        (edge.source, edge.target) for edge in workflow.app.get_graph().edges
    }

    assert ("qa_subgraph", "claim_audit_node") in edges
    assert ("summary_subgraph", "claim_audit_node") in edges
    assert ("compare_subgraph", "claim_audit_node") in edges
    assert ("unknown_node", "claim_audit_node") not in edges


# 验证失败声明只有有限修复机会，校验器错误立即走稳定拦截。
def test_claim_audit_route_is_bounded_and_fail_closed(monkeypatch):
    settings = Settings(
        _env_file=None,
        claim_verification_max_repair_attempts=1,
    )
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)

    assert workflow.claim_audit_check(
        {"claim_audit": {"status": "passed"}}
    ) == workflow.END
    assert workflow.claim_audit_check(
        {"claim_audit": {"status": "failed"}, "claim_repair_count": 0}
    ) == "claim_repair_node"
    assert workflow.claim_audit_check(
        {"claim_audit": {"status": "failed"}, "claim_repair_count": 1}
    ) == "claim_block_node"
    assert workflow.claim_audit_check(
        {"claim_audit": {"status": "error"}}
    ) == "claim_block_node"
    assert workflow.claim_repair_check(
        {"claim_repair_error": "TimeoutError"}
    ) == "claim_block_node"
    assert workflow.claim_repair_citation_check(
        {"claim_repair_citation_valid": True, "claim_repair_count": 1}
    ) == "claim_audit_node"
    assert workflow.claim_repair_citation_check(
        {"claim_repair_citation_valid": False, "claim_repair_count": 0}
    ) == "claim_repair_node"
    assert workflow.claim_repair_citation_check(
        {"claim_repair_citation_valid": False, "claim_repair_count": 1}
    ) == "claim_block_node"
