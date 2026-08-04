from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from cogdoc.graph.state import GraphState
from cogdoc.agents.router import RouterAgent
from cogdoc.graph.subgraphs.qa import qa_subgraph_node
from cogdoc.graph.subgraphs.summary import summary_subgraph_node
from cogdoc.graph.subgraphs.compare import compare_subgraph_node
from cogdoc.agents.claim_evidence_verifier import (
    ClaimEvidenceVerifierAgent,
    ClaimRepairAgent,
    block_unfaithful_answer,
    documents_for_state,
)
from cogdoc.agents.citation_validator import CitationValidatorAgent
from cogdoc.config.settings import get_settings
from cogdoc.observability.logger import log_event


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


def claim_audit_node(state: GraphState) -> dict:
    output = ClaimEvidenceVerifierAgent.audit(state)
    audit = output.get("claim_audit") or {}
    metrics = audit.get("metrics") or {}
    counts = audit.get("counts") or {}
    log_event(
        "claim_audit",
        "claim_audit_completed",
        state,
        status=audit.get("status", "not_run"),
        claim_count=counts.get("claim_count", 0),
        claim_support_rate=metrics.get("claim_support_rate"),
        citation_coverage=metrics.get("citation_coverage"),
        unsupported_claim_rate=metrics.get("unsupported_claim_rate"),
        verifier_error=output.get("claim_verifier_error", ""),
    )
    return output


def claim_audit_check(state: GraphState) -> str:
    audit = state.get("claim_audit") or {}
    status = str(audit.get("status") or "not_run")
    if status in {"not_run", "passed", "repaired"}:
        return END
    if status == "failed" and int(state.get("claim_repair_count", 0) or 0) < (
        get_settings().claim_verification_max_repair_attempts
    ):
        return "claim_repair_node"
    return "claim_block_node"


def claim_repair_node(state: GraphState) -> dict:
    output = ClaimRepairAgent.repair(state)
    log_event(
        "claim_audit",
        "claim_repair_completed",
        state,
        repair_count=output.get("claim_repair_count", 0),
        repair_error=output.get("claim_repair_error", ""),
    )
    return output


def claim_repair_check(state: GraphState) -> str:
    return (
        "claim_block_node"
        if state.get("claim_repair_error")
        else "claim_repair_citation_node"
    )


def claim_repair_citation_node(state: GraphState) -> dict:
    result = CitationValidatorAgent.validate_citations(
        str(state.get("answer") or ""),
        documents_for_state(state),
    )
    return {
        "claim_repair_citation_valid": bool(result.get("is_valid")),
        "claim_repair_critique": str(result.get("critique") or ""),
    }


def claim_repair_citation_check(state: GraphState) -> str:
    if state.get("claim_repair_citation_valid"):
        return "claim_audit_node"
    if int(state.get("claim_repair_count", 0) or 0) < (
        get_settings().claim_verification_max_repair_attempts
    ):
        return "claim_repair_node"
    return "claim_block_node"


def claim_block_node(state: GraphState) -> dict:
    output = block_unfaithful_answer(state)
    log_event(
        "claim_audit",
        "claim_audit_rejected",
        state,
        repair_count=state.get("claim_repair_count", 0),
        verifier_error=state.get("claim_verifier_error", ""),
    )
    return output


workflow = StateGraph(GraphState)

workflow.add_node("intent_router", RouterAgent.route_intent)
workflow.add_node("qa_subgraph", qa_subgraph_node)
workflow.add_node("summary_subgraph", summary_subgraph_node)
workflow.add_node("compare_subgraph", compare_subgraph_node)
workflow.add_node("unknown_node", unknown_node)
workflow.add_node("claim_audit_node", claim_audit_node)
workflow.add_node("claim_repair_node", claim_repair_node)
workflow.add_node("claim_repair_citation_node", claim_repair_citation_node)
workflow.add_node("claim_block_node", claim_block_node)

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
workflow.add_edge("qa_subgraph", "claim_audit_node")
workflow.add_edge("summary_subgraph", "claim_audit_node")
workflow.add_edge("compare_subgraph", "claim_audit_node")
workflow.add_conditional_edges(
    "claim_audit_node",
    claim_audit_check,
    {
        "claim_repair_node": "claim_repair_node",
        "claim_block_node": "claim_block_node",
        END: END,
    },
)
workflow.add_conditional_edges(
    "claim_repair_node",
    claim_repair_check,
    {
        "claim_repair_citation_node": "claim_repair_citation_node",
        "claim_block_node": "claim_block_node",
    },
)
workflow.add_conditional_edges(
    "claim_repair_citation_node",
    claim_repair_citation_check,
    {
        "claim_audit_node": "claim_audit_node",
        "claim_repair_node": "claim_repair_node",
        "claim_block_node": "claim_block_node",
    },
)
workflow.add_edge("claim_block_node", END)
workflow.add_edge("unknown_node", END)

app = workflow.compile()
