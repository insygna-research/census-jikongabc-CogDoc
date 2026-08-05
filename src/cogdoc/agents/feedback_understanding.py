from collections.abc import Mapping
from typing import Any, TypedDict

from cogdoc.tools.public_citation_ledger import validate_public_citation_ledger


class FeedbackTarget(TypedDict):
    trace_id: Any
    chunk_ids: list[str]
    sources: list[str]
    source_type: str


class FeedbackAnalysis(TypedDict):
    feedback_type: str
    sentiment: str
    target: FeedbackTarget
    extracted_claim: str
    recommended_action: str
    weight_delta: float
    confidence: float
    needs_review: bool


_NO_EVIDENCE_TERMS = (
    "无证据",
    "没有证据",
    "没证据",
    "没找到",
    "查不到",
    "no evidence",
)
_BAD_RETRIEVAL_TERMS = (
    "引用不相关",
    "来源不相关",
    "检索不相关",
    "证据不相关",
    "bad retrieval",
    "wrong source",
)


def _legacy_target_items(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in ("citations", "evidence"):
        for item in payload.get(field) or []:
            if not isinstance(item, Mapping):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            targets.append(
                {
                    "chunk_id": chunk_id,
                    "source": str(item.get("source") or "").strip(),
                    "source_type": str(item.get("source_type") or "document").strip(),
                }
            )
    return targets


def _ledger_target_items(
    payload: Mapping[str, Any],
) -> list[dict[str, str]] | None:
    if "citation_ledger" not in payload:
        return None
    raw_ledger = payload.get("citation_ledger")
    # 空账本保留旧反馈语义；trace 路径会先把 citations/evidence 原子替换成
    # 服务端可信快照，因此这里的 fallback 不会混入客户端伪造目标。
    if raw_ledger == []:
        return None
    if not isinstance(raw_ledger, list):
        return []
    validation = validate_public_citation_ledger(
        payload.get("answer"),
        raw_ledger,
        evidence=payload.get("evidence_ledger") or payload.get("evidence"),
        require_evidence=True,
    )
    return list(validation.targets) if validation.is_valid else []


# 从反馈载荷读取可归因分块。非空 ledger 必须完整通过闭集和 occurrence
# 校验；失败时安全地不归因，不再退回更宽泛的 evidence 并误罚分块。
def feedback_target_items(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    ledger_targets = _ledger_target_items(payload)
    if ledger_targets is not None:
        return ledger_targets
    return _legacy_target_items(payload)


# 读取文本字段。
def _text(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# 读取引用目标。
def _target(payload: Mapping[str, Any]) -> FeedbackTarget:
    chunk_ids, sources, source_types = [], [], set()
    for item in feedback_target_items(payload):
        chunk_id = item["chunk_id"]
        source = item["source"]
        source_type = item["source_type"]
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
        if source and source not in sources:
            sources.append(source)
        if source_type:
            source_types.add(source_type)
    if len(source_types) > 1:
        source_type = "mixed"
    elif source_types:
        source_type = next(iter(source_types))
    else:
        source_type = "none"
    return {
        "trace_id": payload.get("trace_id"),
        "chunk_ids": chunk_ids,
        "sources": sources,
        "source_type": source_type,
    }


# 判断反馈情绪。
def _sentiment(payload: Mapping[str, Any]) -> str:
    feedback = str(payload.get("feedback") or "")
    rating = payload.get("rating")
    if isinstance(rating, int):
        if rating >= 4:
            return "positive"
        if rating <= 2:
            return "negative"
    if feedback == "thumbs_up":
        return "positive"
    if feedback in {"thumbs_down", "correction"}:
        return "negative"
    return "neutral"


# 判断反馈类型。
def _feedback_type(payload: Mapping[str, Any], combined_text: str) -> str:
    explicit = payload.get("feedback_type")
    if explicit:
        return str(explicit)
    if _text(payload, "correction_text", "correction"):
        return "correction"
    lowered = combined_text.lower()
    if any(term in lowered for term in _NO_EVIDENCE_TERMS):
        return "no_evidence"
    if any(term in lowered for term in _BAD_RETRIEVAL_TERMS):
        return "bad_retrieval"
    if payload.get("feedback") == "thumbs_down":
        return "wrong_answer"
    return "other"


# 计算建议权重。
def _weight_delta(
    payload: Mapping[str, Any], feedback_type: str, sentiment: str
) -> float:
    rating = payload.get("rating")
    if isinstance(rating, int):
        return (rating - 3) * 0.12
    if feedback_type == "correction":
        return -0.55
    if feedback_type == "bad_retrieval":
        return -0.35
    if feedback_type == "no_evidence":
        return -0.2
    if sentiment == "positive":
        return 0.2
    if sentiment == "negative":
        return -0.25
    return 0.0


# 计算置信度。
def _confidence(
    payload: Mapping[str, Any],
    feedback_type: str,
    target: Mapping[str, Any],
    extracted_claim: str,
) -> float:
    confidence = 0.55
    if payload.get("feedback_type"):
        confidence = max(confidence, 0.76)
    if target.get("chunk_ids"):
        confidence = max(confidence, 0.68)
    if feedback_type == "correction" and extracted_claim:
        confidence = max(confidence, 0.88)
    if not payload.get("kb_id") or not payload.get("query"):
        confidence = min(confidence, 0.72)
    return confidence


# 分析反馈并输出结构化建议。
def analyze_feedback(payload: Mapping[str, Any]) -> FeedbackAnalysis:
    feedback_text = _text(payload, "feedback_text", "comment")
    correction_text = _text(payload, "correction_text", "correction")
    combined_text = " ".join(
        text
        for text in (feedback_text, correction_text, _text(payload, "answer"))
        if text
    )
    target = _target(payload)
    feedback_type = _feedback_type(payload, combined_text)
    sentiment = _sentiment(payload)
    confidence = _confidence(payload, feedback_type, target, correction_text)
    if correction_text and confidence >= 0.8:
        action = "create_pending_knowledge"
    elif target["chunk_ids"] and _weight_delta(payload, feedback_type, sentiment) != 0:
        action = "adjust_retrieval"
    else:
        action = "record_only"
    return {
        "feedback_type": feedback_type,
        "sentiment": sentiment,
        "target": target,
        "extracted_claim": correction_text,
        "recommended_action": action,
        "weight_delta": _weight_delta(payload, feedback_type, sentiment),
        "confidence": confidence,
        "needs_review": confidence < 0.8,
    }
