import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cogdoc.config.settings import Settings, get_settings


@dataclass(frozen=True)
class RetrievalSupport:
    supported: bool
    score: float
    reason: str
    signals: dict[str, float]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _threshold_ratio(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    if threshold <= 0:
        return 1.0
    return min(max(value / threshold, 0.0), 1.0)


def _distance_ratio(value: float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    if maximum <= 0:
        return 1.0 if value <= maximum else 0.0
    if value <= maximum:
        return 1.0
    return min(max(maximum / value, 0.0), 1.0)


# 以首个重排结果的向量/BM25 信号判断是否有足够证据；阈值是当前真实集的保守校准值。
def assess_retrieval_support(
    docs: Sequence[Mapping[str, Any]], settings: Settings | None = None
) -> RetrievalSupport:
    settings = settings or get_settings()
    if not docs:
        return RetrievalSupport(False, 0.0, "no_candidates", {})
    if not settings.qa_abstain_enabled:
        return RetrievalSupport(True, 1.0, "disabled", {})

    top_doc = docs[0]
    meta = top_doc.get("meta") if isinstance(top_doc.get("meta"), Mapping) else {}
    retrieval = (
        top_doc.get("retrieval")
        if isinstance(top_doc.get("retrieval"), Mapping)
        else {}
    )

    if meta.get("source_type") == "derived_knowledge":
        knowledge_score = _finite_float(
            retrieval.get("retrieval_score", retrieval.get("knowledge_score"))
        )
        if knowledge_score is None:
            return RetrievalSupport(True, 1.0, "signals_unavailable", {})
        supported = knowledge_score >= settings.qa_abstain_min_knowledge_score
        return RetrievalSupport(
            supported,
            _threshold_ratio(
                knowledge_score, settings.qa_abstain_min_knowledge_score
            ),
            "supported" if supported else "below_threshold",
            {"knowledge_score": knowledge_score},
        )

    distance = _finite_float(retrieval.get("distance"))
    bm25_score = _finite_float(retrieval.get("bm25_score"))
    signals = {
        key: value
        for key, value in {
            "distance": distance,
            "bm25_score": bm25_score,
        }.items()
        if value is not None
    }
    if not signals:
        return RetrievalSupport(True, 1.0, "signals_unavailable", {})

    semantic_supported = (
        distance is not None
        and distance <= settings.qa_abstain_max_vector_distance
    )
    lexical_supported = (
        bm25_score is not None
        and bm25_score >= settings.qa_abstain_min_bm25_score
    )
    supported = semantic_supported or lexical_supported
    score = max(
        _distance_ratio(distance, settings.qa_abstain_max_vector_distance),
        _threshold_ratio(bm25_score, settings.qa_abstain_min_bm25_score),
    )
    return RetrievalSupport(
        supported,
        score,
        "supported" if supported else "below_threshold",
        signals,
    )
