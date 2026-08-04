from collections.abc import Mapping
from typing import Any


SAFE_RETRIEVAL_METADATA_KEYS = {
    "search_channel",
    "knowledge_score",
    "retrieval_score",
    "rerank_score",
    "feedback_boost",
    "match_coverage",
    "match_density",
    "matched_terms",
    "query_term_count",
    "knowledge_term_count",
    "status_filter",
    "rewrite_query",
    "query_fusion_score",
    "query_hit_count",
    "matched_queries",
    "matched_channels",
    "matched_requirement_ids",
    "best_query_rank",
    "original_query_hit",
    "retrieval_round",
}


# 提取安全检索元数据。
def safe_retrieval_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in SAFE_RETRIEVAL_METADATA_KEYS if key in value}
