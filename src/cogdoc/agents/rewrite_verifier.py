import json
from typing import List, Tuple
from cogdoc.agents.conversation_memory import format_recent_chat_history
from cogdoc.tools.embedder import Embedder


# 关键词改写与自然句问题的相似度阈值需用真实数据继续标定。
DEFAULT_SIMILARITY_THRESHOLD = 0.5
# 相似度基准只取最近 1-2 轮，足以提供指代对象又不过度稀释当前问题语义。
SIMILARITY_BASELINE_HISTORY_LIMIT = 4


def _cosine_normalized(a: List[float], b: List[float]) -> float:
    # 输入向量已由 Embedder 做 L2 归一化，cosine 等于点积。
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def filter_rewrites_by_similarity(
    original_vec: List[float],
    rewrite_vecs: List[List[float]],
    rewrites: List[str],
    threshold: float,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    # 纯函数只负责按相似度过滤改写，便于脱离模型测试。
    kept: List[str] = []
    dropped: List[Tuple[str, float]] = []

    for idx, text in enumerate(rewrites):
        vec = rewrite_vecs[idx] if idx < len(rewrite_vecs) else []
        sim = _cosine_normalized(original_vec, vec)
        if sim >= threshold:
            kept.append(text)
        else:
            dropped.append((text, sim))

    return kept, dropped


class RewriteVerifyAgent:
    @staticmethod
    def verify_rewrites(state: dict) -> dict:
        # 全部改写被过滤时返回空列表，retrieve 节点会只用原问题检索。
        query = state.get("query", "")
        rewrites = state.get("rewritten_queries", [])
        threshold = state.get(
            "rewrite_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
        )

        if not query or not rewrites:
            return {"rewritten_queries": rewrites}

        # 有历史时用近期对话补全相似度基准，避免指代改写被裸省略句误杀。
        history_text = format_recent_chat_history(
            state.get("chat_history"), limit=SIMILARITY_BASELINE_HISTORY_LIMIT
        )
        baseline_text = f"{history_text}\n{query}" if history_text else query

        vectors = Embedder.embed_documents([baseline_text] + rewrites)
        original_vec = vectors[0] if vectors else []
        rewrite_vecs = vectors[1:] if len(vectors) > 1 else []

        kept, dropped = filter_rewrites_by_similarity(
            original_vec,
            rewrite_vecs,
            rewrites,
            threshold,
        )
        trace_payload = {
            "threshold": threshold,
            "kept": kept,
            "dropped": [
                {"query": text, "similarity": round(score, 4)}
                for text, score in dropped
            ],
        }
        return {
            "rewritten_queries": kept,
            "steps_trace": [
                {
                    "step_name": "verify_rewrite",
                    "input_summary": json.dumps(rewrites, ensure_ascii=False),
                    "output_summary": json.dumps(trace_payload, ensure_ascii=False),
                }
            ],
        }
