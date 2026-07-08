from collections import Counter
from typing import Any

from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.graph.state import RetrievedDoc
from cogdoc.tools.tokenizer import tokenize_mixed_text


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# 派生知识召回器，只读取已审核知识，不触碰原始文档索引。
class DerivedKnowledgeRetriever:
    def __init__(self, store: DerivedKnowledgeStore | None = None):
        self.store = store or DerivedKnowledgeStore()

    # 搜索已审核派生知识。
    def search(self, kb_id: str, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        rows = self.store.list(kb_id=kb_id, status="approved")
        if not rows:
            return []
        query_terms = Counter(tokenize_mixed_text(query))
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            text = str(row.get("text") or "")
            source_note = str(row.get("source_note") or "")
            terms = Counter(tokenize_mixed_text(f"{text}\n{source_note}"))
            if not terms:
                continue
            overlap = sum(
                min(count, terms.get(term, 0)) for term, count in query_terms.items()
            )
            if overlap <= 0:
                continue
            coverage = overlap / max(sum(query_terms.values()), 1)
            density = overlap / max(sum(terms.values()), 1)
            matched_terms = sorted(
                term for term in query_terms if terms.get(term, 0) > 0
            )
            explanation = {
                "matched_terms": matched_terms[:12],
                "query_term_count": sum(query_terms.values()),
                "knowledge_term_count": sum(terms.values()),
                "match_coverage": round(coverage, 6),
                "match_density": round(density, 6),
            }
            scored.append((coverage + density, row, explanation))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._row_to_doc(row, score, rank, explanation)
            for rank, (score, row, explanation) in enumerate(scored[:top_k])
        ]

    # 将派生知识记录转换成检索文档。
    def _row_to_doc(
        self,
        row: dict[str, Any],
        score: float,
        rank: int,
        explanation: dict[str, Any],
    ) -> RetrievedDoc:
        knowledge_id = str(row.get("knowledge_id") or "")
        related_source = str(row.get("related_source") or "")
        chunk_ids = [str(item) for item in row.get("related_chunk_ids") or []]
        page_start = _int_or_zero(row.get("related_page_start"))
        page_end = _int_or_zero(row.get("related_page_end"))
        meta = {
            "chunk_id": f"knowledge:{knowledge_id}",
            "knowledge_id": knowledge_id,
            "source_sha256": str(row.get("related_source_sha256") or ""),
            "local_chunk_index": rank,
            "chunk_index": rank,
            "source": f"knowledge:{knowledge_id}",
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "origin": str(row.get("origin") or "manual_entry"),
            "source_type": "derived_knowledge",
            "status": str(row.get("status") or ""),
            "certainty": str(row.get("certainty") or ""),
            "related_source": related_source,
            "related_chunk_ids": chunk_ids,
            "related_page_start": page_start,
            "related_page_end": page_end,
            "related_chunk_text_hash": str(row.get("related_chunk_text_hash") or ""),
            "related_anchor_text": str(row.get("related_anchor_text") or ""),
        }
        if row.get("source_note"):
            meta["context"] = str(row["source_note"])
        return {
            "text": str(row.get("text") or ""),
            "meta": meta,
            "retrieval": {
                **explanation,
                "knowledge_score": score,
                "retrieval_score": score,
                "search_channel": "derived_knowledge",
                "status_filter": "approved",
            },
        }
