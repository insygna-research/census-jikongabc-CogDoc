from collections import Counter
from typing import Any

from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.graph.state import RetrievedDoc
from cogdoc.tools.tokenizer import tokenize_mixed_text


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

        scored: list[tuple[float, dict[str, Any]]] = []
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
            scored.append((coverage + density, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._row_to_doc(row, score, rank)
            for rank, (score, row) in enumerate(scored[:top_k])
        ]

    # 将派生知识记录转换成检索文档。
    def _row_to_doc(self, row: dict[str, Any], score: float, rank: int) -> RetrievedDoc:
        knowledge_id = str(row.get("knowledge_id") or "")
        related_source = str(row.get("related_source") or "")
        chunk_ids = [str(item) for item in row.get("related_chunk_ids") or []]
        meta = {
            "chunk_id": f"knowledge:{knowledge_id}",
            "knowledge_id": knowledge_id,
            "source_sha256": str(row.get("related_source_sha256") or ""),
            "local_chunk_index": rank,
            "chunk_index": rank,
            "source": f"knowledge:{knowledge_id}",
            "page": 0,
            "page_start": 0,
            "page_end": 0,
            "origin": str(row.get("origin") or "manual_entry"),
            "source_type": "derived_knowledge",
            "status": str(row.get("status") or ""),
            "certainty": str(row.get("certainty") or ""),
            "related_source": related_source,
            "related_chunk_ids": chunk_ids,
        }
        if row.get("source_note"):
            meta["context"] = str(row["source_note"])
        return {
            "text": str(row.get("text") or ""),
            "meta": meta,
            "retrieval": {
                "knowledge_score": score,
                "search_channel": "derived_knowledge",
            },
        }
