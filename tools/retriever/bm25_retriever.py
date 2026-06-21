import os
import pickle
import copy
from typing import List
from graph.state import RetrievedDoc
from tools.document_loader import list_sources, load_source_chunks
from tools.tokenizer import tokenize_mixed_text
from tools.rust_core_loader import ensure_rust_core
from tools.retriever.base_retriever import BaseRetriever


# BM25 计算下放 rust_core，持久化仍只存语料和 chunk 注册表。
_rust_core = ensure_rust_core("Bm25Index")


class BM25Retriever(BaseRetriever):
    def __init__(self, collection_id: str, persist_directory: str = "./data/bm25_db"):
        os.makedirs(persist_directory, exist_ok = True)
        self.db_path = os.path.join(persist_directory, f"bm25_{collection_id}.pkl")

        self._init_collection()

    def _build_index(self, tokenized_corpus: List[List[str]]):
        # 原生 BM25 与 rank_bm25.BM25Okapi 逐位对齐（k1=1.5, b=0.75, epsilon=0.25 默认）。
        return _rust_core.Bm25Index(tokenized_corpus)

    def _init_collection(self) -> None:
        self.bm25 = None
        self.doc_registry: List[RetrievedDoc] = []
        self.tokenized_corpus: List[List[str]] = []

        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    state = pickle.load(f)
                    self.doc_registry = state.get("doc_registry", [])
                    self.tokenized_corpus = state.get("tokenized_corpus", [])
                if self.tokenized_corpus and self.doc_registry:
                    self.bm25 = self._build_index(self.tokenized_corpus)
            except Exception:
                self.bm25, self.doc_registry, self.tokenized_corpus = None, [], []
    
    def _tokenize(self, text: str) -> List[str]:
        return tokenize_mixed_text(text)

    def warm_up(self) -> None:
        self._tokenize("知识库 检索 warmup")
    
    def exists(self) -> bool: 
        return self.bm25 is not None and len(self.doc_registry) > 0
    
    def clear(self) -> None:
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self._init_collection()
        except Exception:
            pass

    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks: return

        self.clear()

        # BM25 持久化完整 chunk 身份元数据。
        for c in chunks:
            meta = c["meta"]
            clean_doc: RetrievedDoc = {
                "text": c["text"],
                "meta": {
                    "chunk_id": str(meta["chunk_id"]),
                    "source_sha256": str(meta["source_sha256"]),
                    "local_chunk_index": int(meta["local_chunk_index"]),
                    "chunk_index": int(meta["chunk_index"]),
                    "source": str(meta["source"]),
                    "page": int(meta["page"]),
                    "page_start": int(meta["page_start"]),
                    "page_end": int(meta["page_end"]),
                    "origin": str(meta.get("origin", "file"))
                }
            }
            self.doc_registry.append(clean_doc)
            self.tokenized_corpus.append(self._tokenize(c["text"]))

        self.bm25 = self._build_index(self.tokenized_corpus)

        with open(self.db_path, "wb") as f:
            pickle.dump({
                "doc_registry": self.doc_registry, 
                "tokenized_corpus": self.tokenized_corpus
            }, f)

    def list_sources(self) -> List[str]:
        return list_sources(self.doc_registry)

    def load_source_chunks(self, source: str) -> List[RetrievedDoc]:
        return load_source_chunks(self.doc_registry, source)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        if not self.bm25 or not self.doc_registry:
            return []
            
        # BM25 分数只用于本路排序，融合分数由 RRF 写入。
        tokenized_query = self._tokenize(query)
        ranked = self.bm25.score_topk(tokenized_query, top_k)

        retrieved_docs: List[RetrievedDoc] = []
        for idx, score in ranked:
            if score <= 0:
                continue

            doc_copy = copy.deepcopy(self.doc_registry[idx])
            meta_data = doc_copy["meta"]
            chunk_id = meta_data.get("chunk_id")
            if not chunk_id:
                # 旧索引缺少 chunk_id 时必须重建，不能现场拼回。
                raise RuntimeError("BM25 index is missing stable chunk_id metadata; rebuild the index.")
            source_sha256 = str(meta_data.get("source_sha256", ""))
            page_start = int(meta_data["page_start"])
            page_end = int(meta_data["page_end"])
            local_chunk_index = int(meta_data["local_chunk_index"])
            
            retrieved_docs.append({
                "text": doc_copy["text"],
                "meta": {
                    "chunk_id": str(chunk_id),
                    "source_sha256": source_sha256,
                    "local_chunk_index": local_chunk_index,
                    "chunk_index": int(meta_data["chunk_index"]),
                    "source": str(meta_data["source"]),
                    "page": int(meta_data["page"]),
                    "page_start": page_start,
                    "page_end": page_end,
                    "origin": str(meta_data["origin"])
                },
                "retrieval": {
                    "bm25_score": float(score),
                    "search_channel": "bm25"
                }
            })
        return retrieved_docs
