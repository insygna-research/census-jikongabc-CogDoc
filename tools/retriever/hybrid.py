from typing import List
from config.settings import get_settings
from graph.state import RetrievedDoc
from tools.retriever.base_retriever import BaseRetriever
from tools.rust_core_loader import ensure_rust_core


# Hybrid 检索依赖 Rust RRF 融合函数。
rust_core = ensure_rust_core("rrf_fusion_native")


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        k: int = None,
    ):
        self.vector_retriever = vector_retriever  # 向量检索器
        self.bm25_retriever = bm25_retriever  # BM25检索器
        self.k = k if k is not None else get_settings().hybrid_rrf_k  # RRF平滑系数

    def exists(self) -> bool:
        return (
            self.vector_retriever.exists() and self.bm25_retriever.exists()
        )  # 两路索引均存在才视为可用

    def clear(self) -> None:
        self.vector_retriever.clear()  # 清空向量索引
        self.bm25_retriever.clear()  # 清空BM25索引

    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks:
            return
        self.vector_retriever.index(chunks)  # 构建向量索引
        self.bm25_retriever.index(chunks)  # 构建BM25索引

    def list_sources(self) -> List[str]:
        # 单文档摘要从 BM25 registry 读取完整文档列表。
        return self.bm25_retriever.list_sources()

    def load_source_chunks(self, source: str) -> List[RetrievedDoc]:
        # 单文档摘要加载指定 source 的全部 chunk。
        return self.bm25_retriever.load_source_chunks(source)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        # 两路召回后交给 native RRF 做去重融合。
        recall_top_k = top_k * 3  # 扩大召回池提高融合质量

        vector_results = self.vector_retriever.search(
            query, top_k=recall_top_k
        )  # 向量召回
        bm25_results = self.bm25_retriever.search(query, top_k=recall_top_k)  # BM25召回

        return rust_core.rrf_fusion_native(
            vector_results, bm25_results, float(self.k), top_k
        )
