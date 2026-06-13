import copy
from typing import List
from graph.state import RetrievedDoc
from tools.retriever.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: BaseRetriever, bm25_retriever: BaseRetriever, k: int = 60):
        self.vector_retriever = vector_retriever  # 向量检索器
        self.bm25_retriever = bm25_retriever      # BM25检索器
        self.k = k  # RRF平滑系数

    @staticmethod
    def _doc_key(doc: RetrievedDoc) -> str:
        meta = doc["meta"]
        return f"{meta.get('source', '')}::{meta.get('chunk_index', '')}"

    def exists(self) -> bool:
        return self.vector_retriever.exists() and self.bm25_retriever.exists()  # 两路索引均存在才视为可用
    
    def clear(self) -> None:
        self.vector_retriever.clear()  # 清空向量索引
        self.bm25_retriever.clear()  # 清空BM25索引
    
    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks: return
        self.vector_retriever.index(chunks)  # 构建向量索引
        self.bm25_retriever.index(chunks)  # 构建BM25索引

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        recall_top_k = top_k * 3  # 扩大召回池提高融合质量

        vector_results = self.vector_retriever.search(query, top_k = recall_top_k)  # 向量召回
        bm25_results = self.bm25_retriever.search(query, top_k = recall_top_k)  # BM25召回

        rrf_scores = {}  # source::chunk_index -> RRF得分
        doc_registry = {}  # source::chunk_index -> 文档结果

        for rank, doc in enumerate(vector_results):
            doc_key = self._doc_key(doc)
            doc_registry[doc_key] = copy.deepcopy(doc)  # 注册向量召回结果
            
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (self.k + (rank + 1))    # 累加向量检索RRF得分

        for rank, doc in enumerate(bm25_results):
            doc_key = self._doc_key(doc)

            if doc_key in doc_registry:
                doc_registry[doc_key]["retrieval"]["bm25_score"] = doc["retrieval"]["bm25_score"] # 为已有结果补充BM25得分
            else:
                doc_registry[doc_key] = copy.deepcopy(doc)    # 注册仅被BM25命中的结果
                
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (self.k + (rank + 1))    # 累加BM25检索RRF得分

        sorted_doc_keys = sorted(rrf_scores.keys(), key = lambda x: rrf_scores[x], reverse = True)  # 按RRF得分降序排序

        combined_results: List[RetrievedDoc] = []
        for doc_key in sorted_doc_keys[:top_k]:
            target_doc = doc_registry[doc_key]

            
            target_doc["retrieval"]["rrf_score"] = float(rrf_scores[doc_key])     # 写入最终融合得分
            
            combined_results.append(target_doc)
            
        return combined_results
