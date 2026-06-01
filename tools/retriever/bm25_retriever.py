import os
import pickle
import copy
import jieba
import re
from typing import List
from rank_bm25 import BM25Okapi
from graph.state import RetrievedDoc
from tools.retriever.base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, collection_id: str, persist_directory: str = "./data/bm25_db"):
        os.makedirs(persist_directory, exist_ok = True)  # 确保倒排索引落盘目录存在
        self.db_path = os.path.join(persist_directory, f"bm25_{collection_id}.pkl")

        self._init_collection()

    def _init_collection(self) -> None:
        self.bm25: BM25Okapi = None
        self.doc_registry: List[RetrievedDoc] = []  # 保存原始Chunk数据
        self.tokenized_corpus: List[List[str]] = []  # 保存分词后的语料库

        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    state = pickle.load(f)  # 加载持久化索引状态
                    self.doc_registry = state.get("doc_registry", [])
                    self.tokenized_corpus = state.get("tokenized_corpus", [])
                if self.tokenized_corpus and self.doc_registry:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)  # 重建BM25索引
            except Exception:
                self.bm25, self.doc_registry, self.tokenized_corpus = None, [], []
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()  # 统一英文大小写
        tokens = []
        
        english_words = re.findall(r'[a-z0-9_\-\.]+', text)  # 提取英文单词及技术标识
        tokens.extend([w for w in english_words if len(w) > 1])  # 过滤单字符噪声
        
        chinese_pure = re.sub(r'[a-z0-9_\-\.]+', ' ', text)  # 剥离英文部分后交给jieba处理
        jieba_words = jieba.cut(chinese_pure, cut_all = False)
        for w in jieba_words:
            w_strip = w.strip()
            if w_strip and len(w_strip) > 1:
                tokens.append(w_strip)  # 加入中文分词结果

        return tokens
    
    def exists(self) -> bool: 
        return self.bm25 is not None and len(self.doc_registry) > 0  # 检查倒排索引是否存在
    
    def clear(self) -> None:
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)  # 删除持久化索引文件
            self._init_collection()  # 重置内存状态
        except Exception:
            pass

    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks: return  # 空数据直接跳过

        self.clear()  # 重建前清空旧索引

        for c in chunks:
            meta = c["meta"]
            clean_doc: RetrievedDoc = {
                "text": c["text"],
                "meta": {
                    "chunk_index": int(meta["chunk_index"]),
                    "source": str(meta["source"]),
                    "page": int(meta["page"]),
                    "page_start": int(meta["page_start"]),
                    "page_end": int(meta["page_end"]),
                    "origin": str(meta.get("origin", "file"))
                }
            }
            self.doc_registry.append(clean_doc)  # 保存清洗干净后的原始Chunk
            self.tokenized_corpus.append(self._tokenize(c["text"]))  # 保存分词结果
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)  # 构建BM25倒排索引
        
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "doc_registry": self.doc_registry, 
                "tokenized_corpus": self.tokenized_corpus
            }, f)  # 持久化索引状态

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        if not self.bm25 or not self.doc_registry:
            return []  # 索引为空直接返回
            
        tokenized_query = self._tokenize(query)  # 查询语句分词
        scores = self.bm25.get_scores(tokenized_query)  # 计算Query与所有文档的BM25得分
        
        top_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:top_k]  # 提取TopK结果
        
        retrieved_docs: List[RetrievedDoc] = []
        for idx in top_indices:
            if scores[idx] <= 0: 
                continue  # 过滤无关文档
                
            doc_copy = copy.deepcopy(self.doc_registry[idx])  # 避免修改原始数据
            meta_data = doc_copy["meta"]
            
            retrieved_docs.append({
                "text": doc_copy["text"],  # 命中的文本块
                "meta": {
                    "chunk_index": int(meta_data["chunk_index"]),
                    "source": str(meta_data["source"]),
                    "page": int(meta_data["page"]),
                    "page_start": int(meta_data["page_start"]),
                    "page_end": int(meta_data["page_end"]),
                    "origin": str(meta_data["origin"])
                },
                "retrieval": {
                    "bm25_score": float(scores[idx]),  # BM25相关性得分
                    "search_channel": "bm25"
                }
            })
        return retrieved_docs  # 返回结构化检索结果