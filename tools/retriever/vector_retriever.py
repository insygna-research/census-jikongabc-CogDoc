import os
import chromadb
from typing import List
from graph.state import RetrievedDoc
from tools.embedder import Embedder
from tools.retriever.base_retriever import BaseRetriever

class VectorRetriever(BaseRetriever):
    def __init__(self, collection_id: str, persist_directory: str = "./data/chroma_db"):
        os.makedirs(persist_directory, exist_ok = True)  # 确保向量数据库落盘目录存在
        self.client = chromadb.PersistentClient(path = persist_directory)  # 初始化持久化Chroma客户端
        
        self.collection_name = f"col-{collection_id}"[:60]
        self._init_collection()
        
    def _init_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,  # 每个文档独立collection避免污染
            metadata = {"embedding_model": Embedder.MODEL_NAME}  # 记录当前embedding模型信息
        )

        existing_meta = self.collection.metadata  # 读取已存在collection的元信息
        if existing_meta and existing_meta.get("embedding_model") != Embedder.MODEL_NAME:
            raise RuntimeError(  # 防止不同embedding模型混用导致检索失真
                f"Model Mismatch! Collection model: {existing_meta.get('embedding_model')}, "
                f"System model: {Embedder.MODEL_NAME}"
            )

    def exists(self) -> bool: 
        return self.collection.count() > 0 # 检查向量库中是否存在索引数据
    
    def clear(self) -> None:
        try:
            self.client.delete_collection(name = self.collection_name) # 销毁向量数据库
            self._init_collection() 
        except Exception:
            pass

    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks: return  # 空数据直接跳过避免无意义计算

        self.clear()

        embeddings = Embedder.embed_documents([c["text"] for c in chunks])  # 批量生成向量
        
        ids, metadatas, texts = [], [], []  # 分别存储id、元数据、文本内容
        for c in chunks:
            meta = c["meta"]
            ids.append(f"{meta['source']}_{meta['chunk_index']}")  # 每个chunk生成唯一id
            texts.append(c["text"])  # 原始文本
            metadatas.append({  # 存储检索辅助信息
                "chunk_index": meta["chunk_index"],
                "source": meta["source"],
                "page": meta["page"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "origin": meta.get("origin", "file")
            })
            
        self.collection.upsert(ids = ids, embeddings = embeddings, documents = texts, metadatas = metadatas)  # 幂等写入向量库

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        results = self.collection.query(  # 执行向量相似度检索
            query_embeddings = [Embedder.embed_query(query)],
            n_results = top_k
        )
        if not results or not results["documents"] or not results["documents"][0]:
            return []  # 无结果直接返回空列表
            
        retrieved_docs: List[RetrievedDoc] = []
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        for i in range(len(ids)):
            meta_data = metas[i]
            retrieved_docs.append({
                "text": docs[i],  # 命中的文本块
                "meta": {
                    "chunk_index": int(meta_data["chunk_index"]),
                    "source": str(meta_data["source"]),
                    "page": int(meta_data["page"]),
                    "page_start": int(meta_data["page_start"]),
                    "page_end": int(meta_data["page_end"]),
                    "origin": str(meta_data["origin"])
                },
                "retrieval": {
                    "distance": float(distances[i]),  # 向量距离（越小越相关）
                    "search_channel": "vector"
                }
            })
        return retrieved_docs  # 返回结构化检索结果