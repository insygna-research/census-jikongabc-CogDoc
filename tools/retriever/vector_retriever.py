import os
import chromadb
from typing import List
from graph.state import RetrievedDoc
from tools.embedder import Embedder
from tools.retriever.base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    def __init__(self, collection_id: str, persist_directory: str = "./data/chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection_name = f"col-{collection_id}"[:60]
        self._init_collection()

    def _init_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"embedding_model": Embedder.MODEL_NAME}
        )

        existing_meta = self.collection.metadata
        if (
            existing_meta
            and existing_meta.get("embedding_model") != Embedder.MODEL_NAME
        ):
            raise RuntimeError(
                f"Model Mismatch! Collection model: {existing_meta.get('embedding_model')}, "
                f"System model: {Embedder.MODEL_NAME}"
            )

    def exists(self) -> bool:
        return self.collection.count() > 0

    def clear(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            self._init_collection()
        except Exception:
            pass

    def index(self, chunks: List[RetrievedDoc]) -> None:
        if not chunks:
            return

        self.clear()

        # Chroma 主键直接使用稳定 chunk_id。
        embeddings = Embedder.embed_documents([c["text"] for c in chunks])

        ids, metadatas, texts = [], [], []
        for c in chunks:
            meta = c["meta"]
            chunk_id = str(meta["chunk_id"])
            ids.append(chunk_id)
            texts.append(c["text"])
            metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "source_sha256": meta["source_sha256"],
                    "local_chunk_index": meta["local_chunk_index"],
                    "chunk_index": meta["chunk_index"],
                    "source": meta["source"],
                    "page": meta["page"],
                    "page_start": meta["page_start"],
                    "page_end": meta["page_end"],
                    "origin": meta.get("origin", "file"),
                }
            )

        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        # 返回结构保持与 BM25Retriever 一致。
        results = self.collection.query(
            query_embeddings=[Embedder.embed_query(query)], n_results=top_k
        )
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        retrieved_docs: List[RetrievedDoc] = []
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        distances = (
            results["distances"][0] if "distances" in results else [0.0] * len(ids)
        )

        for i in range(len(ids)):
            meta_data = metas[i]
            chunk_id = meta_data.get("chunk_id")
            if not chunk_id:
                # 旧索引缺少 chunk_id 时必须重建，不能现场拼回。
                raise RuntimeError(
                    "Vector index is missing stable chunk_id metadata; rebuild the index."
                )
            source_sha256 = str(meta_data.get("source_sha256", ""))
            page_start = int(meta_data["page_start"])
            page_end = int(meta_data["page_end"])
            local_chunk_index = int(meta_data["local_chunk_index"])
            retrieved_docs.append(
                {
                    "text": docs[i],
                    "meta": {
                        "chunk_id": str(chunk_id),
                        "source_sha256": source_sha256,
                        "local_chunk_index": local_chunk_index,
                        "chunk_index": int(meta_data["chunk_index"]),
                        "source": str(meta_data["source"]),
                        "page": int(meta_data["page"]),
                        "page_start": page_start,
                        "page_end": page_end,
                        "origin": str(meta_data["origin"]),
                    },
                    "retrieval": {
                        "distance": float(distances[i]),
                        "search_channel": "vector",
                    },
                }
            )
        return retrieved_docs
