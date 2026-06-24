import os
import pickle
import copy
from threading import RLock
from typing import List
from config.settings import get_settings
from graph.state import RetrievedDoc
from tools.document_loader import list_sources, load_source_chunks
from tools.tokenizer import tokenize_mixed_text
from tools.rust_core_loader import ensure_rust_core
from tools.retriever.base_retriever import BaseRetriever


# BM25 计算下放 rust_core，持久化仍只存语料和 chunk 注册表。
_rust_core = ensure_rust_core("Bm25Index")


# 封装 BM25Retriever 的状态与行为。
class BM25Retriever(BaseRetriever):
    # 初始化实例状态。
    def __init__(self, collection_id: str, persist_directory: str = None):
        persist_directory = persist_directory or get_settings().bm25_persist_dir
        os.makedirs(persist_directory, exist_ok=True)
        self.db_path = os.path.join(persist_directory, f"bm25_{collection_id}.pkl")

        # 保护 (bm25, doc_registry, tokenized_corpus) 三元组的原子替换与一致快照读取。
        self._lock = RLock()
        self._init_collection()

    # 构建 build index 相关逻辑。
    def _build_index(self, tokenized_corpus: List[List[str]]):
        # 原生 BM25 与 rank_bm25.BM25Okapi 逐位对齐（k1=1.5, b=0.75, epsilon=0.25 默认）。
        return _rust_core.Bm25Index(tokenized_corpus)

    # 处理 init collection 相关逻辑。
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

    # 分词 tokenize 相关逻辑。
    def _tokenize(self, text: str) -> List[str]:
        return tokenize_mixed_text(text)

    # 预热 warm up 相关逻辑。
    def warm_up(self) -> None:
        self._tokenize("知识库 检索 warmup")

    # 检查是否存在 exists 相关逻辑。
    def exists(self) -> bool:
        return self.bm25 is not None and len(self.doc_registry) > 0

    # 统计 count 相关逻辑。
    def count(self) -> int:
        return len(self.doc_registry)

    # 切分 chunk ids 相关逻辑。
    def chunk_ids(self) -> set:
        # 一致性校验用：registry 内全部 chunk_id。
        return {str(d["meta"]["chunk_id"]) for d in self.doc_registry}

    # 获取最大 max chunk index 相关逻辑。
    def max_chunk_index(self) -> int:
        # 增量续号用：现存最大展示编号，空索引返回 -1。
        return max(
            (int(d["meta"]["chunk_index"]) for d in self.doc_registry), default=-1
        )

    # 清理 clear 相关逻辑。
    def clear(self) -> None:
        # 容忍「文件本就不存在」，但清理后必须确为空；删除失败会让 _init 重载旧数据，借此识破。
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            pass
        with self._lock:
            self._init_collection()
            if self.doc_registry:
                raise RuntimeError("bm25 index was not cleared")

    # 处理 clean doc 相关逻辑。
    @staticmethod
    def _clean_doc(c: RetrievedDoc) -> RetrievedDoc:
        # 只留 chunk 身份元数据，去掉检索期临时字段。
        meta = c["meta"]
        return {
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
                "origin": str(meta.get("origin", "file")),
            },
        }

    # 处理 persist 相关逻辑。
    def _persist(self, registry, corpus) -> None:
        # 原子写：先写临时 pickle 再 os.replace，进程中断不会留下半截文件。
        tmp_path = f"{self.db_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"doc_registry": registry, "tokenized_corpus": corpus}, f)
        os.replace(tmp_path, self.db_path)

    # 处理 swap in 相关逻辑。
    def _swap_in(self, registry, corpus) -> None:
        # 先落盘再切内存：持久化失败时内存与磁盘不会错位（内存仍是旧的一致状态）。
        new_bm25 = self._build_index(corpus) if corpus else None
        if corpus:
            self._persist(registry, corpus)
        else:
            try:
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
            except Exception:
                pass
        # 再在局部建好 Bm25Index 后加锁原子替换三元组：并发 search 永远看到 index 与 registry 匹配的快照。
        with self._lock:
            self.bm25 = new_bm25
            self.doc_registry = registry
            self.tokenized_corpus = corpus

    # 写入索引 index 相关逻辑。
    def index(self, chunks: List[RetrievedDoc]) -> None:
        # 全量重建：在局部构建后原子替换。增量入库走 upsert_documents。
        if not chunks:
            return
        registry = [self._clean_doc(c) for c in chunks]
        corpus = [self._tokenize(c["text"]) for c in chunks]
        self._swap_in(registry, corpus)

    # 增量写入 upsert documents 相关逻辑。
    def upsert_documents(self, new_chunks: List[RetrievedDoc], removed_sources) -> None:
        # 增量：保留未变文档的语料，丢弃删/改文档（按文件名），追加新 chunk，重建 Bm25Index。 BM25 分数依赖全局 IDF/avgdl，故仍整体重建 Bm25Index，但跳过未变文档的重新分词。
        drop = {str(s) for s in removed_sources if s}
        with self._lock:
            current = list(zip(self.doc_registry, self.tokenized_corpus))
        keep_registry, keep_corpus = [], []
        for doc, tokens in current:
            if doc["meta"]["source"] in drop:
                continue
            keep_registry.append(doc)
            keep_corpus.append(tokens)
        for c in new_chunks:
            keep_registry.append(self._clean_doc(c))
            keep_corpus.append(self._tokenize(c["text"]))
        self._swap_in(keep_registry, keep_corpus)

    # 处理 export registry 相关逻辑。
    def export_registry(self) -> List[RetrievedDoc]:
        # 跨代复用权威：返回 registry 深拷贝作文本/metadata 真值来源，向量按 chunk_id 关联。
        with self._lock:
            return copy.deepcopy(self.doc_registry)

    # 列出 list sources 相关逻辑。
    def list_sources(self) -> List[str]:
        return list_sources(self.doc_registry)

    # 加载 load source chunks 相关逻辑。
    def load_source_chunks(self, source: str) -> List[RetrievedDoc]:
        return load_source_chunks(self.doc_registry, source)

    # 检索 search 相关逻辑。
    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        # 一致快照：bm25 与 registry 必须取自同一次原子替换，否则下标会错配到新 registry。
        with self._lock:
            bm25 = self.bm25
            registry = self.doc_registry
        if not bm25 or not registry:
            return []

        # BM25 分数只用于本路排序，融合分数由 RRF 写入。
        tokenized_query = self._tokenize(query)
        ranked = bm25.score_topk(tokenized_query, top_k)

        retrieved_docs: List[RetrievedDoc] = []
        for idx, score in ranked:
            if score <= 0:
                continue

            doc_copy = copy.deepcopy(registry[idx])
            meta_data = doc_copy["meta"]
            chunk_id = meta_data.get("chunk_id")
            if not chunk_id:
                # 旧索引缺少 chunk_id 时必须重建，不能现场拼回。
                raise RuntimeError(
                    "BM25 index is missing stable chunk_id metadata; rebuild the index."
                )
            source_sha256 = str(meta_data.get("source_sha256", ""))
            page_start = int(meta_data["page_start"])
            page_end = int(meta_data["page_end"])
            local_chunk_index = int(meta_data["local_chunk_index"])

            retrieved_docs.append(
                {
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
                        "origin": str(meta_data["origin"]),
                    },
                    "retrieval": {"bm25_score": float(score), "search_channel": "bm25"},
                }
            )
        return retrieved_docs
