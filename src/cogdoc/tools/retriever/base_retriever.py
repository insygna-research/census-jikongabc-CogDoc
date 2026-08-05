from abc import ABC, abstractmethod
from typing import List
from cogdoc.graph.state import RetrievedDoc
from cogdoc.tools.retriever.scope import RetrievalScope


# 抽象基类，定义检索器必须实现的接口。
class BaseRetriever(ABC):
    # 抽象基类，定义检索器必须实现的接口。
    @abstractmethod
    def exists(self) -> bool:
        pass

    # 清理。
    @abstractmethod
    def clear(self) -> None:
        pass

    # 写入索引。
    @abstractmethod
    def index(self, chunks: List[RetrievedDoc]) -> None:
        pass

    # 检索。
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        scope: RetrievalScope | None = None,
    ) -> List[RetrievedDoc]:
        pass


# 写方法被路由到 NullRetriever：调用方未持有活跃代引擎（Phase 3 之前的误用保护）。
class NullWriteError(RuntimeError):
    pass


# 空检索器：无活跃代或嵌入模型不匹配时占位；读方法返回安全空值；写方法显式 raise，避免静默 no-op 掩盖误用。
class NullRetriever(BaseRetriever):
    # 空检索器：无活跃代或嵌入模型不匹配时占位；读方法返回安全空值；写方法显式 raise，避免静默 no-op 掩盖误用。
    def exists(self) -> bool:
        return False

    # 清理。
    def clear(self) -> None:
        pass

    # 检索。
    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        scope: RetrievalScope | None = None,
    ) -> List[RetrievedDoc]:
        _ = scope
        return []

    # 统计数量。
    def count(self) -> int:
        return 0

    # 切分 ids。
    def chunk_ids(self) -> set:
        return set()

    # 完成 max分块索引 处理。
    def max_chunk_index(self) -> int:
        return -1

    # 列出 sources。
    def list_sources(self) -> List[str]:
        return []

    # 加载 source chunks。
    def load_source_chunks(self, source: str) -> List[RetrievedDoc]:
        return []

    # 删除 by source。
    def delete_by_source(self, sources) -> None:
        pass

    # 写入索引。
    def index(self, chunks: List[RetrievedDoc]) -> None:
        raise NullWriteError(
            "index() called on NullRetriever: no active generation engine"
        )

    # 添加 documents。
    def add_documents(self, chunks: List[RetrievedDoc]) -> None:
        raise NullWriteError(
            "add_documents() called on NullRetriever: no active generation engine"
        )

    # 增量写入documents。
    def upsert_documents(self, new_chunks: List[RetrievedDoc], removed_sources) -> None:
        raise NullWriteError(
            "upsert_documents() called on NullRetriever: no active generation engine"
        )
