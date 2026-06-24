from abc import ABC, abstractmethod
from typing import List
from graph.state import RetrievedDoc


# 封装 BaseRetriever 的状态与行为。
class BaseRetriever(ABC):
    # 抽象基类，定义检索器必须实现的接口
    @abstractmethod
    def exists(self) -> bool:
        pass

    # 清理 clear 相关逻辑。
    @abstractmethod
    def clear(self) -> None:
        pass

    # 写入索引 index 相关逻辑。
    @abstractmethod
    def index(self, chunks: List[RetrievedDoc]) -> None:
        pass

    # 检索 search 相关逻辑。
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        pass


# 封装 NullWriteError 的状态与行为。
class NullWriteError(RuntimeError):
    # 写方法被路由到 NullRetriever：调用方未持有活跃代引擎（Phase 3 之前的误用保护）。
    pass


# 封装 NullRetriever 的状态与行为。
class NullRetriever(BaseRetriever):
    # 空检索器：无活跃代或嵌入模型不匹配时占位。 读方法返回安全空值；写方法显式 raise，避免静默 no-op 掩盖误用。
    def exists(self) -> bool:
        return False

    # 清理 clear 相关逻辑。
    def clear(self) -> None:
        pass

    # 检索 search 相关逻辑。
    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        return []

    # 统计 count 相关逻辑。
    def count(self) -> int:
        return 0

    # 切分 chunk ids 相关逻辑。
    def chunk_ids(self) -> set:
        return set()

    # 获取最大 max chunk index 相关逻辑。
    def max_chunk_index(self) -> int:
        return -1

    # 列出 list sources 相关逻辑。
    def list_sources(self) -> List[str]:
        return []

    # 加载 load source chunks 相关逻辑。
    def load_source_chunks(self, source: str) -> List[RetrievedDoc]:
        return []

    # 删除 delete by source 相关逻辑。
    def delete_by_source(self, sources) -> None:
        pass

    # 写入索引 index 相关逻辑。
    def index(self, chunks: List[RetrievedDoc]) -> None:
        raise NullWriteError(
            "index() called on NullRetriever: no active generation engine"
        )

    # 添加 add documents 相关逻辑。
    def add_documents(self, chunks: List[RetrievedDoc]) -> None:
        raise NullWriteError(
            "add_documents() called on NullRetriever: no active generation engine"
        )

    # 增量写入 upsert documents 相关逻辑。
    def upsert_documents(self, new_chunks: List[RetrievedDoc], removed_sources) -> None:
        raise NullWriteError(
            "upsert_documents() called on NullRetriever: no active generation engine"
        )
