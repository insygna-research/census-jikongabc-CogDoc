from abc import ABC, abstractmethod
from typing import List
from graph.state import RetrievedDoc


class BaseRetriever(ABC):
    # 抽象基类，定义检索器必须实现的接口
    @abstractmethod
    def exists(self) -> bool:
        pass  # 检查检索库是否存在可用索引

    @abstractmethod
    def clear(self) -> None:
        pass  # 清空检索库中的全部索引数据

    @abstractmethod
    def index(self, chunks: List[RetrievedDoc]) -> None:
        pass  # 建立索引入库

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        pass  # 文档检索接口
