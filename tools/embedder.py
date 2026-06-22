import torch
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    # 单例模型，整个程序只加载一次
    _model = None
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    # 自动选择运行设备：NVIDIA GPU -> Apple M系列 -> CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch, "backends") and torch.backends.mps.is_available()
            else "cpu"
        )
    )

    # 加载模型
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(
                "BAAI/bge-small-zh-v1.5", device=cls.device
            )
        return cls._model

    # 问题向量化
    @classmethod
    def embed_query(cls, text: str) -> List[float]:
        return cls.get_model().encode([text], normalize_embeddings=True)[0].tolist()

    # 文档向量化
    @classmethod
    def embed_documents(cls, texts: List[str]) -> List[List[float]]:
        return (
            cls.get_model()
            .encode(
                texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            .tolist()
        )
