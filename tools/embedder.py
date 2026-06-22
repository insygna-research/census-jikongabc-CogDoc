from sentence_transformers import SentenceTransformer
from typing import List
from tools.device import required_cuda_free_bytes, resolve_device


class Embedder:
    # 单例模型，整个程序只加载一次
    _model = None
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    # bge-small-zh-v1.5 权重约 0.4G + 批量活化余量，空闲低于此值回落 CPU，避免 CUDA OOM。
    REQUIRED_CUDA_FREE_BYTES = required_cuda_free_bytes("EMBEDDER_MIN_CUDA_FREE_MB", 800)

    device = "cpu"  # 实际设备在首次加载时按空闲显存动态判定，默认安全回落 CPU

    # 加载模型
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls.device = resolve_device(cls.REQUIRED_CUDA_FREE_BYTES)
            cls._model = SentenceTransformer(cls.MODEL_NAME, device=cls.device)
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
