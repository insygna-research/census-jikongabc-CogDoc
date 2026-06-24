import math
from sentence_transformers import SentenceTransformer
from typing import List
from tools.device import required_cuda_free_bytes, resolve_device


# 封装 Embedder 的状态与行为。
class Embedder:
    # 单例模型，整个程序只加载一次
    _model = None
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    # 固定到 HF commit SHA：分支会移动（远端更新权重后契约不变但模型已变），SHA 才能真正钉死权重版本。 升级权重必须改此 SHA，连带 EMBEDDING_CONTRACT_VERSION 变化使旧向量失效、强制全量重建。
    MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
    EMBEDDING_DIM = 512  # 输出维度，编码后强校验
    NORMALIZE = True  # 归一化方式，影响距离度量，变更即不可复用旧向量

    # 嵌入兼容契约：模型名/revision/维度/归一化任一变化都使旧向量不可复用，强制全量重建。
    EMBEDDING_CONTRACT_VERSION = (
        f"{MODEL_NAME}@{MODEL_REVISION}|dim={EMBEDDING_DIM}|norm={NORMALIZE}"
    )

    # bge-small-zh-v1.5 权重约 0.4G + 批量活化余量，空闲低于此值回落 CPU，避免 CUDA OOM。
    REQUIRED_CUDA_FREE_BYTES = required_cuda_free_bytes("EMBEDDER_MIN_CUDA_FREE_MB")

    device = "cpu"  # 实际设备在首次加载时按空闲显存动态判定，默认安全回落 CPU

    # 加载模型：pin revision，使契约声明的权重版本约束实际加载。
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls.device = resolve_device(cls.REQUIRED_CUDA_FREE_BYTES)
            cls._model = SentenceTransformer(
                cls.MODEL_NAME, device=cls.device, revision=cls.MODEL_REVISION
            )
        return cls._model

    # 校验 validate embeddings 相关逻辑。
    @classmethod
    def validate_embeddings(cls, embeddings) -> None:
        # 统一校验：逐个向量维度等于契约值，且数值全为有限值（拒绝 NaN/Inf）。 编码后与跨代复用写入前共用，绝不让不兼容或污染的向量进入索引。
        for vector in embeddings:
            if len(vector) != cls.EMBEDDING_DIM:
                raise ValueError(
                    f"embedding dim {len(vector)} != contract {cls.EMBEDDING_DIM}: "
                    "嵌入契约与实际模型不符"
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding contains non-finite value (NaN/Inf)")

    # 问题向量化
    @classmethod
    def embed_query(cls, text: str) -> List[float]:
        vector = (
            cls.get_model()
            .encode([text], normalize_embeddings=cls.NORMALIZE)[0]
            .tolist()
        )
        cls.validate_embeddings([vector])
        return vector

    # 文档向量化
    @classmethod
    def embed_documents(cls, texts: List[str]) -> List[List[float]]:
        vectors = (
            cls.get_model()
            .encode(
                texts,
                batch_size=64,
                normalize_embeddings=cls.NORMALIZE,
                show_progress_bar=False,
            )
            .tolist()
        )
        cls.validate_embeddings(vectors)
        return vectors
