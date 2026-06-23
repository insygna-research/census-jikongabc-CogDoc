import torch
import copy
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from graph.state import RetrievedDoc
from tools.device import required_cuda_free_bytes, resolve_device


class BGEReranker:
    _tokenizer = None  # Tokenizer单例
    _model = None  # 模型单例
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # Reranker模型名称
    MAX_LENGTH = 512  # 模型单次处理的最大Token长度保护

    # bge-reranker-v2-m3 权重约 2.3G + 推理活化余量，空闲低于此值回落 CPU，避免 CUDA OOM。
    REQUIRED_CUDA_FREE_BYTES = required_cuda_free_bytes("RERANKER_MIN_CUDA_FREE_MB")

    device = None  # None=未显式指定，加载时按空闲显存自动判定；set_device 后固定

    @classmethod
    def set_device(cls, device: str) -> None:
        if device != cls.device:
            cls._model = None  # 设备变更时清除单例，下次调用重新加载到新设备
            cls.device = device

    @classmethod
    def default_device(cls) -> str:
        return resolve_device(
            cls.REQUIRED_CUDA_FREE_BYTES, cls.device, cls._model is not None
        )

    @classmethod
    def _get_resources(cls):
        # Tokenizer 与模型按进程级单例懒加载。
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
        if cls._model is None:
            if cls.device is None:
                cls.device = cls.default_device()  # 直连调用也按显存选设备，不退化成 CPU
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME
            )
            cls._model.to(cls.device)
            cls._model.eval()
        return cls._tokenizer, cls._model

    @classmethod
    def warm_up(cls) -> None:
        cls._get_resources()

    @classmethod
    def rerank(
        cls, query: str, docs: List[RetrievedDoc], top_n: int = 3
    ) -> List[RetrievedDoc]:
        # 精排只修改深拷贝结果，避免污染召回缓存。
        if not docs:
            return []  # 无候选文档直接返回
        if len(docs) <= 1:
            return copy.deepcopy(docs)[:top_n]  # 单文档无需精排

        tokenizer, model = cls._get_resources()  # 获取单例资源

        pairs = [[query, doc["text"]] for doc in docs]  # 构造[Query, Chunk]配对

        with torch.no_grad():  # 关闭梯度计算
            inputs = tokenizer(
                pairs,
                padding=True,  # 自动补齐长度
                truncation=True,  # 超长自动截断
                max_length=cls.MAX_LENGTH,  # 最大长度限制
                return_tensors="pt",  # 返回PyTorch张量
            ).to(cls.device)  # 输入迁移到目标设备

            outputs = model(**inputs, return_dict=True)  # 执行前向推理
            scores = outputs.logits.view(-1).float().cpu().numpy()  # 提取相关性得分

        ranked_docs: List[RetrievedDoc] = []
        for idx, score in enumerate(scores):
            doc_copy = copy.deepcopy(docs[idx])  # 深拷贝避免污染原数据

            retrieval_meta = doc_copy.setdefault(
                "retrieval", {}
            )  # 获取或创建检索元数据
            retrieval_meta["rerank_score"] = float(score)  # 写入精排得分

            ranked_docs.append(doc_copy)

        ranked_docs.sort(
            key=lambda x: x["retrieval"]["rerank_score"], reverse=True
        )  # 按精排得分降序排序

        return ranked_docs[:top_n]  # 返回TopN结果
