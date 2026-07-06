import torch
import copy
import threading
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from cogdoc.graph.state import RetrievedDoc
from cogdoc.tools.device import (
    model_inference_semaphore,
    required_cuda_free_bytes,
    resolve_device,
)


# 返回跳过 CPU 重排的候选文档。
def skipped_cpu_rerank_docs(
    docs: List[RetrievedDoc], top_n: int, reason: str = "cpu_disabled"
) -> List[RetrievedDoc]:
    selected = copy.deepcopy(docs[:top_n])
    for doc in selected:
        doc.setdefault("retrieval", {})["rerank_skipped_reason"] = reason
    return selected


# 定义 BGEReranker 数据结构。
class BGEReranker:
    _tokenizer = None  # Tokenizer单例
    _model = None  # 模型单例
    _models = {}  # 按设备缓存模型单例
    _lock = threading.RLock()  # 保护懒加载与设备缓存
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # Reranker模型名称
    MAX_LENGTH = 512  # 模型单次处理的最大Token长度保护

    # bge-reranker-v2-m3 权重约 2.3G + 推理活化余量，空闲低于此值回落 CPU，避免 CUDA OOM。
    REQUIRED_CUDA_FREE_BYTES = required_cuda_free_bytes("RERANKER_MIN_CUDA_FREE_MB")

    device = None  # None=未显式指定，加载时按空闲显存自动判定；set_device 后固定

    # 设置 device。
    @classmethod
    def set_device(cls, device: str) -> None:
        with cls._lock:
            if device != cls.device:
                cls._model = cls._models.get(
                    device
                )  # 切换当前设备视图，不清空其它设备缓存
                cls.device = device

    # 完成 default设备 处理。
    @classmethod
    def default_device(cls) -> str:
        with cls._lock:
            current_device = cls.device
            model_loaded = cls._model is not None
            if current_device is None and "cuda" in cls._models:
                current_device = (
                    "cuda"  # 已加载 GPU 模型时继续复用，避免被自身显存占用误判。
                )
                model_loaded = True
            return resolve_device(
                cls.REQUIRED_CUDA_FREE_BYTES, current_device, model_loaded
            )

    # 获取 resources。
    @classmethod
    def _get_resources(cls, device: str | None = None):
        # Tokenizer 与模型按进程级单例懒加载。
        with cls._lock:
            explicit_device = device is not None
            target_device = device or cls.device
            if target_device is None:
                target_device = (
                    cls.default_device()
                )  # 直连调用也按显存选设备，不退化成 CPU
            if cls._tokenizer is None:
                cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            model = cls._models.get(target_device)
            if model is None:
                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_NAME
                )
                model.to(target_device)
                model.eval()
                cls._models[target_device] = model
            if not explicit_device:
                cls.device = target_device
                cls._model = model
            elif cls.device == target_device:
                cls._model = model
            return cls._tokenizer, model, target_device

    # 完成 预热流程预热流程 处理。
    @classmethod
    def warm_up(cls) -> None:
        cls._get_resources()

    # 重排。
    @classmethod
    def rerank(
        cls,
        query: str,
        docs: List[RetrievedDoc],
        top_n: int = 3,
        device: str | None = None,
    ) -> List[RetrievedDoc]:
        # 精排只修改深拷贝结果，避免污染召回缓存。
        if not docs:
            return []  # 无候选文档直接返回
        if len(docs) <= 1:
            return copy.deepcopy(docs)[:top_n]  # 单文档无需精排

        with model_inference_semaphore("reranker"):
            tokenizer, model, target_device = cls._get_resources(device)  # 获取单例资源

            pairs = [[query, doc["text"]] for doc in docs]  # 构造[Query, Chunk]配对

            with torch.no_grad():  # 关闭梯度计算
                inputs = tokenizer(
                    pairs,
                    padding=True,  # 自动补齐长度
                    truncation=True,  # 超长自动截断
                    max_length=cls.MAX_LENGTH,  # 最大长度限制
                    return_tensors="pt",  # 返回PyTorch张量
                ).to(target_device)  # 输入迁移到目标设备

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
