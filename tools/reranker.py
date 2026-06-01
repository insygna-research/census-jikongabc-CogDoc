import torch
import copy
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from graph.state import RetrievedDoc

class BGEReranker:
    _tokenizer = None  # Tokenizer单例
    _model = None  # 模型单例
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # Reranker模型名称
    MAX_LENGTH = 512  # 模型单次处理的最大Token长度保护

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch, "backends") and torch.backends.mps.is_available()
        else "cpu"
    )  # 自动选择运行设备

    @classmethod
    def _get_resources(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)  # 加载分词器
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(cls.MODEL_NAME)  # 加载精排模型
            cls._model.to(cls.device)  # 模型迁移到目标设备
            cls._model.eval()  # 切入推理模式
        return cls._tokenizer, cls._model

    @classmethod
    def rerank(cls, query: str, docs: List[RetrievedDoc], top_n: int = 3) -> List[RetrievedDoc]:
        if not docs:
            return []  # 无候选文档直接返回
        if len(docs) <= 1:
            return copy.deepcopy(docs)[:top_n]  # 单文档无需精排

        tokenizer, model = cls._get_resources()  # 获取单例资源

        pairs = [[query, doc["text"]] for doc in docs]  # 构造[Query, Chunk]配对

        with torch.no_grad():  # 关闭梯度计算
            inputs = tokenizer(
                pairs,
                padding = True,  # 自动补齐长度
                truncation = True,  # 超长自动截断
                max_length = cls.MAX_LENGTH,  # 最大长度限制
                return_tensors = "pt"  # 返回PyTorch张量
            ).to(cls.device)  # 输入迁移到目标设备

            outputs = model(**inputs, return_dict = True)  # 执行前向推理
            scores = outputs.logits.view(-1).float().cpu().numpy()  # 提取相关性得分

        ranked_docs: List[RetrievedDoc] = []
        for idx, score in enumerate(scores):
            doc_copy = copy.deepcopy(docs[idx])  # 深拷贝避免污染原数据

            retrieval_meta = doc_copy.setdefault("retrieval", {})  # 获取或创建检索元数据
            retrieval_meta["rerank_score"] = float(score)  # 写入精排得分

            ranked_docs.append(doc_copy)

        ranked_docs.sort(key = lambda x: x["retrieval"]["rerank_score"], reverse = True)  # 按精排得分降序排序

        return ranked_docs[:top_n]  # 返回TopN结果