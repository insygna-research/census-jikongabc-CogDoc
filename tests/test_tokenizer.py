import pytest
from cogdoc.tools.tokenizer import (
    _tokenize_mixed_text_python,
    tokenize_corpus,
    tokenize_mixed_text,
)


def test_tokenize_mixed_text_keeps_word_level_chinese_tokens():
    tokens = tokenize_mixed_text("模型 方法 架构")

    assert "模型" in tokens
    assert "方法" in tokens
    assert "架构" in tokens
    assert "模" not in tokens


def test_tokenize_mixed_text_keeps_latin_tokens():
    tokens = tokenize_mixed_text("BGE-reranker v2.5 chunk_id")

    # 含连字符/数字/下划线的标识符、版本号原样保留，不被词干化破坏。
    assert "bge-reranker" in tokens
    assert "v2.5" in tokens
    assert "chunk_id" in tokens


def test_tokenize_english_stems_and_drops_stopwords():
    tokens = tokenize_mixed_text("The models are retrieving documents")

    # 停用词被剔除。
    assert "the" not in tokens
    assert "are" not in tokens
    # 纯字母词被 Snowball 词干化归一。
    assert "model" in tokens
    assert "retriev" in tokens
    assert "document" in tokens


# jieba 仅作 native 分词的对齐参照，其 import pkg_resources 的弃用告警与运行链路无关。
@pytest.mark.filterwarnings("ignore:pkg_resources is deprecated")
@pytest.mark.parametrize(
    "text",
    [
        "模型 方法 架构",
        "BGE-reranker v2.5 chunk_id",
        "对比 a.pdf 和 b.pdf 的方法和实验结论",
        "本文提出了一种基于Transformer的检索增强生成框架，用于多文档摘要任务。",
        "技术方案设计规范与数据治理体系建设",
        "RRF融合 BM25与向量召回 top_k=9 的排序稳定性",
        "知识库 检索 warmup",
        "大语言模型在中文信息抽取与命名实体识别上的表现",
        "Retrieval Augmented Generation models for the documents",
        "These rankings compare retrieved chunks and their citations",
        "running runner runs studies studied organization",
        "混合检索 hybrid retrieval 把 BM25 and vector results 融合",
        "",
        "   ",
    ],
)
def test_native_tokenizer_matches_python_reference(text):
    assert tokenize_mixed_text(text) == _tokenize_mixed_text_python(text)


def test_tokenize_corpus_matches_per_text_tokenization():
    texts = [
        "模型 方法 架构",
        "BGE-reranker v2.5 chunk_id",
        "对比 a.pdf 和 b.pdf 的方法和实验结论",
        "",
        "   ",
    ]
    assert tokenize_corpus(texts) == [tokenize_mixed_text(t) for t in texts]


def test_tokenize_corpus_empty_input():
    assert tokenize_corpus([]) == []
