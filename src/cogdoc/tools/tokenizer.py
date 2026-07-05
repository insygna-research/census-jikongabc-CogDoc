import re
from typing import List
from cogdoc.tools.rust_core_loader import ensure_rust_core


# 分词下放到 rust_core，正常路径不再 import jieba。
_rust_core = ensure_rust_core("tokenize_mixed_text_native", "tokenize_corpus_native")

# 分词规则变化时 bump：进入增量复用门控，避免新旧文档混用不同分词的语料。
TOKENIZER_VERSION = "jieba_mixed_native_v2"

# 英文停用词表，必须与 rust_core 端逐词一致（Elasticsearch _english_ 默认表）。
_EN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)


# 分词mixed文本python。
def _tokenize_mixed_text_python(text: str) -> List[str]:
    # 纯 Python 参照实现，仅供分词对齐测试比对；运行链路统一走 native。
    import snowballstemmer

    stemmer = snowballstemmer.stemmer("english")
    text = text.lower()
    tokens = []

    for word in re.findall(r"[a-z0-9_\-\.]+", text):
        if len(word) <= 1:
            continue
        # 纯字母英文词去停用词后词干化；含数字/下划线/连字符的标识符、版本号原样保留。
        if word.isalpha():
            if word in _EN_STOPWORDS:
                continue
            stemmed = stemmer.stemWord(word)
            if len(stemmed) > 1:
                tokens.append(stemmed)
        else:
            tokens.append(word)

    chinese_pure = re.sub(r"[a-z0-9_\-\.]+", " ", text)
    import jieba

    for word in jieba.cut(chinese_pure, cut_all=False):
        word = word.strip()
        if word and len(word) > 1:
            tokens.append(word)

    return tokens


# 分词mixed文本。
def tokenize_mixed_text(text: str) -> List[str]:
    # 中英文混合检索统一分词规则，供 BM25 与摘要章节选择共用。
    return list(_rust_core.tokenize_mixed_text_native(text))


# 分词corpus。
def tokenize_corpus(texts: List[str]) -> List[List[str]]:
    # 批量入库走单次跨界 + rayon 并行，逐条结果与 tokenize_mixed_text 完全一致。
    return _rust_core.tokenize_corpus_native(list(texts))
