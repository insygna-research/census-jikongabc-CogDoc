import re
from typing import List

from tools.rust_core_loader import ensure_rust_core


# 分词下放到 rust_core，正常路径不再 import jieba。
_rust_core = ensure_rust_core("tokenize_mixed_text_native")


def _tokenize_mixed_text_python(text: str) -> List[str]:
    # 纯 Python 参照实现，仅供分词对齐测试比对；运行链路统一走 native。
    text = text.lower()
    tokens = []

    english_words = re.findall(r'[a-z0-9_\-\.]+', text)
    tokens.extend([word for word in english_words if len(word) > 1])

    chinese_pure = re.sub(r'[a-z0-9_\-\.]+', ' ', text)
    import jieba

    for word in jieba.cut(chinese_pure, cut_all = False):
        word = word.strip()
        if word and len(word) > 1:
            tokens.append(word)

    return tokens


def tokenize_mixed_text(text: str) -> List[str]:
    # 中英文混合检索统一分词规则，供 BM25 与摘要章节选择共用。
    return list(_rust_core.tokenize_mixed_text_native(text))
