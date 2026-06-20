import re
from typing import List

import jieba


def tokenize_mixed_text(text: str) -> List[str]:
    # 中英文混合检索统一分词规则，供 BM25 与摘要章节选择共用。
    text = text.lower()
    tokens = []

    english_words = re.findall(r'[a-z0-9_\-\.]+', text)
    tokens.extend([word for word in english_words if len(word) > 1])

    chinese_pure = re.sub(r'[a-z0-9_\-\.]+', ' ', text)
    for word in jieba.cut(chinese_pure, cut_all = False):
        word = word.strip()
        if word and len(word) > 1:
            tokens.append(word)

    return tokens
