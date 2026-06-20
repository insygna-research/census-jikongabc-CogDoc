from tools.tokenizer import tokenize_mixed_text


def test_tokenize_mixed_text_keeps_word_level_chinese_tokens():
    # 中文分词输出词级 token，不输出单字噪声。
    tokens = tokenize_mixed_text("模型 方法 架构")

    assert "模型" in tokens
    assert "方法" in tokens
    assert "架构" in tokens
    assert "模" not in tokens


def test_tokenize_mixed_text_keeps_latin_tokens():
    # 英文、数字和常见技术符号按 BM25 检索规则保留。
    tokens = tokenize_mixed_text("BGE-reranker v2.5 chunk_id")

    assert "bge-reranker" in tokens
    assert "v2.5" in tokens
    assert "chunk_id" in tokens
