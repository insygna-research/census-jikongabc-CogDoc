import pytest
from cogdoc.config.settings import Settings, get_settings


# 清理 settings cache。
@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# 验证 settings defaults match current runtime contract 场景。
def test_settings_defaults_match_current_runtime_contract():
    settings = Settings()

    assert settings.cogdoc_doc_dir == "your_documents"
    assert settings.cogdoc_default_doc_id == "arch_blueprint_2026"
    assert settings.llm_model_name == "deepseek-chat"
    assert settings.ollama_model_name == "qwen2.5:7b"
    assert settings.qa_retrieval_top_k == 9
    assert settings.qa_rerank_top_n == 3
    assert settings.qa_abstain_enabled is True
    assert settings.qa_abstain_max_vector_distance == 0.86
    assert settings.qa_abstain_min_bm25_score == 10.0
    assert settings.qa_abstain_min_knowledge_score == 0.5
    assert settings.qa_evidence_verify_enabled is True
    assert settings.qa_evidence_verify_max_docs == 3
    assert settings.qa_evidence_verify_max_chars_per_doc == 1600
    assert settings.qa_evidence_verify_borderline_min_score == 0.75
    assert settings.hybrid_rrf_k == 60
    assert settings.memory_retrieval_enabled is True
    assert settings.memory_semantic_enabled is True
    assert settings.memory_retrieval_short_limit == 8
    assert settings.memory_retrieval_mid_limit == 4
    assert settings.memory_retrieval_recent_pin == 4
    assert settings.memory_semantic_include_short is False
    assert settings.memory_rrf_k == 60.0
    assert settings.memory_recency_weight == 1.0
    assert settings.memory_lexical_weight == 1.4
    assert settings.memory_semantic_weight == 1.6
    assert settings.memory_importance_weight == 0.8
    assert settings.memory_mid_priority_weight == 0.8
    assert settings.cogdoc_log_level == "INFO"
    assert settings.cogdoc_log_file == "logs/cogdoc.jsonl"
    assert settings.cogdoc_log_to_console is False
    assert settings.cogdoc_trace_enabled is True
    assert settings.cogdoc_trace_dir == "logs/traces"
    assert settings.cogdoc_ocr_enabled is False
    assert settings.cogdoc_ocr_provider == "tesseract"
    assert settings.cogdoc_ocr_binary == "tesseract"
    assert settings.cogdoc_ocr_languages == "eng+chi_sim"
    assert settings.cogdoc_ocr_dpi == 300
    assert settings.cogdoc_ocr_min_native_chars == 40
    assert settings.cogdoc_ocr_max_pages == 100
    assert settings.cogdoc_ocr_page_timeout_seconds == 30.0
    assert settings.cogdoc_ocr_required is False


# 验证 settings reads environment overrides 场景。
def test_settings_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("COGDOC_DOC_DIR", "papers")
    monkeypatch.setenv("LLM_MODEL_NAME", "custom-model")
    monkeypatch.setenv("QA_RETRIEVAL_TOP_K", "11")
    monkeypatch.setenv("QA_ABSTAIN_MAX_VECTOR_DISTANCE", "0.75")
    monkeypatch.setenv("COGDOC_MEMORY_SEMANTIC_ENABLED", "false")
    monkeypatch.setenv("COGDOC_MEMORY_RETRIEVAL_SHORT_LIMIT", "6")
    monkeypatch.setenv("COGDOC_MEMORY_RETRIEVAL_RECENT_PIN", "2")
    monkeypatch.setenv("COGDOC_MEMORY_SEMANTIC_WEIGHT", "2.5")
    monkeypatch.setenv("COGDOC_OCR_ENABLED", "true")
    monkeypatch.setenv("COGDOC_OCR_DPI", "240")
    monkeypatch.setenv("COGDOC_OCR_REQUIRED", "true")

    settings = get_settings()

    assert settings.cogdoc_doc_dir == "papers"
    assert settings.llm_model_name == "custom-model"
    assert settings.qa_retrieval_top_k == 11
    assert settings.qa_abstain_max_vector_distance == 0.75
    assert settings.memory_semantic_enabled is False
    assert settings.memory_retrieval_short_limit == 6
    assert settings.memory_retrieval_recent_pin == 2
    assert settings.memory_semantic_weight == 2.5
    assert settings.cogdoc_ocr_enabled is True
    assert settings.cogdoc_ocr_dpi == 240
    assert settings.cogdoc_ocr_required is True


# 验证节点可以独立选择后端和模型。
def test_settings_resolves_node_backend_and_model(monkeypatch):
    monkeypatch.setenv("LLM_EVIDENCE_VERIFIER_BACKEND", "local")
    monkeypatch.setenv("OLLAMA_EVIDENCE_VERIFIER_MODEL_NAME", "qwen-review:7b")

    settings = get_settings()

    is_local = settings.is_local_for_node("evidence_verifier", request_is_local=False)
    assert is_local is True
    assert (
        settings.model_name_for_node("evidence_verifier", is_local=is_local)
        == "qwen-review:7b"
    )
    assert settings.is_local_for_node("qa_generator", request_is_local=False) is False
    assert (
        settings.model_name_for_node("qa_generator", is_local=False)
        == settings.llm_model_name
    )


# 验证非法节点后端不会被静默解释为云端或本地。
def test_settings_rejects_invalid_node_backend(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_BACKEND", "somewhere")
    settings = get_settings()

    with pytest.raises(ValueError, match="无效节点后端"):
        settings.is_local_for_node("router", request_is_local=False)


# 验证 cuda thresholds are exposed as bytes 场景。
def test_cuda_thresholds_are_exposed_as_bytes(monkeypatch):
    monkeypatch.setenv("EMBEDDER_MIN_CUDA_FREE_MB", "123")

    settings = get_settings()

    assert settings.cuda_min_free_bytes("EMBEDDER_MIN_CUDA_FREE_MB") == (
        123 * 1024 * 1024
    )


# 验证 cuda thresholds reject unknown keys 场景。
def test_cuda_thresholds_reject_unknown_keys():
    settings = get_settings()

    with pytest.raises(ValueError, match="未知 CUDA 显存阈值配置"):
        settings.cuda_min_free_bytes("UNKNOWN_MIN_CUDA_FREE_MB")
