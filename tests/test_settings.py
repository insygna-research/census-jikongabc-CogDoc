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
    assert settings.hybrid_rrf_k == 60
    assert settings.cogdoc_log_level == "INFO"
    assert settings.cogdoc_log_file == "logs/cogdoc.jsonl"
    assert settings.cogdoc_log_to_console is False
    assert settings.cogdoc_trace_enabled is True
    assert settings.cogdoc_trace_dir == "logs/traces"


# 验证 settings reads environment overrides 场景。
def test_settings_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("COGDOC_DOC_DIR", "papers")
    monkeypatch.setenv("LLM_MODEL_NAME", "custom-model")
    monkeypatch.setenv("QA_RETRIEVAL_TOP_K", "11")

    settings = get_settings()

    assert settings.cogdoc_doc_dir == "papers"
    assert settings.llm_model_name == "custom-model"
    assert settings.qa_retrieval_top_k == 11


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
