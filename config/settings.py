from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # 项目路径。
    cogdoc_doc_dir: str = Field(default="测试论文", validation_alias="COGDOC_DOC_DIR")
    cogdoc_data_dir: str = Field(default="./data", validation_alias="COGDOC_DATA_DIR")
    cogdoc_default_doc_id: str = Field(
        default="arch_blueprint_2026", validation_alias="COGDOC_DEFAULT_DOC_ID"
    )
    cogdoc_log_level: str = Field(default="INFO", validation_alias="COGDOC_LOG_LEVEL")
    cogdoc_log_file: str = Field(
        default="logs/cogdoc.jsonl", validation_alias="COGDOC_LOG_FILE"
    )
    cogdoc_log_to_console: bool = Field(
        default=False, validation_alias="COGDOC_LOG_TO_CONSOLE"
    )
    cogdoc_trace_enabled: bool = Field(
        default=True, validation_alias="COGDOC_TRACE_ENABLED"
    )
    cogdoc_trace_dir: str = Field(
        default="logs/traces", validation_alias="COGDOC_TRACE_DIR"
    )

    # 云端 OpenAI 兼容后端。
    llm_model_name: str = Field(
        default="deepseek-chat", validation_alias="LLM_MODEL_NAME"
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1", validation_alias="LLM_BASE_URL"
    )
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_structured_output_method: str = Field(
        default="auto", validation_alias="LLM_STRUCTURED_OUTPUT_METHOD"
    )
    # 云端韧性：单次调用硬超时与传输层重试次数，运维可按后端 SLA 调。
    llm_timeout_seconds: float = Field(
        default=90.0, validation_alias="LLM_TIMEOUT_SECONDS"
    )
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")

    # 本地 Ollama OpenAI 兼容后端。
    ollama_model_name: str = Field(
        default="qwen2.5:7b", validation_alias="OLLAMA_MODEL_NAME"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_api_key: str = Field(default="ollama", validation_alias="OLLAMA_API_KEY")
    # 本地 Ollama 比云端慢得多：超时更长、重试更少（本地失败重试意义不大）。
    ollama_timeout_seconds: float = Field(
        default=180.0, validation_alias="OLLAMA_TIMEOUT_SECONDS"
    )
    ollama_max_retries: int = Field(default=1, validation_alias="OLLAMA_MAX_RETRIES")

    # 检索与生成控制。
    qa_retrieval_top_k: int = Field(default=9, validation_alias="QA_RETRIEVAL_TOP_K")
    qa_rerank_top_n: int = Field(default=3, validation_alias="QA_RERANK_TOP_N")
    hybrid_rrf_k: int = Field(default=60, validation_alias="HYBRID_RRF_K")
    cloud_section_max_workers: int = Field(
        default=6, validation_alias="CLOUD_SECTION_MAX_WORKERS"
    )

    # 模型设备阈值，单位 MB。
    embedder_min_cuda_free_mb: int = Field(
        default=800, validation_alias="EMBEDDER_MIN_CUDA_FREE_MB"
    )
    reranker_min_cuda_free_mb: int = Field(
        default=2800, validation_alias="RERANKER_MIN_CUDA_FREE_MB"
    )

    # 入库上传单文件大小上限，最小毒丸防护。
    max_upload_mb: int = Field(default=50, validation_alias="COGDOC_MAX_UPLOAD_MB")

    # 评测默认路径。
    eval_set_path: str = Field(
        default="eval/retrieval_eval.jsonl", validation_alias="COGDOC_EVAL_SET"
    )
    eval_example_set_path: str = Field(
        default="eval/retrieval_eval.example.jsonl",
        validation_alias="COGDOC_EVAL_EXAMPLE_SET",
    )
    quality_eval_set_path: str = Field(
        default="eval/quality_eval.jsonl", validation_alias="COGDOC_QUALITY_EVAL_SET"
    )
    quality_eval_example_set_path: str = Field(
        default="eval/quality_eval.example.jsonl",
        validation_alias="COGDOC_QUALITY_EVAL_EXAMPLE_SET",
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @property
    def data_dir(self) -> Path:
        return Path(self.cogdoc_data_dir)

    @property
    def chroma_persist_dir(self) -> str:
        return str(self.data_dir / "chroma_db")

    @property
    def bm25_persist_dir(self) -> str:
        return str(self.data_dir / "bm25_db")

    @property
    def manifest_dir(self) -> str:
        return str(self.data_dir / "manifests")

    @property
    def kb_root(self) -> str:
        return str(self.data_dir / "kb")

    @property
    def kb_registry_path(self) -> str:
        return str(self.data_dir / "kb" / "registry.json")

    def kb_source_dir(self, kb_id: str) -> str:
        # 每个知识库一个源 PDF 目录；chroma/bm25/manifest 已按 collection_id=kb_id 隔离。
        return str(self.data_dir / "kb" / kb_id / "sources")

    @property
    def state_db_path(self) -> str:
        # 会话与入库任务的 SQLite 落盘，进程重启不丢对话与任务状态。
        return str(self.data_dir / "state.db")

    @property
    def feedback_log_path(self) -> str:
        return str(self.data_dir / "feedback" / "feedback.jsonl")

    @property
    def bad_cases_path(self) -> str:
        # 点踩/纠错自动归集到此，喂离线质量评测 harness。
        return str(self.data_dir / "feedback" / "bad_cases.jsonl")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def cuda_min_free_bytes(self, setting_name: str) -> int:
        mb_by_name = {
            "EMBEDDER_MIN_CUDA_FREE_MB": self.embedder_min_cuda_free_mb,
            "RERANKER_MIN_CUDA_FREE_MB": self.reranker_min_cuda_free_mb,
        }
        if setting_name not in mb_by_name:
            raise ValueError(f"未知 CUDA 显存阈值配置: {setting_name}")
        return int(mb_by_name[setting_name]) * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
