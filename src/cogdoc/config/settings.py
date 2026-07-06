from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 仓库根（.env / 默认 data 根的锚点）。本文件位于 src/cogdoc/config/，上溯 3 层到仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# 项目路径。
class Settings(BaseSettings):
    cogdoc_doc_dir: str = Field(
        default="your_documents", validation_alias="COGDOC_DOC_DIR"
    )
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

    # 访问控制：API key 逗号分隔，留空则鉴权关闭（本地/测试透明）。
    cogdoc_api_keys: str = Field(default="", validation_alias="COGDOC_API_KEYS")
    # 限流令牌桶：每分钟补充速率 + 突发容量；容量<=0 关闭限流。
    rate_limit_per_minute: int = Field(
        default=120, validation_alias="RATE_LIMIT_PER_MINUTE"
    )
    rate_limit_burst: int = Field(default=120, validation_alias="RATE_LIMIT_BURST")

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
    qa_rerank_max_candidates: int = Field(
        default=12, validation_alias="QA_RERANK_MAX_CANDIDATES"
    )
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

    # 完成 data目录 处理。
    @property
    def data_dir(self) -> Path:
        return Path(self.cogdoc_data_dir)

    # 完成 chroma持久化流程目录 处理。
    @property
    def chroma_persist_dir(self) -> str:
        return str(self.data_dir / "chroma_db")

    # 完成 bm25持久化流程目录 处理。
    @property
    def bm25_persist_dir(self) -> str:
        return str(self.data_dir / "bm25_db")

    # 构造目录。
    @property
    def manifest_dir(self) -> str:
        return str(self.data_dir / "manifests")

    # 完成 知识库根目录 处理。
    @property
    def kb_root(self) -> str:
        return str(self.data_dir / "kb")

    # 完成 知识库注册表路径 处理。
    @property
    def kb_registry_path(self) -> str:
        return str(self.data_dir / "kb" / "registry.json")

    # 完成 知识库来源目录 处理。
    def kb_source_dir(self, kb_id: str) -> str:
        # 每个知识库一个源 PDF 目录；上传落在这里，构建时硬链接快照到 generation 工作区。
        return str(self.data_dir / "kb" / kb_id / "sources")

    # 完成 知识库状态路径 处理。
    def kb_state_path(self, kb_id: str) -> str:
        # 事务化索引的提交指针：active_generation/epoch/generations 表，与每文档 manifest 分离。
        return str(self.data_dir / "kb" / kb_id / "state.json")

    # 完成 知识库索引代目录 处理。
    def kb_generation_dir(self, kb_id: str, generation_id: str) -> str:
        # 单个 generation 的工作区：源文件硬链接快照 + 该代 manifest 等内部产物。
        return str(self.data_dir / "kb" / kb_id / "generations" / generation_id)

    # 完成 知识库collectionid 处理。
    def kb_collection_id(self, kb_id: str, gen_id: str) -> str:
        # Chroma/BM25 集合 ID：sha256(kb_id)[:8]-{gen_id}，固定 22 字符，不受 kb_id 长度影响。
        import hashlib

        return f"{hashlib.sha256(kb_id.encode()).hexdigest()[:8]}-{gen_id}"

    # 完成 API密钥set 处理。
    @property
    def api_key_set(self) -> set[str]:
        # 解析逗号分隔的 key 列表，去空白与空项；空集合表示鉴权关闭。
        return {k.strip() for k in self.cogdoc_api_keys.split(",") if k.strip()}

    # 完成 状态db路径 处理。
    @property
    def state_db_path(self) -> str:
        # 会话与入库任务的 SQLite 落盘，进程重启不丢对话与任务状态。
        return str(self.data_dir / "state.db")

    # 完成 反馈log路径 处理。
    @property
    def feedback_log_path(self) -> str:
        return str(self.data_dir / "feedback" / "feedback.jsonl")

    # 完成 坏样本用例列表路径 处理。
    @property
    def bad_cases_path(self) -> str:
        # 点踩/纠错自动归集到此，喂离线质量评测 harness。
        return str(self.data_dir / "feedback" / "bad_cases.jsonl")

    # 返回根目录。
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    # 完成 CUDAminfreebytes 处理。
    def cuda_min_free_bytes(self, setting_name: str) -> int:
        mb_by_name = {
            "EMBEDDER_MIN_CUDA_FREE_MB": self.embedder_min_cuda_free_mb,
            "RERANKER_MIN_CUDA_FREE_MB": self.reranker_min_cuda_free_mb,
        }
        if setting_name not in mb_by_name:
            raise ValueError(f"未知 CUDA 显存阈值配置: {setting_name}")
        return int(mb_by_name[setting_name]) * 1024 * 1024


# 返回设置。
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
