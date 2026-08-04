from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 仓库根目录，作为环境文件与默认数据目录的锚点。
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
    cogdoc_webhook_url: str = Field(default="", validation_alias="COGDOC_WEBHOOK_URL")
    cogdoc_webhook_secret: str = Field(
        default="", validation_alias="COGDOC_WEBHOOK_SECRET"
    )
    cogdoc_webhook_timeout_seconds: float = Field(
        default=3.0, validation_alias="COGDOC_WEBHOOK_TIMEOUT_SECONDS"
    )
    cogdoc_feedback_store: str = Field(
        default="jsonl", validation_alias="COGDOC_FEEDBACK_STORE"
    )
    cogdoc_derived_knowledge_index_auto_refresh: bool = Field(
        default=False,
        validation_alias="COGDOC_DERIVED_KNOWLEDGE_INDEX_AUTO_REFRESH",
    )
    cogdoc_trace_enabled: bool = Field(
        default=True, validation_alias="COGDOC_TRACE_ENABLED"
    )
    cogdoc_trace_dir: str = Field(
        default="logs/traces", validation_alias="COGDOC_TRACE_DIR"
    )

    # 访问控制：密钥逗号分隔，留空则关闭鉴权。
    cogdoc_api_keys: str = Field(default="", validation_alias="COGDOC_API_KEYS")
    # 限流令牌桶：每分钟补充速率 + 突发容量；容量<=0 关闭限流。
    rate_limit_per_minute: int = Field(
        default=120, validation_alias="RATE_LIMIT_PER_MINUTE"
    )
    rate_limit_burst: int = Field(default=120, validation_alias="RATE_LIMIT_BURST")
    cogdoc_offload_workers: int = Field(
        default=2, validation_alias="COGDOC_OFFLOAD_WORKERS"
    )

    # 分层记忆预算：展示历史不受这些限制，只有送入模型的工作上下文会被裁剪。
    memory_short_message_limit: int = Field(
        default=12, ge=2, le=100, validation_alias="COGDOC_MEMORY_SHORT_MESSAGE_LIMIT"
    )
    memory_short_char_limit: int = Field(
        default=6000,
        ge=500,
        le=100000,
        validation_alias="COGDOC_MEMORY_SHORT_CHAR_LIMIT",
    )
    memory_mid_char_limit: int = Field(
        default=4000, ge=500, le=50000, validation_alias="COGDOC_MEMORY_MID_CHAR_LIMIT"
    )
    memory_long_fact_limit: int = Field(
        default=64, ge=1, le=1000, validation_alias="COGDOC_MEMORY_LONG_FACT_LIMIT"
    )
    memory_context_long_limit: int = Field(
        default=8, ge=0, le=100, validation_alias="COGDOC_MEMORY_CONTEXT_LONG_LIMIT"
    )
    memory_retrieval_enabled: bool = Field(
        default=True, validation_alias="COGDOC_MEMORY_RETRIEVAL_ENABLED"
    )
    memory_semantic_enabled: bool = Field(
        default=True, validation_alias="COGDOC_MEMORY_SEMANTIC_ENABLED"
    )
    memory_retrieval_short_limit: int = Field(
        default=8,
        ge=0,
        le=100,
        validation_alias="COGDOC_MEMORY_RETRIEVAL_SHORT_LIMIT",
    )
    memory_retrieval_mid_limit: int = Field(
        default=4,
        ge=0,
        le=100,
        validation_alias="COGDOC_MEMORY_RETRIEVAL_MID_LIMIT",
    )
    memory_retrieval_recent_pin: int = Field(
        default=4,
        ge=0,
        le=100,
        validation_alias="COGDOC_MEMORY_RETRIEVAL_RECENT_PIN",
    )
    memory_semantic_include_short: bool = Field(
        default=False,
        validation_alias="COGDOC_MEMORY_SEMANTIC_INCLUDE_SHORT",
    )
    memory_rrf_k: float = Field(
        default=60.0, gt=0.0, validation_alias="COGDOC_MEMORY_RRF_K"
    )
    memory_recency_weight: float = Field(
        default=1.0, ge=0.0, validation_alias="COGDOC_MEMORY_RECENCY_WEIGHT"
    )
    memory_lexical_weight: float = Field(
        default=1.4, ge=0.0, validation_alias="COGDOC_MEMORY_LEXICAL_WEIGHT"
    )
    memory_semantic_weight: float = Field(
        default=1.6, ge=0.0, validation_alias="COGDOC_MEMORY_SEMANTIC_WEIGHT"
    )
    memory_importance_weight: float = Field(
        default=0.8, ge=0.0, validation_alias="COGDOC_MEMORY_IMPORTANCE_WEIGHT"
    )
    memory_mid_priority_weight: float = Field(
        default=0.8,
        ge=0.0,
        validation_alias="COGDOC_MEMORY_MID_PRIORITY_WEIGHT",
    )

    # 云端模型兼容后端。
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
    # 云端韧性：单次调用硬超时与传输层重试次数。
    llm_timeout_seconds: float = Field(
        default=90.0, validation_alias="LLM_TIMEOUT_SECONDS"
    )
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")
    # 独立阅卷模型配置；留空时回退到云端主模型，但角色仍保持 Judge。
    llm_judge_model_name: str = Field(
        default="", validation_alias="LLM_JUDGE_MODEL_NAME"
    )
    llm_judge_enabled: bool = Field(
        default=True, validation_alias="LLM_JUDGE_ENABLED"
    )
    llm_judge_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, validation_alias="LLM_JUDGE_TEMPERATURE"
    )

    # 云端节点级模型覆盖；留空时回退到 LLM_MODEL_NAME。
    llm_router_model_name: str = Field(
        default="", validation_alias="LLM_ROUTER_MODEL_NAME"
    )
    llm_query_rewriter_model_name: str = Field(
        default="", validation_alias="LLM_QUERY_REWRITER_MODEL_NAME"
    )
    llm_source_resolver_model_name: str = Field(
        default="", validation_alias="LLM_SOURCE_RESOLVER_MODEL_NAME"
    )
    llm_evidence_verifier_model_name: str = Field(
        default="", validation_alias="LLM_EVIDENCE_VERIFIER_MODEL_NAME"
    )
    llm_qa_generator_model_name: str = Field(
        default="", validation_alias="LLM_QA_GENERATOR_MODEL_NAME"
    )
    llm_summary_generator_model_name: str = Field(
        default="", validation_alias="LLM_SUMMARY_GENERATOR_MODEL_NAME"
    )
    llm_compare_profile_model_name: str = Field(
        default="", validation_alias="LLM_COMPARE_PROFILE_MODEL_NAME"
    )
    llm_compare_conclusion_model_name: str = Field(
        default="", validation_alias="LLM_COMPARE_CONCLUSION_MODEL_NAME"
    )
    llm_router_backend: str = Field(
        default="default", validation_alias="LLM_ROUTER_BACKEND"
    )
    llm_query_rewriter_backend: str = Field(
        default="default", validation_alias="LLM_QUERY_REWRITER_BACKEND"
    )
    llm_source_resolver_backend: str = Field(
        default="default", validation_alias="LLM_SOURCE_RESOLVER_BACKEND"
    )
    llm_evidence_verifier_backend: str = Field(
        default="default", validation_alias="LLM_EVIDENCE_VERIFIER_BACKEND"
    )
    llm_qa_generator_backend: str = Field(
        default="default", validation_alias="LLM_QA_GENERATOR_BACKEND"
    )
    llm_summary_generator_backend: str = Field(
        default="default", validation_alias="LLM_SUMMARY_GENERATOR_BACKEND"
    )
    llm_compare_profile_backend: str = Field(
        default="default", validation_alias="LLM_COMPARE_PROFILE_BACKEND"
    )
    llm_compare_conclusion_backend: str = Field(
        default="default", validation_alias="LLM_COMPARE_CONCLUSION_BACKEND"
    )

    # 本地模型兼容后端。
    ollama_model_name: str = Field(
        default="qwen2.5:7b", validation_alias="OLLAMA_MODEL_NAME"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_api_key: str = Field(default="ollama", validation_alias="OLLAMA_API_KEY")
    # 本地模型通常比云端慢，超时更长且重试更少。
    ollama_timeout_seconds: float = Field(
        default=180.0, validation_alias="OLLAMA_TIMEOUT_SECONDS"
    )
    ollama_max_retries: int = Field(default=1, validation_alias="OLLAMA_MAX_RETRIES")

    # 本地节点级模型覆盖；留空时回退到 OLLAMA_MODEL_NAME。
    ollama_router_model_name: str = Field(
        default="", validation_alias="OLLAMA_ROUTER_MODEL_NAME"
    )
    ollama_query_rewriter_model_name: str = Field(
        default="", validation_alias="OLLAMA_QUERY_REWRITER_MODEL_NAME"
    )
    ollama_source_resolver_model_name: str = Field(
        default="", validation_alias="OLLAMA_SOURCE_RESOLVER_MODEL_NAME"
    )
    ollama_evidence_verifier_model_name: str = Field(
        default="", validation_alias="OLLAMA_EVIDENCE_VERIFIER_MODEL_NAME"
    )
    ollama_qa_generator_model_name: str = Field(
        default="", validation_alias="OLLAMA_QA_GENERATOR_MODEL_NAME"
    )
    ollama_summary_generator_model_name: str = Field(
        default="", validation_alias="OLLAMA_SUMMARY_GENERATOR_MODEL_NAME"
    )
    ollama_compare_profile_model_name: str = Field(
        default="", validation_alias="OLLAMA_COMPARE_PROFILE_MODEL_NAME"
    )
    ollama_compare_conclusion_model_name: str = Field(
        default="", validation_alias="OLLAMA_COMPARE_CONCLUSION_MODEL_NAME"
    )

    # 检索与生成控制。
    qa_retrieval_top_k: int = Field(default=9, validation_alias="QA_RETRIEVAL_TOP_K")
    qa_rerank_top_n: int = Field(default=3, validation_alias="QA_RERANK_TOP_N")
    qa_rerank_max_candidates: int = Field(
        default=12, validation_alias="QA_RERANK_MAX_CANDIDATES"
    )
    qa_rerank_on_cpu: bool = Field(default=False, validation_alias="QA_RERANK_ON_CPU")
    qa_abstain_enabled: bool = Field(
        default=True, validation_alias="QA_ABSTAIN_ENABLED"
    )
    qa_abstain_max_vector_distance: float = Field(
        default=0.86,
        ge=0.0,
        validation_alias="QA_ABSTAIN_MAX_VECTOR_DISTANCE",
    )
    qa_abstain_min_bm25_score: float = Field(
        default=10.0,
        ge=0.0,
        validation_alias="QA_ABSTAIN_MIN_BM25_SCORE",
    )
    qa_abstain_min_knowledge_score: float = Field(
        default=0.5,
        ge=0.0,
        validation_alias="QA_ABSTAIN_MIN_KNOWLEDGE_SCORE",
    )
    qa_evidence_verify_enabled: bool = Field(
        default=True, validation_alias="QA_EVIDENCE_VERIFY_ENABLED"
    )
    qa_evidence_verify_max_docs: int = Field(
        default=3, ge=1, le=10, validation_alias="QA_EVIDENCE_VERIFY_MAX_DOCS"
    )
    qa_evidence_verify_max_chars_per_doc: int = Field(
        default=1600,
        ge=200,
        le=10000,
        validation_alias="QA_EVIDENCE_VERIFY_MAX_CHARS_PER_DOC",
    )
    qa_evidence_verify_borderline_min_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        validation_alias="QA_EVIDENCE_VERIFY_BORDERLINE_MIN_SCORE",
    )
    hybrid_rrf_k: int = Field(default=60, validation_alias="HYBRID_RRF_K")
    cloud_section_max_workers: int = Field(
        default=6, validation_alias="CLOUD_SECTION_MAX_WORKERS"
    )

    # 模型设备阈值，单位兆字节。
    embedder_min_cuda_free_mb: int = Field(
        default=800, validation_alias="EMBEDDER_MIN_CUDA_FREE_MB"
    )
    reranker_min_cuda_free_mb: int = Field(
        default=2800, validation_alias="RERANKER_MIN_CUDA_FREE_MB"
    )
    torch_num_threads: int = Field(
        default=2, validation_alias="COGDOC_TORCH_NUM_THREADS"
    )
    cogdoc_embedder_max_concurrency: int = Field(
        default=1, validation_alias="COGDOC_EMBEDDER_MAX_CONCURRENCY"
    )
    cogdoc_reranker_max_concurrency: int = Field(
        default=1, validation_alias="COGDOC_RERANKER_MAX_CONCURRENCY"
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

    # 处理数据目录。
    @property
    def data_dir(self) -> Path:
        return Path(self.cogdoc_data_dir)

    # 构造分层记忆策略。
    @property
    def memory_policy(self):
        from cogdoc.memory.manager import MemoryPolicy

        return MemoryPolicy(
            short_term_message_limit=self.memory_short_message_limit,
            short_term_char_limit=self.memory_short_char_limit,
            mid_term_char_limit=self.memory_mid_char_limit,
            long_term_fact_limit=self.memory_long_fact_limit,
            context_long_term_limit=self.memory_context_long_limit,
            memory_retrieval_enabled=self.memory_retrieval_enabled,
            memory_semantic_enabled=self.memory_semantic_enabled,
            memory_retrieval_short_limit=self.memory_retrieval_short_limit,
            memory_retrieval_mid_limit=self.memory_retrieval_mid_limit,
            memory_retrieval_recent_pin=self.memory_retrieval_recent_pin,
            memory_semantic_include_short=self.memory_semantic_include_short,
            memory_rrf_k=self.memory_rrf_k,
            memory_recency_weight=self.memory_recency_weight,
            memory_lexical_weight=self.memory_lexical_weight,
            memory_semantic_weight=self.memory_semantic_weight,
            memory_importance_weight=self.memory_importance_weight,
            memory_mid_priority_weight=self.memory_mid_priority_weight,
        )

    # 处理向量持久化目录。
    @property
    def chroma_persist_dir(self) -> str:
        return str(self.data_dir / "chroma_db")

    # 处理关键词索引持久化目录。
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
        # 每个知识库一个源文档目录，构建时硬链接快照到索引代工作区。
        return str(self.data_dir / "kb" / kb_id / "sources")

    # 完成 知识库状态路径 处理。
    def kb_state_path(self, kb_id: str) -> str:
        # 事务化索引的提交指针，与每文档清单分离。
        return str(self.data_dir / "kb" / kb_id / "state.json")

    # 完成 知识库索引代目录 处理。
    def kb_generation_dir(self, kb_id: str, generation_id: str) -> str:
        # 单个索引代的工作区，保存源文件快照和内部产物。
        return str(self.data_dir / "kb" / kb_id / "generations" / generation_id)

    # 处理知识库集合标识。
    def kb_collection_id(self, kb_id: str, gen_id: str) -> str:
        # 集合标识由知识库短哈希和索引代标识组成。
        import hashlib

        return f"{hashlib.sha256(kb_id.encode()).hexdigest()[:8]}-{gen_id}"

    # 处理接口密钥集合。
    @property
    def api_key_set(self) -> set[str]:
        # 解析逗号分隔的密钥列表，空集合表示鉴权关闭。
        return {k.strip() for k in self.cogdoc_api_keys.split(",") if k.strip()}

    # 处理状态库路径。
    @property
    def state_db_path(self) -> str:
        # 会话与入库任务落盘，进程重启不丢状态。
        return str(self.data_dir / "state.db")

    # 处理反馈数据库路径。
    @property
    def feedback_db_path(self) -> str:
        return str(self.data_dir / "feedback" / "feedback.db")

    # 处理反馈日志路径。
    @property
    def feedback_log_path(self) -> str:
        return str(self.data_dir / "feedback" / "feedback.jsonl")

    # 完成 坏样本用例列表路径 处理。
    @property
    def bad_cases_path(self) -> str:
        # 点踩和纠错自动归集到此，供离线质量评测使用。
        return str(self.data_dir / "feedback" / "bad_cases.jsonl")

    # 完成 反馈分析路径 处理。
    @property
    def feedback_analysis_path(self) -> str:
        return str(self.data_dir / "feedback" / "feedback_analysis.jsonl")

    # 完成 派生知识路径 处理。
    @property
    def derived_knowledge_path(self) -> str:
        return str(self.data_dir / "knowledge" / "derived_knowledge.jsonl")

    # 完成 检索反馈路径 处理。
    @property
    def retrieval_feedback_path(self) -> str:
        return str(self.data_dir / "feedback" / "retrieval_feedback.jsonl")

    # 返回根目录。
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    # 处理显存阈值。
    def cuda_min_free_bytes(self, setting_name: str) -> int:
        mb_by_name = {
            "EMBEDDER_MIN_CUDA_FREE_MB": self.embedder_min_cuda_free_mb,
            "RERANKER_MIN_CUDA_FREE_MB": self.reranker_min_cuda_free_mb,
        }
        if setting_name not in mb_by_name:
            raise ValueError(f"未知 CUDA 显存阈值配置: {setting_name}")
        return int(mb_by_name[setting_name]) * 1024 * 1024

    # 返回节点配置的模型名，节点未覆盖时使用对应后端的全局模型。
    def model_name_for_node(self, node_name: str | None, *, is_local: bool) -> str:
        default = self.ollama_model_name if is_local else self.llm_model_name
        if not node_name:
            return default
        prefix = "ollama" if is_local else "llm"
        value = getattr(self, f"{prefix}_{node_name}_model_name", "")
        return str(value or default).strip()

    # 节点可显式选择云端或本地后端；default 跟随本次请求模式。
    def is_local_for_node(self, node_name: str, *, request_is_local: bool) -> bool:
        backend = str(getattr(self, f"llm_{node_name}_backend", "default")).lower()
        if backend == "default":
            return request_is_local
        if backend in {"local", "ollama"}:
            return True
        if backend == "cloud":
            return False
        raise ValueError(
            f"无效节点后端 LLM_{node_name.upper()}_BACKEND={backend!r}; "
            "可选值为 default、cloud、local"
        )


# 返回设置。
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
