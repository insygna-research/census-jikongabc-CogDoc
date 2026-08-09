from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from cogdoc.config.settings import get_settings
from cogdoc.tools.citation_ledger import is_valid_evidence_id
from cogdoc.tools.public_citation_ledger import validate_public_citation_ledger
from cogdoc.tools.retriever.metadata import safe_retrieval_metadata


API_SCHEMA_VERSION: Literal["v1"] = "v1"


# 所有接口模型的基类，统一严格契约与枚举字符串化。
class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# 对话请求模式，支持自动路由或强制指定任务。
class ChatMode(str, Enum):
    AUTO = "auto"
    QA = "qa"
    SUMMARY = "summary"
    COMPARE = "compare"


# 实际执行的任务类型，响应里回显。
class ChatTask(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    COMPARE = "compare"
    UNKNOWN = "unknown"


# 稳定错误码，前端按码处理失败、不依赖文案。
class ErrorCode(str, Enum):
    CITATION_REJECTED = "CITATION_REJECTED"
    NO_EVIDENCE = "NO_EVIDENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    STREAM_INTERRUPTED = "STREAM_INTERRUPTED"
    UNKNOWN_INTENT = "UNKNOWN_INTENT"
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    REQUEST_THROTTLED = "REQUEST_THROTTLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    KB_NOT_FOUND = "KB_NOT_FOUND"
    KB_EXISTS = "KB_EXISTS"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INVALID_PDF = "INVALID_PDF"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    INGEST_FAILED = "INGEST_FAILED"
    KB_CLEANUP_FAILED = "KB_CLEANUP_FAILED"
    TRACE_NOT_FOUND = "TRACE_NOT_FOUND"
    KNOWLEDGE_NOT_FOUND = "KNOWLEDGE_NOT_FOUND"
    RESEARCH_JOB_NOT_FOUND = "RESEARCH_JOB_NOT_FOUND"
    RESEARCH_JOB_REVISION_CONFLICT = "RESEARCH_JOB_REVISION_CONFLICT"
    RESEARCH_JOB_STATE_CONFLICT = "RESEARCH_JOB_STATE_CONFLICT"


# 带查询和知识库标识的请求基类。
class QueryDocRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    query: str = Field(min_length=1)
    doc_id: str = Field(default_factory=lambda: get_settings().cogdoc_default_doc_id)

    # 清理必填文本。
    @field_validator("query", "doc_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


# 对话接口请求体。
class ChatRequest(QueryDocRequest):
    session_id: str | None = None
    mode: ChatMode = ChatMode.AUTO
    is_local: bool = False

    # 解析强制任务模式。
    @property
    def forced_task(self) -> str | None:
        # 枚举值已转成字符串，自动模式不强制任务。
        return None if self.mode == ChatMode.AUTO else str(self.mode)


# 引用来源取自文档元数据，不含正文。
class Citation(ApiModel):
    chunk_id: str = ""
    source_type: str = "document"
    knowledge_id: str = ""
    source: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None


# 单次引用在最终答案中的位置；偏移是 Unicode code point 的 0-based half-open 区间。
class CitationOccurrence(ApiModel):
    """One citation position in Python/Unicode code-point coordinates."""

    index: int = Field(ge=0, strict=True, description="全答案引用出现序号")
    answer_start: int = Field(
        ge=0,
        strict=True,
        description="0-based Unicode code-point 起点（包含）",
    )
    answer_end: int = Field(
        ge=0,
        strict=True,
        description="0-based Unicode code-point 终点（不包含）",
    )

    @model_validator(mode="after")
    def _validate_range(self):
        if self.answer_end <= self.answer_start:
            raise ValueError("answer_end must be greater than answer_start")
        return self


# 精确引用账本只公开定位信息，不含证据正文、内部展示模板或私有元数据。
class CitationLedgerEntry(ApiModel):
    evidence_id: str = Field(min_length=4, max_length=160)
    chunk_id: str = Field(min_length=1)
    source_type: str = "document"
    knowledge_id: str = ""
    source: str = ""
    page: int | None = Field(default=None, ge=0, strict=True)
    page_start: int | None = Field(default=None, ge=0, strict=True)
    page_end: int | None = Field(default=None, ge=0, strict=True)
    span_start: int = Field(ge=0, strict=True)
    span_end: int = Field(ge=0, strict=True)
    occurrences: list[CitationOccurrence] = Field(min_length=1)

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not is_valid_evidence_id(value):
            raise ValueError("evidence_id must use canonical E001/E1000 spelling")
        return value

    @model_validator(mode="after")
    def _validate_entry(self):
        if self.span_end <= self.span_start:
            raise ValueError("span_end must be greater than span_start")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.source_type == "derived_knowledge":
            if not self.knowledge_id.strip():
                raise ValueError("derived knowledge citation requires knowledge_id")
        elif self.source_type == "document":
            if not self.source.strip() or self.page is None:
                raise ValueError("document citation requires source and page")
        else:
            raise ValueError("unsupported citation source_type")
        return self


# 证据片段带截断预览，供前端证据面板展示。
class Evidence(ApiModel):
    chunk_id: str = ""
    parent_chunk_id: str = ""
    section_title: str = ""
    section_path: str = ""
    section_level: int | None = None
    child_index_in_parent: int | None = None
    source_type: str = "document"
    knowledge_id: str = ""
    chunk_index: int | None = None
    source: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    rerank_score: float | None = None
    rewrite_query: str | None = None
    text_preview: str = ""
    retrieval: dict[str, Any] = Field(default_factory=dict)


# 公开审计摘要不含声明全文和证据正文，完整明细仅留在 trace。
class ClaimAuditSummary(ApiModel):
    status: str = "not_run"
    reason_code: str = ""
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    repair: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None


# 对话接口结构化响应。
class ChatResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    request_id: str
    trace_id: str
    doc_id: str
    session_id: str | None = None
    task_type: ChatTask
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    citation_ledger: list[CitationLedgerEntry] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    critique: str = ""
    is_valid: bool
    claim_audit: ClaimAuditSummary | None = None


# 独立任务接口请求体，摘要和对比任务由路由层固定。
class TaskRequest(QueryDocRequest):
    session_id: str | None = None
    is_local: bool = False


# 检索接口请求体，不调用模型。
class RetrieveRequest(QueryDocRequest):
    top_k: int = Field(default=8, ge=1, le=50)
    rerank: bool = False
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)


# 检索命中项，供前端证据面板和调试面板直接消费。
class RetrieveHit(Evidence):
    rank: int


# 检索接口结构化响应。
class RetrieveResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    doc_id: str
    query: str
    top_k: int
    rerank: bool
    hits: list[RetrieveHit] = Field(default_factory=list)


# 统一错误响应体，所有失败路径共用。
class ErrorResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    error_code: ErrorCode
    message: str
    request_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] | None = None


# 入库任务状态机。
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# 建知识库请求体，限制长度避免集合名截断后碰撞。
class KnowledgeBaseCreate(ApiModel):
    kb_id: str = Field(min_length=1, max_length=56)

    # 校验结果。
    @field_validator("kb_id")
    @classmethod
    def _slug(cls, value: str) -> str:
        # 标识符会进入路径，禁止分隔符与空白，避免目录穿越。
        stripped = value.strip()
        if (
            not stripped
            or any(c in stripped for c in "/\\ \t")
            or stripped in {".", ".."}
        ):
            raise ValueError("kb_id 只能是不含路径分隔符与空白的标识符")
        return stripped


# 知识库元数据，预留多租户字段。
class KnowledgeBase(ApiModel):
    kb_id: str
    created_at: str
    document_count: int = 0
    tenant_id: str = "default"
    owner_id: str = "default"


class ResearchJobCreate(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str = Field(min_length=1, max_length=56)
    objective: str = Field(min_length=1, max_length=4000)
    title: str = Field(default="", max_length=160)
    section_titles: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("kb_id", "objective", "title")
    @classmethod
    def _strip_research_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("kb_id")
    @classmethod
    def _require_research_kb_id(cls, value: str) -> str:
        if not value:
            raise ValueError("kb_id must not be blank")
        return value

    @field_validator("objective")
    @classmethod
    def _require_objective(cls, value: str) -> str:
        if not value:
            raise ValueError("objective must not be blank")
        return value

    @field_validator("section_titles")
    @classmethod
    def _normalize_section_titles(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()) for value in values]
        if any(not value for value in normalized):
            raise ValueError("section titles must not be blank")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("section titles must be unique")
        return normalized


class ResearchPlanSectionInput(ApiModel):
    title: str = Field(min_length=1, max_length=160)
    research_question: str = Field(min_length=1, max_length=2000)

    @field_validator("title", "research_question")
    @classmethod
    def _strip_plan_text(cls, value: str) -> str:
        stripped = " ".join(value.split())
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ResearchPlanUpdate(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    expected_revision: int = Field(ge=1, strict=True)
    sections: list[ResearchPlanSectionInput] = Field(min_length=1, max_length=12)


class ResearchReviewDecision(ApiModel):
    section_id: str = Field(min_length=1, max_length=64)
    decision: Literal["approved", "accepted_gap", "changes_requested"]
    note: str = Field(default="", max_length=2000)

    @field_validator("section_id", "note")
    @classmethod
    def _strip_review_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def _require_change_note(self):
        if self.decision == "changes_requested" and not self.note:
            raise ValueError("changes_requested review requires a non-blank note")
        return self


class ResearchReportReviewUpdate(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    expected_revision: int = Field(ge=1, strict=True)
    decisions: list[ResearchReviewDecision] = Field(min_length=1, max_length=12)


class ResearchReportPublishRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    expected_revision: int = Field(ge=1, strict=True)


class ResearchPlanSection(ApiModel):
    section_id: str
    position: int = Field(ge=1, strict=True)
    title: str
    research_question: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    evidence_status: Literal[
        "unsearched", "missing", "partial", "supported", "contradictory"
    ] = "unsearched"
    evidence_requirement_ids: list[str] = Field(default_factory=list)
    evidence: list["ResearchEvidenceItem"] = Field(default_factory=list)
    execution_metrics: dict[str, Any] = Field(default_factory=dict)
    citation_ledger: list[CitationLedgerEntry] = Field(default_factory=list)
    revision_instruction: str = ""
    verification_status: str = ""
    verification_reason_code: str = ""
    generation_status: str = ""
    content: str = ""
    review_status: Literal[
        "not_started",
        "pending",
        "approved",
        "accepted_gap",
        "changes_requested",
    ] = "not_started"
    review_note: str = ""
    reviewed_at: str | None = None
    error: str = ""


class ResearchEvidenceItem(ApiModel):
    chunk_id: str = ""
    source_type: str = "document"
    knowledge_id: str = ""
    source: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str = ""
    text_preview: str = ""
    search_channel: str = ""
    rerank_score: float | int | None = None
    rrf_score: float | int | None = None


class ResearchReportArtifact(ApiModel):
    format: Literal["markdown"] = "markdown"
    content: str
    citation_ledger: list[CitationLedgerEntry] = Field(default_factory=list)
    verification_metrics: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1, strict=True)
    generated_at: str
    published_at: str | None = None


class ResearchReportVersion(ApiModel):
    version: int = Field(ge=1, strict=True)
    report_status: str
    review_status: str
    archived_at: str
    report: ResearchReportArtifact


class ResearchJob(ApiModel):
    job_id: str
    kb_id: str
    title: str
    objective: str
    status: Literal[
        "planned",
        "running",
        "paused",
        "evidence_ready",
        "generating",
        "completed",
        "failed",
        "cancelled",
    ]
    revision: int = Field(ge=1, strict=True)
    created_at: str
    updated_at: str
    sections: list[ResearchPlanSection]
    execution_id: str = ""
    started_at: str | None = None
    evidence_completed_at: str | None = None
    report_status: Literal[
        "not_started",
        "generating",
        "ready",
        "ready_with_gaps",
        "published",
        "failed",
    ] = "not_started"
    report_execution_id: str = ""
    report_completed_at: str | None = None
    report: ResearchReportArtifact | None = None
    report_version: int = Field(default=0, ge=0, strict=True)
    report_history: list[ResearchReportVersion] = Field(default_factory=list)
    review_status: Literal[
        "not_started", "pending", "approved", "changes_requested", "published"
    ] = "not_started"
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    published_report: ResearchReportArtifact | None = None
    published_at: str | None = None
    regeneration_section_ids: list[str] = Field(default_factory=list)
    last_regenerated_section_ids: list[str] = Field(default_factory=list)
    error: str = ""


class ResearchJobResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    job: ResearchJob


class ResearchJobListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    jobs: list[ResearchJob] = Field(default_factory=list)


# 知识库内的一篇文档，来自清单。
class Document(ApiModel):
    name: str
    sha256: str = ""


# 知识库来源文件列表响应。
class SourceListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    sources: list[str] = Field(default_factory=list)


# 文档分块预览，不返回完整正文。
class ChunkPreview(ApiModel):
    chunk_id: str = ""
    chunk_index: int | None = None
    source: str = ""
    source_sha256: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    text_hash: str = ""
    anchor_hit: bool = False
    text_preview: str = ""
    context_preview: str = ""


# 单个来源文件的分块预览响应。
class SourceChunksResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    source: str
    total_count: int
    offset: int
    limit: int
    chunks: list[ChunkPreview] = Field(default_factory=list)


# OCR 入库汇总，不包含页面正文或错误详情。
class OcrSummary(ApiModel):
    candidate_pages: int = 0
    attempted_pages: int = 0
    succeeded_pages: int = 0
    degraded_pages: int = 0
    failed_pages: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)


# 后台入库任务记录，供轮询状态。
class IndexJob(ApiModel):
    job_id: str
    kb_id: str
    status: JobStatus
    created_at: str
    finished_at: str | None = None
    document_count: int | None = None
    chunk_count: int | None = None
    ocr_summary: OcrSummary | None = None
    error_code: ErrorCode | None = None
    message: str | None = None


# 反馈类型：赞 / 踩 / 纠错。
class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"


# 回答反馈原因，用于后续结构化分析和调权。
class FeedbackIssueType(str, Enum):
    NO_EVIDENCE = "no_evidence"
    WRONG_ANSWER = "wrong_answer"
    BAD_RETRIEVAL = "bad_retrieval"
    CORRECTION = "correction"
    OTHER = "other"


# 被反馈来源类型。
class FeedbackSourceType(str, Enum):
    DOCUMENT = "document"
    DERIVED_KNOWLEDGE = "derived_knowledge"
    MIXED = "mixed"
    NONE = "none"


# 反馈请求体，跟踪标识关联被反馈的那次回答。
class FeedbackRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    trace_id: str = Field(min_length=1)
    feedback: FeedbackType
    kb_id: str | None = None
    session_id: str | None = None
    query: str | None = None
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source_type: FeedbackSourceType | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_type: FeedbackIssueType | None = None
    feedback_text: str | None = None
    correction_text: str | None = None
    comment: str | None = None
    correction: str | None = None
    save_as_knowledge: bool = False
    skip_retrieval_feedback: bool = False
    related_document_id: str | None = None
    related_source: str | None = None
    related_source_sha256: str | None = None
    related_chunk_ids: list[str] = Field(default_factory=list)
    related_page_start: int | None = Field(default=None, ge=0)
    related_page_end: int | None = Field(default=None, ge=0)
    related_chunk_text_hash: str | None = None
    related_anchor_text: str | None = None
    source_note: str | None = None
    certainty: Literal["high", "medium", "low"] = "medium"
    created_by: str | None = None


# 反馈落盘结果，标记是否进入坏样本集。
class FeedbackResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    feedback_id: str
    status: str = "recorded"
    is_bad_case: bool
    feedback_analysis_id: str | None = None
    feedback_analysis_action: str | None = None
    feedback_analysis_confidence: float | None = None
    knowledge_id: str | None = None
    knowledge_status: str | None = None
    knowledge_deduplicated: bool = False
    retrieval_eval_draft_id: str | None = None
    retrieval_eval_draft_status: str | None = None


# 证据评测草稿的人工审核请求。reviewer 身份只能来自服务端审核密钥，
# 不接受客户端自报 actor。
class RetrievalEvalDraftReviewRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    decision: Literal["approved", "rejected"]
    expected_revision: int = Field(ge=1, strict=True)
    annotations: dict[str, Any] | None = None
    reason: str = ""


# 反馈列表响应。
class FeedbackListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    feedback: list[dict[str, Any]] = Field(default_factory=list)


# 派生知识状态。
class KnowledgeStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"
    ARCHIVED = "archived"


# 派生知识来源。
class KnowledgeOrigin(str, Enum):
    MANUAL_ENTRY = "manual_entry"
    CORRECTION = "correction"
    NO_EVIDENCE = "no_evidence"
    SAVED_ANSWER = "saved_answer"
    AGENT_SUGGESTED = "agent_suggested"


# 派生知识可信度。
class KnowledgeCertainty(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# 主动新增知识请求体。
class KnowledgeCreateRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    related_document_id: str | None = None
    related_source: str | None = None
    related_source_sha256: str | None = None
    related_chunk_ids: list[str] = Field(default_factory=list)
    related_page_start: int | None = Field(default=None, ge=0)
    related_page_end: int | None = Field(default=None, ge=0)
    related_chunk_text_hash: str | None = None
    related_anchor_text: str | None = None
    source_note: str | None = None
    certainty: KnowledgeCertainty = KnowledgeCertainty.MEDIUM
    origin: KnowledgeOrigin = KnowledgeOrigin.MANUAL_ENTRY
    created_from_trace_id: str | None = None
    created_by: str | None = None

    # 清理必填文本。
    @field_validator("kb_id", "text")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


# 审核操作请求体。
class KnowledgeReviewRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    actor: str | None = None
    note: str | None = None
    related_document_id: str | None = None
    related_source: str | None = None
    related_source_sha256: str | None = None
    related_chunk_ids: list[str] | None = None
    related_page_start: int | None = Field(default=None, ge=0)
    related_page_end: int | None = Field(default=None, ge=0)
    related_chunk_text_hash: str | None = None
    related_anchor_text: str | None = None


# 知识修订请求体。
class KnowledgeReviseRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    text: str = Field(min_length=1)
    related_document_id: str | None = None
    related_source: str | None = None
    related_source_sha256: str | None = None
    related_chunk_ids: list[str] | None = None
    related_page_start: int | None = Field(default=None, ge=0)
    related_page_end: int | None = Field(default=None, ge=0)
    related_chunk_text_hash: str | None = None
    related_anchor_text: str | None = None
    source_note: str | None = None
    certainty: KnowledgeCertainty = KnowledgeCertainty.MEDIUM
    created_from_trace_id: str | None = None
    created_by: str | None = None

    # 清理修订正文。
    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


# 批量审核请求体。
class KnowledgeBatchReviewRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    actor: str | None = None
    note: str | None = None
    knowledge_ids: list[str] = Field(min_length=1)


# 派生知识公开视图。
class DerivedKnowledge(ApiModel):
    knowledge_id: str
    kb_id: str
    text: str
    normalized_text: str
    normalized_hash: str
    version: int = 1
    previous_version_id: str | None = None
    conflict_group_id: str | None = None
    related_document_id: str | None = None
    related_source: str | None = None
    related_source_sha256: str | None = None
    related_chunk_ids: list[str] = Field(default_factory=list)
    related_page_start: int | None = None
    related_page_end: int | None = None
    related_chunk_text_hash: str | None = None
    related_anchor_text: str | None = None
    source_note: str | None = None
    certainty: KnowledgeCertainty = KnowledgeCertainty.MEDIUM
    status: KnowledgeStatus = KnowledgeStatus.PENDING
    origin: KnowledgeOrigin = KnowledgeOrigin.MANUAL_ENTRY
    created_from_trace_id: str | None = None
    created_by: str | None = None
    created_at: str
    updated_at: str
    archived_at: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None


# 知识冲突候选。
class KnowledgeConflictCandidate(ApiModel):
    knowledge_id: str
    text: str
    status: KnowledgeStatus
    origin: KnowledgeOrigin = KnowledgeOrigin.MANUAL_ENTRY
    related_source: str | None = None
    created_at: str


# 新增知识响应。
class KnowledgeCreateResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    knowledge: DerivedKnowledge
    deduplicated: bool = False
    requires_review: bool = False
    conflicts: list[KnowledgeConflictCandidate] = Field(default_factory=list)


# 知识列表响应。
class KnowledgeListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    knowledge: list[DerivedKnowledge] = Field(default_factory=list)


# 批量审核响应。
class KnowledgeBatchReviewResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    updated: list[DerivedKnowledge] = Field(default_factory=list)
    missing_ids: list[str] = Field(default_factory=list)


# 过期知识扫描响应。
class KnowledgeStaleScanResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    stale_marked: int = 0
    stale_knowledge: list[DerivedKnowledge] = Field(default_factory=list)


# 审核队列摘要响应。
class ReviewQueueSummaryResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    knowledge: dict[str, int] = Field(default_factory=dict)
    knowledge_origin: dict[str, int] = Field(default_factory=dict)
    knowledge_conflicts: dict[str, int] = Field(default_factory=dict)
    knowledge_auto_review: dict[str, int] = Field(default_factory=dict)
    feedback_counts: dict[str, Any] = Field(default_factory=dict)
    feedback_analysis: dict[str, int] = Field(default_factory=dict)
    feedback_analysis_type: dict[str, int] = Field(default_factory=dict)
    retrieval_feedback: dict[str, int] = Field(default_factory=dict)


# 审核队列导出响应。
class ReviewQueueExportResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    generated_at: str
    summary: ReviewQueueSummaryResponse
    pending_knowledge: list[DerivedKnowledge] = Field(default_factory=list)
    stale_knowledge: list[DerivedKnowledge] = Field(default_factory=list)
    auto_review_events: list[dict[str, Any]] = Field(default_factory=list)
    feedback_analysis_needs_review: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_feedback_enabled: list[dict[str, Any]] = Field(default_factory=list)
    feedback_bad_cases: list[dict[str, Any]] = Field(default_factory=list)


# 待审核计数响应。
class KnowledgePendingCountResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    pending: int = 0
    stale: int = 0
    feedback_analysis_needs_review: int = 0
    total: int = 0


# 反馈闭环指标响应。
class FeedbackLoopMetricsResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    kb_id: str
    counts: dict[str, int] = Field(default_factory=dict)
    rates: dict[str, float | None] = Field(default_factory=dict)


# 会话多轮历史，刷新后前端据此还原聊天记录。
class SessionHistoryResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    session_id: str
    doc_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


# 三层记忆调试响应。
class MemorySnapshotResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    session_id: str
    doc_id: str
    short_term: list[dict[str, Any]] = Field(default_factory=list)
    mid_term: dict[str, Any] = Field(default_factory=dict)
    long_term: list[dict[str, Any]] = Field(default_factory=list)


# 会话列表里的一条，标题取首条用户消息。
class SessionSummary(ApiModel):
    session_id: str
    title: str
    message_count: int


# 某知识库下的全部会话，供前端多对话列表。
class SessionListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    doc_id: str
    sessions: list[SessionSummary] = Field(default_factory=list)


# 跟踪文件步骤摘要。
class TraceSummary(ApiModel):
    step_count: int = 0
    error_count: int = 0
    evidence_ref_count: int = 0
    node_names: list[str] = Field(default_factory=list)
    claim_audit: ClaimAuditSummary | None = None


# 跟踪文件响应体。
class TraceResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    trace_id: str
    request_id: str
    task_type: str
    status: str
    duration_ms: float | None = None
    execution_status: str = "SUCCESS"
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_completeness: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    summary: TraceSummary = Field(default_factory=TraceSummary)
    error: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)


# 跟踪列表项，供调试控制台浏览最近请求。
class TraceListItem(ApiModel):
    trace_id: str
    request_id: str
    query_preview: str = ""
    task_type: str
    status: str
    duration_ms: float | None = None
    modified_at: str
    summary: TraceSummary = Field(default_factory=TraceSummary)


# 最近跟踪文件列表响应。
class TraceListResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    traces: list[TraceListItem] = Field(default_factory=list)


# 转换为映射。
def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


# 解析整数或空值。
def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# 解析浮点数或空值。
def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int:
    parsed = _int_or_none(value)
    return max(parsed, 0) if parsed is not None else 0


# 规范化任务类型。
def _normalize_task_type(value: Any) -> ChatTask:
    # 上游意外或缺失时一律归一到未知任务，契约不因字段漂移崩。
    task = str(value or ChatTask.UNKNOWN.value)
    return (
        ChatTask(task)
        if task in {item.value for item in ChatTask}
        else ChatTask.UNKNOWN
    )


# 从映射构建引用。
def _citation_from_mapping(item: Any) -> Citation:
    data = _as_mapping(item)
    page = _int_or_none(data.get("page"))
    return Citation(
        chunk_id=str(data.get("chunk_id", "") or ""),
        source_type=str(data.get("source_type", "document") or "document"),
        knowledge_id=str(data.get("knowledge_id", "") or ""),
        source=str(data.get("source", "") or ""),
        page=page,
        page_start=_int_or_none(data.get("page_start", page)),
        page_end=_int_or_none(data.get("page_end", page)),
    )


def _citation_occurrence_from_mapping(item: Any) -> CitationOccurrence:
    data = _as_mapping(item)
    return CitationOccurrence(
        index=_nonnegative_int(data.get("index")),
        answer_start=_nonnegative_int(data.get("answer_start")),
        answer_end=_nonnegative_int(data.get("answer_end")),
    )


def _citation_ledger_entry_from_mapping(item: Any) -> CitationLedgerEntry:
    data = _as_mapping(item)
    return CitationLedgerEntry(
        evidence_id=str(data.get("evidence_id", "") or ""),
        chunk_id=str(data.get("chunk_id", "") or ""),
        source_type=str(data.get("source_type", "document") or "document"),
        knowledge_id=str(data.get("knowledge_id", "") or ""),
        source=str(data.get("source", "") or ""),
        page=_int_or_none(data.get("page")),
        page_start=_int_or_none(data.get("page_start")),
        page_end=_int_or_none(data.get("page_end")),
        span_start=_nonnegative_int(data.get("span_start")),
        span_end=_nonnegative_int(data.get("span_end")),
        occurrences=[
            _citation_occurrence_from_mapping(occurrence)
            for occurrence in list(data.get("occurrences", []) or [])
        ],
    )


# 从映射构建证据。
def _evidence_from_mapping(item: Any) -> Evidence:
    data = _as_mapping(item)
    page = _int_or_none(data.get("page"))
    return Evidence(
        chunk_id=str(data.get("chunk_id", "") or ""),
        parent_chunk_id=str(data.get("parent_chunk_id", "") or ""),
        section_title=str(data.get("section_title", "") or ""),
        section_path=str(data.get("section_path", "") or ""),
        section_level=_int_or_none(data.get("section_level")),
        child_index_in_parent=_int_or_none(data.get("child_index_in_parent")),
        source_type=str(data.get("source_type", "document") or "document"),
        knowledge_id=str(data.get("knowledge_id", "") or ""),
        chunk_index=_int_or_none(data.get("chunk_index")),
        source=str(data.get("source", "") or ""),
        page=page,
        page_start=_int_or_none(data.get("page_start", page)),
        page_end=_int_or_none(data.get("page_end", page)),
        rerank_score=_float_or_none(data.get("rerank_score")),
        rewrite_query=data.get("rewrite_query"),
        text_preview=str(data.get("text_preview", "") or ""),
        retrieval=safe_retrieval_metadata(data.get("retrieval")),
    )


def _claim_audit_summary_from_mapping(item: Any) -> ClaimAuditSummary | None:
    data = _as_mapping(item)
    if not data:
        return None
    verifier = _as_mapping(data.get("verifier"))
    raw_counts = _as_mapping(data.get("counts"))
    raw_metrics = _as_mapping(data.get("metrics"))
    raw_repair = _as_mapping(data.get("repair"))
    count_keys = (
        "claim_count",
        "supported",
        "unsupported",
        "insufficient",
        "cited",
        "skipped_statements",
    )
    metric_keys = (
        "claim_support_rate",
        "citation_coverage",
        "unsupported_claim_rate",
    )
    return ClaimAuditSummary(
        status=str(data.get("status") or "not_run"),
        reason_code=str(data.get("reason_code") or ""),
        counts={
            key: _nonnegative_int(raw_counts.get(key, 0))
            for key in count_keys
            if key in raw_counts
        },
        metrics={
            key: (
                _float_or_none(raw_metrics.get(key))
                if raw_metrics.get(key) is not None
                else None
            )
            for key in metric_keys
            if key in raw_metrics
        },
        repair={
            "attempted": bool(raw_repair.get("attempted", False)),
            "attempt_count": _nonnegative_int(raw_repair.get("attempt_count", 0)),
            "succeeded": bool(raw_repair.get("succeeded", False)),
        },
        duration_ms=_float_or_none(
            data.get("duration_ms", verifier.get("duration_ms"))
        ),
    )


# 把对话结果转换成响应。
def chat_result_to_response(
    result: Any,
    *,
    doc_id: str,
    session_id: str | None = None,
) -> ChatResponse:
    # 防御式取值，且不暴露原始输出、步骤和证据全文。
    raw_output = _as_mapping(getattr(result, "raw_output", None))
    task_type = _normalize_task_type(getattr(result, "task_type", None))
    answer = str(getattr(result, "answer", "") or "")
    raw_evidence = getattr(result, "evidence", []) or []
    raw_ledger = getattr(result, "citation_ledger", [])
    ledger_evidence = raw_output.get("evidence_ledger") or raw_evidence
    ledger_validation = validate_public_citation_ledger(
        answer,
        raw_ledger,
        evidence=ledger_evidence,
        require_evidence=bool(raw_ledger),
    )
    public_ledger = ledger_validation.entries if ledger_validation.is_valid else ()
    critique = str(getattr(result, "critique", "") or "")
    if task_type in {"qa", "summary", "compare"} and (
        critique or not ledger_validation.is_valid
    ):
        critique = "回答未通过引用或声明证据校验。"
    return ChatResponse(
        request_id=str(getattr(result, "request_id", "") or ""),
        trace_id=str(getattr(result, "trace_id", "") or ""),
        doc_id=doc_id,
        session_id=session_id,
        task_type=task_type,
        answer=answer,
        citations=[
            _citation_from_mapping(item)
            for item in list(getattr(result, "citations", []) or [])
        ],
        citation_ledger=[
            _citation_ledger_entry_from_mapping(item) for item in public_ledger
        ],
        evidence=[_evidence_from_mapping(item) for item in list(raw_evidence)],
        critique=critique,
        is_valid=bool(getattr(result, "is_valid", False))
        and ledger_validation.is_valid,
        claim_audit=_claim_audit_summary_from_mapping(raw_output.get("claim_audit")),
    )


# 构建错误响应。
def build_error_response(
    error_code: ErrorCode,
    message: str,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error_code=error_code,
        message=message,
        request_id=request_id,
        trace_id=trace_id,
        details=details,
    )
