from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import get_settings


API_SCHEMA_VERSION = "v1"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ChatMode(str, Enum):
    AUTO = "auto"
    QA = "qa"
    SUMMARY = "summary"
    COMPARE = "compare"


class ChatTask(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    COMPARE = "compare"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    CITATION_REJECTED = "CITATION_REJECTED"
    NO_EVIDENCE = "NO_EVIDENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    STREAM_INTERRUPTED = "STREAM_INTERRUPTED"
    UNKNOWN_INTENT = "UNKNOWN_INTENT"
    BAD_REQUEST = "BAD_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    KB_NOT_FOUND = "KB_NOT_FOUND"
    KB_EXISTS = "KB_EXISTS"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INVALID_PDF = "INVALID_PDF"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    INGEST_FAILED = "INGEST_FAILED"


class ChatRequest(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    query: str = Field(min_length=1)
    doc_id: str = Field(default_factory=lambda: get_settings().cogdoc_default_doc_id)
    session_id: str | None = None
    mode: ChatMode = ChatMode.AUTO
    is_local: bool = False

    @field_validator("query", "doc_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @property
    def forced_task(self) -> str | None:
        return None if self.mode == ChatMode.AUTO else str(self.mode)


class Citation(ApiModel):
    chunk_id: str = ""
    source: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None


class Evidence(ApiModel):
    chunk_id: str = ""
    chunk_index: int | None = None
    source: str = ""
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    rerank_score: float | None = None
    rewrite_query: str | None = None
    text_preview: str = ""


class ChatResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    request_id: str
    trace_id: str
    doc_id: str
    session_id: str | None = None
    task_type: ChatTask
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    critique: str = ""
    is_valid: bool


class ErrorResponse(ApiModel):
    schema_version: Literal["v1"] = API_SCHEMA_VERSION
    error_code: ErrorCode
    message: str
    request_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class KnowledgeBaseCreate(ApiModel):
    # 上限 56：VectorRetriever 把 collection 名截到 f"col-{kb_id}"[:60]，超长会撞库。
    kb_id: str = Field(min_length=1, max_length=56)

    @field_validator("kb_id")
    @classmethod
    def _slug(cls, value: str) -> str:
        # kb_id 进路径，禁止分隔符与空白，避免目录穿越。
        stripped = value.strip()
        if not stripped or any(c in stripped for c in "/\\ \t") or stripped in {".", ".."}:
            raise ValueError("kb_id 只能是不含路径分隔符与空白的标识符")
        return stripped


class KnowledgeBase(ApiModel):
    kb_id: str
    created_at: str
    document_count: int = 0
    tenant_id: str = "default"
    owner_id: str = "default"


class Document(ApiModel):
    name: str
    sha256: str = ""


class IndexJob(ApiModel):
    job_id: str
    kb_id: str
    status: JobStatus
    created_at: str
    finished_at: str | None = None
    document_count: int | None = None
    chunk_count: int | None = None
    error_code: ErrorCode | None = None
    message: str | None = None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_task_type(value: Any) -> str:
    task = str(value or ChatTask.UNKNOWN.value)
    return task if task in {item.value for item in ChatTask} else ChatTask.UNKNOWN.value


def _citation_from_mapping(item: Any) -> Citation:
    data = _as_mapping(item)
    page = _int_or_none(data.get("page"))
    return Citation(
        chunk_id=str(data.get("chunk_id", "") or ""),
        source=str(data.get("source", "") or ""),
        page=page,
        page_start=_int_or_none(data.get("page_start", page)),
        page_end=_int_or_none(data.get("page_end", page)),
    )


def _evidence_from_mapping(item: Any) -> Evidence:
    data = _as_mapping(item)
    page = _int_or_none(data.get("page"))
    return Evidence(
        chunk_id=str(data.get("chunk_id", "") or ""),
        chunk_index=_int_or_none(data.get("chunk_index")),
        source=str(data.get("source", "") or ""),
        page=page,
        page_start=_int_or_none(data.get("page_start", page)),
        page_end=_int_or_none(data.get("page_end", page)),
        rerank_score=_float_or_none(data.get("rerank_score")),
        rewrite_query=data.get("rewrite_query"),
        text_preview=str(data.get("text_preview", "") or ""),
    )


def chat_result_to_response(
    result: Any,
    *,
    doc_id: str,
    session_id: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        request_id=str(getattr(result, "request_id", "") or ""),
        trace_id=str(getattr(result, "trace_id", "") or ""),
        doc_id=doc_id,
        session_id=session_id,
        task_type=_normalize_task_type(getattr(result, "task_type", None)),
        answer=str(getattr(result, "answer", "") or ""),
        citations=[
            _citation_from_mapping(item)
            for item in list(getattr(result, "citations", []) or [])
        ],
        evidence=[
            _evidence_from_mapping(item)
            for item in list(getattr(result, "evidence", []) or [])
        ],
        critique=str(getattr(result, "critique", "") or ""),
        is_valid=bool(getattr(result, "is_valid", False)),
    )


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
