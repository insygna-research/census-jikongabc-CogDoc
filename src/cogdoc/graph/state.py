from typing import TypedDict, List, Optional, Annotated, Any, Dict
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_lists(old_list: Optional[Any], new_list: Optional[Any]) -> List[Any]:
    # 空列表表示显式清空历史。
    if new_list is not None and len(new_list) == 0:
        return []
    old = list(old_list) if old_list is not None else []
    new = list(new_list) if new_list is not None else []
    return old + new


class DocMeta(TypedDict):
    # chunk_id 是检索链路里的稳定身份键。
    chunk_id: str
    source_sha256: str
    local_chunk_index: int
    chunk_index: int
    source: str
    page: int
    page_start: int
    page_end: int
    score: float
    origin: str
    context: NotRequired[str]


class RetrievalMetrics(TypedDict, total=False):
    # retrieval 只保存本次检索产生的动态指标。
    distance: float
    bm25_score: float
    rrf_score: float
    rerank_score: float
    search_channel: str
    rewrite_query: str


class RetrievedDoc(TypedDict):
    # RetrievedDoc 是检索、重排和生成节点共享的文档结构。
    text: str
    meta: DocMeta
    retrieval: NotRequired[RetrievalMetrics]


class ChatMessage(TypedDict):
    # ChatMessage 保存会话历史中的单条消息。
    role: str
    content: str
    timestamp: Optional[str]


class AgentStepTrace(TypedDict):
    # AgentStepTrace 保存图节点的输入输出摘要。
    step_name: str
    input_summary: str
    output_summary: str


class Evidence(TypedDict, total=False):
    # Evidence 面向前端和审计展示。
    chunk_id: str
    chunk_index: int
    source: str
    page: int
    page_start: int
    page_end: int
    rerank_score: Optional[float]
    rewrite_query: Optional[str]
    text_preview: str


class SummarySectionPlan(TypedDict):
    # SummarySectionPlan 定义单文档摘要的固定章节。
    section_id: str
    title: str
    instruction: str


class SummarySectionResult(TypedDict):
    # SummarySectionResult 保存单个章节的带引用摘要。
    section_id: str
    title: str
    content: str
    evidence: NotRequired[List[Evidence]]


class CompareDimensionPlan(TypedDict):
    # CompareDimensionPlan 定义多文档对比的固定维度。
    dimension_id: str
    title: str
    instruction: str


class CompareCell(TypedDict):
    # CompareCell 保存某篇文档在某个对比维度下的带引用短描述。
    dimension_id: str
    source: str
    content: str
    evidence: NotRequired[List[Evidence]]


class DocumentProfile(TypedDict):
    # DocumentProfile 是 Compare 子图里单篇文档的结构化画像。
    source: str
    cells: List[CompareCell]


class GraphState(TypedDict):
    # GraphState 是 LangGraph 节点间传递的全局状态。
    messages: Annotated[List[BaseMessage], add_messages]
    docs: NotRequired[List[RetrievedDoc]]

    query: NotRequired[str]
    doc_id: NotRequired[str]
    request_id: NotRequired[str]
    trace_id: NotRequired[str]
    is_local: NotRequired[bool]
    task_type: NotRequired[str]
    router_reason: NotRequired[str]
    top_k: NotRequired[int]

    rewritten_queries: NotRequired[List[str]]
    rewrite_similarity_threshold: NotRequired[float]
    retrieved_docs: NotRequired[List[RetrievedDoc]]
    reranked_docs: NotRequired[List[RetrievedDoc]]
    context: NotRequired[str]

    answer: NotRequired[str]
    critique: NotRequired[str]
    sources: NotRequired[List[DocMeta]]
    evidence: NotRequired[List[Evidence]]
    summary_source: NotRequired[str]
    summary_docs: NotRequired[List[RetrievedDoc]]
    summary_section_plans: NotRequired[List[SummarySectionPlan]]
    summary_section_results: NotRequired[List[SummarySectionResult]]
    compare_sources: NotRequired[List[str]]
    compare_docs_by_source: NotRequired[Dict[str, List[RetrievedDoc]]]
    compare_dimensions: NotRequired[List[CompareDimensionPlan]]
    document_profiles: NotRequired[List[DocumentProfile]]
    compare_table_answer: NotRequired[str]
    compare_conclusion: NotRequired[str]
    compare_conclusion_warning: NotRequired[str]

    chat_history: NotRequired[Annotated[List[ChatMessage], merge_lists]]

    route: NotRequired[str]
    iteration_count: NotRequired[int]
    max_iteration_count: NotRequired[int]

    steps_trace: NotRequired[Annotated[List[AgentStepTrace], merge_lists]]

    error: NotRequired[Optional[str]]


class ParsedPage(TypedDict):
    # ParsedPage 是 PDF 解析后的页级输入。
    page: int
    source: str
    text: str
    is_ocr_fallback: bool
