from typing import TypedDict, List, Optional, Annotated, Any
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

class RetrievalMetrics(TypedDict, total = False):
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
    role: str
    content: str
    timestamp: Optional[str]

class AgentStepTrace(TypedDict):
    step_name: str
    input_summary: str
    output_summary: str

class Evidence(TypedDict, total = False):
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

class GraphState(TypedDict):
    # GraphState 是 LangGraph 节点间传递的全局状态。
    messages: Annotated[List[BaseMessage], add_messages]
    docs: NotRequired[List[RetrievedDoc]]

    query: NotRequired[str]
    doc_id: NotRequired[str]
    is_local: NotRequired[bool]
    task_type: NotRequired[str]
    router_reason: NotRequired[str]
    top_k: NotRequired[int]

    rewritten_queries: NotRequired[List[str]]
    retrieved_docs: NotRequired[List[RetrievedDoc]]
    reranked_docs: NotRequired[List[RetrievedDoc]]
    context: NotRequired[str]

    answer: NotRequired[str]
    critique: NotRequired[str]
    sources: NotRequired[List[DocMeta]]
    evidence: NotRequired[List[Evidence]]

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

