from typing import TypedDict, List, Optional, Annotated, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# LangGraph状态合并函数
def merge_lists(old_list: Optional[Any], new_list: Optional[Any]) -> List[Any]:
    if new_list is not None and len(new_list) == 0:
        return []   
    old = list(old_list) if old_list is not None else []
    new = list(new_list) if new_list is not None else []
    return old + new

# 文档Chunk元数据
class DocMeta(TypedDict):
    chunk_index: int                # 当前Chunk在原文中的序号
    source: str                     # 来源文件名
    page: int                       # 主页码
    page_start: int                 # 起始页码
    page_end: int                   # 终止页码
    score: float                    # 检索/Rerank相关度得分
    origin: str                     # 来源渠道(vector/bm25/web/...)

class RetrievalMetrics(TypedDict, total = False):
    distance: float                 # 向量密集检索距离（越小越相关）
    bm25_score: float               # 稀疏检索词频得分（越大越相关）
    rrf_score: float                # 混合检索倒数排名融合得分
    rerank_score: float             # 交叉编码器精排模型最终得分
    search_channel: str             # 动态召回渠道标记 ("vector" / "bm25" / "hybrid")

# 检索得到的文档片段
class RetrievedDoc(TypedDict):
    text: str                                   # Chunk正文
    meta: DocMeta                               # Chunk元数据
    retrieval: NotRequired[RetrievalMetrics]    # Chunk动态检索指标


# 消息
class ChatMessage(TypedDict):
    role: str                   # 消息发送方角色（user / assistant / system）
    content: str                # 消息内容
    timestamp: Optional[str]    # 消息创建时间

# Agent运行轨迹
class AgentStepTrace(TypedDict):
    step_name: str          # 当前执行的节点
    input_summary: str      # 输入内容摘要
    output_summary: str     # 输出内容摘要

# LangGraph全局状态
class GraphState(TypedDict):
    # LangGraph 消息流（当前工作流实际使用的字段）
    messages: Annotated[List[BaseMessage], add_messages]
    docs: NotRequired[List[RetrievedDoc]]

    # 外部输入
    query: NotRequired[str]         # 用户问题
    doc_id: NotRequired[str]        # 当前文档ID或知识库ID
    is_local: NotRequired[bool]     # 是否使用本地Ollama模型
    task_type: NotRequired[str]     # qa/summary/compare
    router_reason: NotRequired[str]   # 当前任务分类的文本理由
    top_k: NotRequired[int]        # 检索返回数量

    # 核心RAG管道中间态
    rewritten_queries: NotRequired[List[str]]             # Query Rewrite结果
    retrieved_docs: NotRequired[List[RetrievedDoc]]     # 混合检索召回结果
    reranked_docs: NotRequired[List[RetrievedDoc]]       # Reranker排序后结果
    context: NotRequired[str]                          # 最终送入LLM的上下文

    # 最终输出与溯源
    answer: NotRequired[str]                 # 最终回答
    critique: NotRequired[str]               # Critic反思结果
    sources: NotRequired[List[DocMeta]]      # 答案来源

    # 历史
    chat_history: NotRequired[Annotated[List[ChatMessage], merge_lists]]

    # Agent编排与控制流
    route: NotRequired[str]                  # 下一步执行节点
    iteration_count: NotRequired[int]        # 当前反思次数
    max_iteration_count: NotRequired[int]    # 最大允许反思次数

    # Agent执行轨迹
    steps_trace: NotRequired[Annotated[List[AgentStepTrace], merge_lists]]

    # 异常
    error: NotRequired[Optional[str]]

class ParsedPage(TypedDict):
    page: int                # 页码
    source: str              # 来源文件名
    text: str                # 当前页解析出的正文
    is_ocr_fallback: bool    # 当前页面是否是需要OCR才能读取的页面（非正常文本）





