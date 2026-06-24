import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from cogdoc.config.settings import get_settings
from cogdoc.agents.answer_markers import CITATION_WARNING_HEADING
from cogdoc.agents.citation_validator import CitationValidatorAgent
from cogdoc.agents.qa_generator import Generator
from cogdoc.graph.state import Evidence, RetrievedDoc, SummarySectionPlan, SummarySectionResult
from cogdoc.tools.tokenizer import tokenize_mixed_text


MAX_SECTION_CONTEXT_CHUNKS = 8
LOCAL_LARGE_SECTION_CONTEXT_CHUNKS = 6
CITATION_PATTERN = re.compile(
    r"[\[［]\s*([^:：\]］]+?)\s*[:：]\s*[pPｐＰ]?\s*([0-9０-９]+)\s*[\]］]"
)

# 云端逐单元 LLM 调用相互独立，并发执行降低 Summary/Compare 端到端延迟；本地走串行。
CLOUD_SECTION_MAX_WORKERS = get_settings().cloud_section_max_workers


def resolve_section_workers(is_local: bool, task_count: int) -> int:
    # 本地 Ollama 并发会放大显存/内存压力，退回串行；单任务无需起线程池。
    if task_count <= 1 or is_local:
        return 1
    return min(task_count, CLOUD_SECTION_MAX_WORKERS)


def run_section_cells(tasks: List, worker: Callable, is_local: bool) -> List:
    # ThreadPoolExecutor.map 按输入顺序返回结果，与完成先后无关，保证列序/章节序确定。
    workers = resolve_section_workers(is_local, len(tasks))
    if workers == 1:
        return [worker(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, tasks))


def tokenize_for_section(text: str) -> set[str]:
    return set(tokenize_mixed_text(text))


def select_section_docs(
    docs: List[RetrievedDoc],
    plan: SummarySectionPlan,
    query: str,
    doc_tokens: List[Tuple[RetrievedDoc, set[str]]] = None,
    max_chunks: int = MAX_SECTION_CONTEXT_CHUNKS,
) -> List[RetrievedDoc]:
    # 按维度召回相关 chunk，并保持原文顺序。
    if len(docs) <= max_chunks:
        return docs

    section_text = f"{query} {plan['title']} {plan['instruction']}"
    section_tokens = tokenize_for_section(section_text)
    doc_tokens = doc_tokens or [
        (doc, tokenize_for_section(doc["text"])) for doc in docs
    ]

    scored = []
    for idx, (doc, tokens) in enumerate(doc_tokens):
        score = len(section_tokens & tokens)
        scored.append((score, idx, doc))

    selected = [item for item in scored if item[0] > 0]
    if not selected:
        selected = scored[:max_chunks]
    else:
        selected = sorted(selected, key=lambda item: (-item[0], item[1]))[:max_chunks]

    return [doc for _, _, doc in sorted(selected, key=lambda item: item[1])]


def format_summary_context(docs: List[RetrievedDoc]) -> str:
    # Document 标签属性是 citation checker 的唯一来源。
    blocks = []
    for doc in docs:
        meta = doc.get("meta", {})
        source = meta.get("source", "未知文件")
        page = meta.get("page", 1)
        chunk_id = meta.get("chunk_id", meta.get("chunk_index", 0))
        blocks.append(
            f'<Document source="{source}" page="{page}" chunk_id="{chunk_id}">\n'
            f"{doc['text'].strip()}\n"
            f"</Document>"
        )
    return "\n\n".join(blocks)


def build_section_citations(docs: List[RetrievedDoc]) -> str:
    # 引用由实际使用的 chunk 元数据确定性生成。
    seen = set()
    citations = []
    for doc in docs:
        meta = doc.get("meta", {})
        source = meta.get("source", "")
        page = meta.get("page")
        key = (source, page)
        if not source or page is None or key in seen:
            continue
        seen.add(key)
        citations.append(f"[{source}:P{page}]")

    return "".join(citations)


NO_EVIDENCE_MARKER = "文档中未明确说明"
# 标记之后出现转折/补充词，说明仍陈述了实质内容，不能当无依据跳过引用。
_CONTENT_TURN_RE = re.compile(r"(但|不过|然而|另外|此外)")
# 句子终止符用于判断标记之后是否还另起了第二句实质内容。
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def is_no_evidence_summary(content: str) -> bool:
    # 只有单句无依据声明才跳过引用；转折或第二句都按实质内容绑定引用。
    normalized = content.strip()
    if not normalized.startswith(NO_EVIDENCE_MARKER):
        return False
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if len(sentences) > 1:
        return False
    return not _CONTENT_TURN_RE.search(normalized[len(NO_EVIDENCE_MARKER) :])


def attach_section_citations(content: str, docs: List[RetrievedDoc]) -> str:
    # 模型只写正文，程序统一替换为合法引用。
    content = content.strip()
    if not content or is_no_evidence_summary(content) or not docs:
        return content

    citations = build_section_citations(docs)
    if not citations:
        return content

    content = CITATION_PATTERN.sub("", content).strip()
    return f"{content}{citations}"


def all_contents_no_evidence(contents: Iterable[str]) -> bool:
    # 仅全为显式无依据声明时才允许跳过缺引用校验。
    contents = list(contents)
    return bool(contents) and all(
        is_no_evidence_summary(content) for content in contents
    )


def build_summary_evidence(docs: List[RetrievedDoc]) -> List[Evidence]:
    return [
        Evidence(
            chunk_id=doc.get("meta", {}).get("chunk_id", ""),
            chunk_index=doc.get("meta", {}).get("chunk_index", -1),
            source=doc.get("meta", {}).get("source", ""),
            page=doc.get("meta", {}).get("page", 0),
            page_start=doc.get("meta", {}).get(
                "page_start", doc.get("meta", {}).get("page", 0)
            ),
            page_end=doc.get("meta", {}).get(
                "page_end", doc.get("meta", {}).get("page", 0)
            ),
            text_preview=doc["text"][:100],
        )
        for doc in docs
    ]


def build_cell_evidence(content: str, docs: List[RetrievedDoc]) -> List[Evidence]:
    # 明确无依据的 cell/section 不展示支撑 chunk，避免 evidence 面板误导审计。
    if is_no_evidence_summary(content):
        return []
    return build_summary_evidence(docs)


def section_context_limit(is_local: bool, doc_count: int) -> int:
    # 本地模型按文档长度分档，避免长文档 4k context 截断，同时不过度牺牲摘要覆盖面。
    if is_local and doc_count > 16:
        return LOCAL_LARGE_SECTION_CONTEXT_CHUNKS
    return MAX_SECTION_CONTEXT_CHUNKS


def collect_evidence_items(
    evidence_groups: Iterable[Iterable[Evidence]],
    fallback_docs: List[RetrievedDoc],
    fallback_when_empty: bool = True,
) -> List[Evidence]:
    # 旧结果缺 evidence 字段时才回退全文 docs。
    seen = set()
    evidence: List[Evidence] = []

    for group in evidence_groups:
        for item in group:
            key = (
                item.get("chunk_id", ""),
                item.get("source", ""),
                item.get("page_start", item.get("page", 0)),
                item.get("page_end", item.get("page", 0)),
                item.get("chunk_index", -1),
            )
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)

    if evidence:
        return evidence
    if fallback_when_empty:
        return build_summary_evidence(fallback_docs)
    return []


def collect_section_evidence(
    results: List[SummarySectionResult],
    fallback_docs: List[RetrievedDoc],
) -> List[Evidence]:
    return collect_evidence_items(
        (result.get("evidence", []) for result in results),
        fallback_docs,
        fallback_when_empty=not any("evidence" in result for result in results),
    )


def append_citation_warning(answer: str, critique: str, unit_label: str) -> str:
    return (
        f"{answer}\n\n"
        f"{CITATION_WARNING_HEADING}\n"
        f"以下问题需要人工复核或重新生成对应{unit_label}：\n"
        f"{critique}"
    )


def _summary_messages(source: str, plan: SummarySectionPlan, query: str, context: str):
    return [
        SystemMessage(
            content=(
                "你是一位严谨的技术文档摘要助手。你的唯一工作是：仅依据给定的 <Document> 标签文本，"
                "为指定章节写一段简短的中文摘要。\n\n"
                "【硬性约束】\n"
                "1. 只能使用 <Document> 标签内的信息，禁止引入任何标签外的知识、常识或推测。\n"
                "2. 不要输出引用标签、页码、文件名或 <Document> 标签，程序会在生成后自动绑定引用。\n"
                "3. 不要使用占位词，不要输出章节标题，不要解释规则。\n\n"
                "【范围与篇幅】\n"
                "- 章节名是内容归类维度，不代表文档必须是论文；通知、规程、赛事规则也要按最接近维度归纳。\n"
                "- 只写当前指定章节的内容，不复述其它章节、不重复章节标题、不加前言/结语/过渡语。\n"
                "- 输出 2-4 句中文短句。\n\n"
                "【无依据时】\n"
                "- 只有当所有给定片段都没有任何可归入本章节的信息时，才输出一行：文档中未明确说明"
                "（不加引用、不编造）。\n"
                "- 若片段中有目标、对象、赛制、模块、要求、评分、奖项、日程、注意事项等赛事/通知信息，"
                "必须按章节聚焦摘要，不要因为不是论文实验而输出“文档中未明确说明”。\n\n"
                "【输出】只输出摘要正文（或“文档中未明确说明”），"
                "不要输出章节标题、不要解释、不要任何额外文字。"
            )
        ),
        HumanMessage(
            content=(
                f"【目标文档】{source}\n"
                f"【本次章节】{plan['title']}\n"
                f"【章节聚焦】{plan['instruction']}\n"
                f"【用户摘要意图】{query}\n\n"
                f"下面是从该文档中筛选出的、与本章节最相关的片段：\n"
                f"【参考资料开始】\n{context}\n【参考资料结束】\n\n"
                "请据此写出本章节摘要，2-4 句，严格遵守上面的全部约束。"
            )
        ),
    ]


def _summary_retry_messages(
    source: str, plan: SummarySectionPlan, query: str, context: str
):
    return [
        SystemMessage(
            content=(
                "你是一位严谨的中文资料整理助手。下面片段已经由程序筛选为与指定摘要维度相关。"
                "你的任务是从片段中提炼事实，不要因为文档不是论文而回答“文档中未明确说明”。\n\n"
                "【要求】\n"
                "1. 只能依据 <Document> 标签内文字。\n"
                "2. 若能找到任何与该维度相关的目标、对象、流程、规则、要求、评分、奖项、日程、注意事项或价值，"
                "就写 2-3 句中文摘要。\n"
                "3. 不要输出章节标题、引用、页码或解释。\n"
                "4. 只有片段完全没有相关事实时，才输出：文档中未明确说明。"
            )
        ),
        HumanMessage(
            content=(
                f"【目标文档】{source}\n"
                f"【摘要维度】{plan['title']}\n"
                f"【维度说明】{plan['instruction']}\n"
                f"【用户意图】{query}\n\n"
                f"【参考资料开始】\n{context}\n【参考资料结束】\n\n"
                "请重新提炼该维度摘要。"
            )
        ),
    ]


def generate_section_cell(
    llm,
    source: str,
    plan: SummarySectionPlan,
    docs: List[RetrievedDoc],
    query: str,
    is_local: bool,
    doc_tokens: List[Tuple[RetrievedDoc, set[str]]],
    build_messages: Callable[[str, SummarySectionPlan, str, str], list],
    build_retry_messages: Callable[[str, SummarySectionPlan, str, str], list] = None,
    max_chunks: int = None,
) -> Tuple[str, List[Evidence]]:
    # Summary 和 Compare 共用单元生成、引用绑定与 evidence 契约。
    max_chunks = max_chunks or section_context_limit(is_local, len(docs))
    section_docs = select_section_docs(
        docs, plan, query, doc_tokens=doc_tokens, max_chunks=max_chunks
    )
    context = format_summary_context(section_docs)
    content = llm.invoke(build_messages(source, plan, query, context)).content

    if (
        is_local
        and section_docs
        and is_no_evidence_summary(content)
        and build_retry_messages is not None
    ):
        content = llm.invoke(build_retry_messages(source, plan, query, context)).content

    content = attach_section_citations(content, section_docs)
    return content, build_cell_evidence(content, section_docs)


class SectionSummaryAgent:
    @staticmethod
    def summarize_sections(state: dict) -> dict:
        docs: List[RetrievedDoc] = state.get("summary_docs", [])
        plans: List[SummarySectionPlan] = state.get("summary_section_plans", [])
        query = state.get("query", "")
        source = state.get("summary_source", "")
        is_local = state.get("is_local", False)

        if not docs or not plans:
            return {"summary_section_results": []}

        llm = Generator._get_client(is_local=is_local)
        doc_tokens = [(doc, tokenize_for_section(doc["text"])) for doc in docs]

        def build_section_result(plan: SummarySectionPlan) -> SummarySectionResult:
            content, evidence = generate_section_cell(
                llm,
                source,
                plan,
                docs,
                query,
                is_local,
                doc_tokens,
                _summary_messages,
                _summary_retry_messages,
            )
            return SummarySectionResult(
                section_id=plan["section_id"],
                title=plan["title"],
                content=content,
                evidence=evidence,
            )

        results = run_section_cells(plans, build_section_result, is_local)
        return {"summary_section_results": results}


class GlobalSummaryAgent:
    @staticmethod
    def build_final_summary(state: dict) -> dict:
        source = state.get("summary_source", "")
        docs: List[RetrievedDoc] = state.get("summary_docs", [])
        results: List[SummarySectionResult] = state.get("summary_section_results", [])

        if not docs:
            return {}
        if not results:
            answer = f"未能生成文档 {source} 的摘要章节。"
            return {
                "answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }

        lines = [f"# {source} 结构化摘要"]
        for result in results:
            lines.append(f"\n## {result['title']}\n{result['content']}")
        answer = "\n".join(lines)

        # 只有全无依据摘要可跳过缺引用校验。
        critique = ""
        section_contents = [result.get("content", "") for result in results]
        if not all_contents_no_evidence(section_contents):
            check_res = CitationValidatorAgent.validate_citations(answer, docs)
            if not check_res["is_valid"]:
                critique = check_res["critique"]
                answer = append_citation_warning(answer, critique, "章节")

        return {
            "answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "sources": [doc["meta"] for doc in docs],
            "evidence": collect_section_evidence(results, docs),
            "critique": critique,
        }
