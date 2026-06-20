from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from agents.citation_validator import CitationValidatorAgent
from agents.generator import Generator
from graph.state import Evidence, RetrievedDoc, SummarySectionPlan, SummarySectionResult
from tools.tokenizer import tokenize_mixed_text


MAX_SECTION_CONTEXT_CHUNKS = 8


def tokenize_for_section(text: str) -> set[str]:
    # 复用 BM25 检索同等分词规则，避免中文单字交集噪声。
    return set(tokenize_mixed_text(text))


def select_section_docs(
    docs: List[RetrievedDoc],
    plan: SummarySectionPlan,
    query: str,
    doc_tokens: List[Tuple[RetrievedDoc, set[str]]] = None,
    max_chunks: int = MAX_SECTION_CONTEXT_CHUNKS,
) -> List[RetrievedDoc]:
    # 每个章节只喂相关 chunk，降低全文广播的 token 成本。
    if len(docs) <= max_chunks:
        return docs

    section_text = f"{query} {plan['title']} {plan['instruction']}"
    section_tokens = tokenize_for_section(section_text)
    doc_tokens = doc_tokens or [(doc, tokenize_for_section(doc["text"])) for doc in docs]

    scored = []
    for idx, (doc, tokens) in enumerate(doc_tokens):
        score = len(section_tokens & tokens)
        scored.append((score, idx, doc))

    selected = [item for item in scored if item[0] > 0]
    if not selected:
        selected = scored[:max_chunks]
    else:
        selected = sorted(selected, key = lambda item: (-item[0], item[1]))[:max_chunks]

    return [doc for _, _, doc in sorted(selected, key = lambda item: item[1])]


def format_summary_context(docs: List[RetrievedDoc]) -> str:
    # Summary prompt 沿用 Document 标签契约，确保 citation checker 可复用。
    blocks = []
    for doc in docs:
        meta = doc.get("meta", {})
        source = meta.get("source", "未知文件")
        page = meta.get("page", 1)
        chunk_id = meta.get("chunk_id", meta.get("chunk_index", 0))
        blocks.append(
            f'<Document source="{source}" page="{page}" chunk_id="{chunk_id}">\n'
            f'{doc["text"].strip()}\n'
            f'</Document>'
        )
    return "\n\n".join(blocks)


def build_summary_evidence(docs: List[RetrievedDoc]) -> List[Evidence]:
    # Summary evidence 暂先暴露参与摘要的全部单文档 chunk。
    return [
        Evidence(
            chunk_id = doc.get("meta", {}).get("chunk_id", ""),
            chunk_index = doc.get("meta", {}).get("chunk_index", -1),
            source = doc.get("meta", {}).get("source", ""),
            page = doc.get("meta", {}).get("page", 0),
            page_start = doc.get("meta", {}).get("page_start", doc.get("meta", {}).get("page", 0)),
            page_end = doc.get("meta", {}).get("page_end", doc.get("meta", {}).get("page", 0)),
            text_preview = doc["text"][:100],
        )
        for doc in docs
    ]


class SectionSummaryAgent:
    @staticmethod
    def summarize_sections(state: dict) -> dict:
        # 每个章节单独生成短摘要，降低单次输出结构漂移。
        docs: List[RetrievedDoc] = state.get("summary_docs", [])
        plans: List[SummarySectionPlan] = state.get("summary_section_plans", [])
        query = state.get("query", "")
        source = state.get("summary_source", "")
        is_local = state.get("is_local", False)

        if not docs or not plans:
            return {"summary_section_results": []}

        llm = Generator._get_client(is_local = is_local)
        results: List[SummarySectionResult] = []
        doc_tokens = [(doc, tokenize_for_section(doc["text"])) for doc in docs]

        for plan in plans:
            section_docs = select_section_docs(docs, plan, query, doc_tokens = doc_tokens)
            context = format_summary_context(section_docs)
            messages = [
                SystemMessage(
                    content = (
                        "你是一位严谨的技术文档摘要助手。你的唯一工作是：仅依据给定的 <Document> 标签文本，"
                        "为指定章节写一段简短的中文摘要，并为每个事实标注可被机器校验的精确引用。\n\n"
                        "【硬性约束】\n"
                        "1. 只能使用 <Document> 标签内的信息，禁止引入任何标签外的知识、常识或推测。\n"
                        "2. 每一句陈述文档事实的句子，句尾必须附加引用，格式为 [source:P页码]。\n"
                        "3. 引用里的 source 和 page 必须逐字复制对应 <Document> 标签的 source、page 属性，"
                        "不得改写文件名、不得使用占位词、不得引用标签中不存在的文件名或页码——"
                        "这些引用会被逐条机器校验，任何一处对不上都会导致整篇摘要被判废。\n"
                        "4. 同一句涉及多个来源时，可连续附加多个标签，例如 [a.pdf:P3][a.pdf:P5]。\n"
                        "5. 只能用半角方括号与冒号 []:，禁止中文括号（）和任何全角符号。\n\n"
                        "【范围与篇幅】\n"
                        "- 只写当前指定章节的内容，不复述其它章节、不重复章节标题、不加前言/结语/过渡语。\n"
                        "- 输出 2-4 句中文短句。\n\n"
                        "【无依据时】\n"
                        "- 若给定片段中确实找不到与本章节相关的内容，只输出一行：文档中未明确说明"
                        "（不加引用、不编造）。\n\n"
                        "【示例】\n"
                        "参考资料含 <Document source=\"赛事规程.pdf\" page=\"2\">……报名于 6 月 1 日截止……</Document> 时，"
                        "合格输出形如：\n"
                        "本赛事报名于 6 月 1 日截止，参赛团队须在此前完成组队与材料提交。[赛事规程.pdf:P2]\n\n"
                        "【输出】只输出摘要正文（或“文档中未明确说明”），"
                        "不要输出章节标题、不要解释、不要任何额外文字。"
                    )
                ),
                HumanMessage(
                    content = (
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
            response = llm.invoke(messages)
            results.append(
                SummarySectionResult(
                    section_id = plan["section_id"],
                    title = plan["title"],
                    content = response.content,
                )
            )

        return {"summary_section_results": results}


class GlobalSummaryAgent:
    @staticmethod
    def build_final_summary(state: dict) -> dict:
        # GlobalSummary 整合章节结果，并复用 CitationValidatorAgent 做最终审计。
        source = state.get("summary_source", "")
        docs: List[RetrievedDoc] = state.get("summary_docs", [])
        results: List[SummarySectionResult] = state.get("summary_section_results", [])

        if not docs:
            return {}
        if not results:
            answer = f"未能生成文档 {source} 的摘要章节。"
            return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

        lines = [f"# {source} 结构化摘要"]
        for result in results:
            lines.append(f"\n## {result['title']}\n{result['content']}")
        answer = "\n".join(lines)

        check_res = CitationValidatorAgent.validate_citations(answer, docs)
        if not check_res["is_valid"]:
            answer = (
                f"{answer}\n\n"
                "## 引用校验警告\n"
                "以下问题需要人工复核或重新生成对应章节：\n"
                f"{check_res['critique']}"
            )

        return {
            "answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "sources": [doc["meta"] for doc in docs],
            "evidence": build_summary_evidence(docs),
            "critique": check_res["critique"],
        }
