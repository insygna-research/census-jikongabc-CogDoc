from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from agents.citation_validator import CitationValidatorAgent
from agents.qa_generator import Generator
from agents.summary_generator import all_contents_no_evidence, append_citation_warning, collect_evidence_items
from graph.state import CompareDimensionPlan, DocumentProfile, RetrievedDoc


def _profile_lookup(profiles: List[DocumentProfile]) -> Dict[Tuple[str, str], str]:
    # 展平成 (文档, 维度) 索引，输出阶段按用户点名顺序取值。
    lookup = {}
    for profile in profiles:
        source = profile["source"]
        for cell in profile.get("cells", []):
            lookup[(source, cell["dimension_id"])] = cell["content"]
    return lookup


def _union_compare_docs(docs_by_source: Dict[str, List[RetrievedDoc]], sources: List[str]) -> List[RetrievedDoc]:
    # 引用校验必须拿到所有参与对比文档的原始 chunk。
    docs: List[RetrievedDoc] = []
    for source in sources:
        docs.extend(docs_by_source.get(source, []))
    return docs


def _format_compare_blocks(
    sources: List[str],
    dimensions: List[CompareDimensionPlan],
    profiles: List[DocumentProfile],
) -> str:
    # 长文本对比使用分块列表，避免 Markdown 宽表在终端错位。
    lookup = _profile_lookup(profiles)
    lines = ["# 多文档对比"]

    for dimension in dimensions:
        lines.append(f"\n## {dimension['title']}")
        for source in sources:
            content = lookup.get((source, dimension["dimension_id"]), "文档中未明确说明")
            content = str(content or "文档中未明确说明").replace("\n", " ").strip()
            lines.append(f"- **{source}**：{content}")

    return "\n".join(lines)


class CompareGeneratorAgent:
    @staticmethod
    def build_compare_answer(state: dict) -> dict:
        sources: List[str] = state.get("compare_sources", [])
        dimensions: List[CompareDimensionPlan] = state.get("compare_dimensions", [])
        profiles: List[DocumentProfile] = state.get("document_profiles", [])
        docs_by_source: Dict[str, List[RetrievedDoc]] = state.get("compare_docs_by_source", {})

        if len(sources) < 2 or not dimensions or not profiles:
            answer = "未能生成对比表：请在问题中点名至少 2 篇可用文档。"
            return {"answer": answer}

        table_answer = _format_compare_blocks(sources, dimensions, profiles)
        conclusion_warning = ""
        if state.get("is_local", False):
            # 本地模式跳过额外结论生成，避免再次触发 Ollama 内存加载。
            conclusion = ""
            conclusion_warning = "本地 Ollama 模式已跳过简短结论生成，以降低内存占用。"
        else:
            try:
                conclusion = CompareGeneratorAgent._generate_conclusion(state, table_answer)
            except Exception as exc:
                conclusion = ""
                conclusion_warning = f"结论生成失败，已降级为纯表格：{type(exc).__name__}: {exc}"
        answer = table_answer
        if conclusion:
            answer = f"{table_answer}\n\n## 简短结论\n{conclusion}"

        union_docs = _union_compare_docs(docs_by_source, sources)
        steps_trace = [
            {
                "step_name": "compare_blocks",
                "input_summary": "，".join(sources),
                "output_summary": f"{len(dimensions)} dimensions x {len(sources)} documents",
            }
        ]
        if conclusion_warning:
            steps_trace.append(
                {
                    "step_name": "compare_conclusion_warning",
                    "input_summary": "LLM conclusion",
                    "output_summary": conclusion_warning,
                }
            )

        return {
            "compare_table_answer": table_answer,
            "compare_conclusion": conclusion,
            "compare_conclusion_warning": conclusion_warning,
            "answer": answer,
            "sources": [doc["meta"] for doc in union_docs],
            # 只汇总 cell 真实证据；旧状态缺 evidence 字段时才回退到全文 chunk。
            "evidence": collect_evidence_items(
                (
                    cell.get("evidence", [])
                    for profile in profiles
                    for cell in profile.get("cells", [])
                ),
                union_docs,
                fallback_when_empty = not any(
                    "evidence" in cell
                    for profile in profiles
                    for cell in profile.get("cells", [])
                ),
            ),
            "steps_trace": steps_trace,
        }

    @staticmethod
    def _generate_conclusion(state: dict, table_answer: str) -> str:
        # 结论只能复用已带引用的对比块，生成后还会单独过引用校验。
        is_local = state.get("is_local", False)
        query = state.get("query", "")

        llm = Generator._get_client(is_local = is_local)
        messages = [
            SystemMessage(
                content = (
                    "你是一位严谨的技术方案对比助手。只能依据用户提供的 Markdown 对比内容写结论。\n\n"
                    "【硬性约束】\n"
                    "1. 只能复用对比条目中已经出现的事实，禁止引入新事实、新指标、新评价。\n"
                    "2. 每一句结论都必须带有对比条目中已出现的引用标签，引用格式保持原样。\n"
                    "3. 输出 2-3 句中文短句，不要输出标题、列表、解释或额外文字。"
                )
            ),
            HumanMessage(
                content = (
                    f"【用户对比意图】{query}\n\n"
                    f"【对比内容开始】\n{table_answer}\n【对比内容结束】\n\n"
                    "请基于上述内容写简短结论。"
                )
            ),
        ]
        return llm.invoke(messages).content.strip()

    @staticmethod
    def validate_compare_answer(state: dict) -> dict:
        # 结论单独校验，避免表格引用掩盖无引用结论。
        sources: List[str] = state.get("compare_sources", [])
        docs_by_source: Dict[str, List[RetrievedDoc]] = state.get("compare_docs_by_source", {})
        profiles: List[DocumentProfile] = state.get("document_profiles", [])
        union_docs = _union_compare_docs(docs_by_source, sources)
        table_answer = state.get("compare_table_answer", state.get("answer", ""))
        conclusion = state.get("compare_conclusion", "")
        answer = state.get("answer", table_answer)
        cell_contents = [
            cell.get("content", "")
            for profile in profiles
            for cell in profile.get("cells", [])
        ]

        critiques = []
        if conclusion:
            # 先查结论，防止表格里的合法引用掩盖结论缺引用。
            conclusion_check = CitationValidatorAgent.validate_citations(conclusion, union_docs)
            if not conclusion_check["is_valid"]:
                critiques.append(conclusion_check["critique"])
                answer = table_answer

        # 只有全无依据 cell 可跳过缺引用校验。
        if not all_contents_no_evidence(cell_contents):
            # 结论失败后仍校验纯对比块，避免单元格错误被提前降级吞掉。
            table_or_answer = table_answer if critiques else answer
            full_check = CitationValidatorAgent.validate_citations(table_or_answer, union_docs)
            if not full_check["is_valid"]:
                critiques.append(full_check["critique"])
                answer = table_answer

        critique = "\n\n".join(critiques)
        if critique:
            answer = append_citation_warning(answer, critique, "单元格")

        return {
            "answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "critique": critique,
        }
