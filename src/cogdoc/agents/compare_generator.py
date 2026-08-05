import re
from typing import Any, Dict, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from cogdoc.agents.claim_evidence_verifier import (
    CLAIM_AUDIT_EXEMPTION_GUIDANCE,
    CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
    make_claim_audit_exemption,
)
from cogdoc.agents.compare_profile import ensure_compare_evidence_ids
from cogdoc.agents.no_evidence import is_no_evidence_statement
from cogdoc.agents.qa_generator import Generator
from cogdoc.agents.summary_generator import (
    EVIDENCE_UNIT_FAILURE_MESSAGE,
    append_citation_warning,
    collect_evidence_items,
)
from cogdoc.graph.state import (
    CompareCell,
    CompareDimensionPlan,
    DocumentProfile,
    RetrievedDoc,
)
from cogdoc.service.claim_audit_projection import (
    ClaimAuditProjectionSegment,
    build_claim_audit_projection,
)
from cogdoc.tools.citation_ledger import (
    extract_evidence_ids,
    validate_evidence_citations,
)


COMPARE_CONCLUSION_SYSTEM_PROMPT = (
    "你是一位严谨的技术方案对比助手。只能依据用户提供的 Markdown 对比内容写结论。\n\n"
    "【硬性约束】\n1. 只能复用对比条目中已经出现的事实，禁止引入新事实、新指标、新评价。\n"
    "2. 每一句结论都必须带有对比条目中已出现的 Evidence ID，严格保持 [E001] 格式和原编号。\n"
    "3. 输出 2-3 句中文短句，不要输出标题、列表、解释或额外文字。"
)
COMPARE_CONCLUSION_USER_PROMPT_TEMPLATE = (
    "【用户对比意图】{query}\n\n【对比内容开始】\n{table_answer}\n"
    "【对比内容结束】\n\n请基于上述内容写简短结论。"
)
_EVIDENCE_AFTER_TERMINATOR_RE = re.compile(
    r"([。！？!?；;])((?:\[E[0-9]{3,}\])+)(?=\s|$|[^\[])"
)


def _normalize_evidence_citation_placement(content: str) -> str:
    # 引用放在终止符前，保证 claim splitter 不会把下一句与上一句的 ID 合并。
    return _EVIDENCE_AFTER_TERMINATOR_RE.sub(r"\2\1", content)


# 完成 画像查询表 处理。
def _profile_lookup(profiles: List[DocumentProfile]) -> Dict[Tuple[str, str], str]:
    # 展平成 (文档, 维度) 索引，输出阶段按用户点名顺序取值。
    lookup = {}
    for profile in profiles:
        source = profile["source"]
        for cell in profile.get("cells", []):
            lookup[(source, cell["dimension_id"])] = cell["content"]
    return lookup


# 合并对比文档列表。
def _union_compare_docs(
    docs_by_source: Dict[str, List[RetrievedDoc]], sources: List[str]
) -> List[RetrievedDoc]:
    # 引用校验必须拿到所有参与对比文档的原始 chunk。
    docs: List[RetrievedDoc] = []
    for source in sources:
        docs.extend(docs_by_source.get(source, []))
    return docs


# 格式化 compare blocks。
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
            content = lookup.get(
                (source, dimension["dimension_id"]), EVIDENCE_UNIT_FAILURE_MESSAGE
            )
            content = (
                str(content or EVIDENCE_UNIT_FAILURE_MESSAGE).replace("\n", " ").strip()
            )
            lines.append(f"- **{source}**：{content}")

    return "\n".join(lines)


def _profile_cells(profiles: List[DocumentProfile]) -> List[CompareCell]:
    return [cell for profile in profiles for cell in profile.get("cells", [])]


def _is_no_evidence_cell(cell: CompareCell) -> bool:
    return cell.get("status") == "no_evidence" or is_no_evidence_statement(
        cell.get("content")
    )


def _is_generated_cell(cell: CompareCell) -> bool:
    content = str(cell.get("content") or "").strip()
    return (
        cell.get("status") in (None, "", "generated")
        and bool(content)
        and content != EVIDENCE_UNIT_FAILURE_MESSAGE
        and not _is_no_evidence_cell(cell)
    )


def _compare_claim_audit_projection(
    answer: str,
    sources: List[str],
    dimensions: List[CompareDimensionPlan],
    profiles: List[DocumentProfile],
    conclusion: str,
) -> dict[str, Any]:
    """Project cells in their final dimension-major render order."""

    resolved_sources = sources or [
        source
        for profile in profiles
        if (source := str(profile.get("source") or ""))
    ]
    resolved_dimension_ids = [
        str(dimension.get("dimension_id") or "") for dimension in dimensions
    ] or list(
        dict.fromkeys(
            dimension_id
            for profile in profiles
            for cell in profile.get("cells", [])
            if (dimension_id := str(cell.get("dimension_id") or ""))
        )
    )
    cell_lookup = {
        (str(cell.get("source") or profile.get("source") or ""), dimension_id): cell
        for profile in profiles
        for cell in profile.get("cells", [])
        if (dimension_id := str(cell.get("dimension_id") or ""))
    }
    segments: list[ClaimAuditProjectionSegment] = []
    for dimension_id in resolved_dimension_ids:
        for source in resolved_sources:
            cell = cell_lookup.get((source, dimension_id))
            raw_content = (
                cell.get("content") if cell is not None else EVIDENCE_UNIT_FAILURE_MESSAGE
            )
            content = str(raw_content or EVIDENCE_UNIT_FAILURE_MESSAGE).replace(
                "\n", " "
            ).strip()
            segment_id = f"compare:cell:{source}:{dimension_id}"
            unit_id = str(cell.get("unit_id") or "").strip() if cell else ""
            obligation_ids = (unit_id,) if unit_id else ()
            source_status = (
                str(cell.get("status") or "legacy_generated")
                if cell is not None
                else "missing"
            )
            if cell is not None and _is_no_evidence_cell(cell):
                segment = ClaimAuditProjectionSegment.deterministic(
                    segment_id,
                    content,
                    source_status=source_status,
                    obligation_ids=obligation_ids,
                )
            elif cell is not None and _is_generated_cell(cell):
                segment = ClaimAuditProjectionSegment.generated(
                    segment_id,
                    content,
                    source_status=source_status,
                    obligation_ids=obligation_ids,
                )
            else:
                segment = ClaimAuditProjectionSegment.operational(
                    segment_id,
                    content,
                    source_status=source_status,
                    obligation_ids=obligation_ids,
                )
            segments.append(segment)

    normalized_conclusion = str(conclusion or "").strip()
    if normalized_conclusion and normalized_conclusion in answer:
        segments.append(
            ClaimAuditProjectionSegment.generated(
                "compare:conclusion",
                normalized_conclusion,
                source_status="generated",
            )
        )
    return build_claim_audit_projection(answer, segments).to_state()


def _missing_cell_count(
    sources: List[str],
    dimensions: List[CompareDimensionPlan],
    profiles: List[DocumentProfile],
) -> int:
    expected = {
        (source, dimension["dimension_id"])
        for source in sources
        for dimension in dimensions
    }
    actual = {
        (
            str(cell.get("source") or profile.get("source") or ""),
            str(cell.get("dimension_id") or ""),
        )
        for profile in profiles
        for cell in profile.get("cells", [])
    }
    return len(expected - actual)


# 定义 CompareGeneratorAgent 数据结构。
class CompareGeneratorAgent:
    # 构建 compare answer。
    @staticmethod
    def build_compare_answer(state: dict) -> dict:
        sources: List[str] = state.get("compare_sources", [])
        dimensions: List[CompareDimensionPlan] = state.get("compare_dimensions", [])
        profiles: List[DocumentProfile] = state.get("document_profiles", [])
        docs_by_source: Dict[str, List[RetrievedDoc]] = state.get(
            "compare_docs_by_source", {}
        )

        if len(sources) < 2 or not dimensions or not profiles:
            answer = "未能生成对比表：请在问题中点名至少 2 篇可用文档。"
            return {
                "answer": answer,
                "claim_audit_exemption": make_claim_audit_exemption(
                    answer,
                    CLAIM_AUDIT_EXEMPTION_GUIDANCE,
                ),
            }

        table_answer = _format_compare_blocks(sources, dimensions, profiles)
        conclusion_warning = ""
        cells = _profile_cells(profiles)
        generated_cells = [cell for cell in cells if _is_generated_cell(cell)]
        incomplete_cells = [
            cell
            for cell in cells
            if not _is_generated_cell(cell) and not _is_no_evidence_cell(cell)
        ]
        missing_cell_count = _missing_cell_count(sources, dimensions, profiles)
        if incomplete_cells or missing_cell_count:
            conclusion = ""
            conclusion_warning = (
                "部分对比单元处理未完成，已跳过简短结论，避免基于不完整维度推断。"
            )
        elif not generated_cells:
            # 全 no-evidence 矩阵没有可供结论模型复用的 EID，直接保留确定性矩阵。
            conclusion = ""
        elif state.get("is_local", False):
            # 本地模式跳过额外结论生成，避免再次触发 Ollama 内存加载。
            conclusion = ""
            conclusion_warning = "本地 Ollama 模式已跳过简短结论生成，以降低内存占用。"
        else:
            try:
                conclusion = CompareGeneratorAgent._generate_conclusion(
                    state, table_answer
                )
            except Exception as exc:
                conclusion = ""
                conclusion_warning = (
                    f"结论生成失败，已降级为纯表格：{type(exc).__name__}: {exc}"
                )
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

        output: dict[str, Any] = {
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
                fallback_when_empty=not any(
                    "evidence" in cell
                    for profile in profiles
                    for cell in profile.get("cells", [])
                ),
            ),
            "steps_trace": steps_trace,
        }
        if not generated_cells and (incomplete_cells or missing_cell_count):
            output["error"] = "compare_evidence_units_incomplete"
            output["claim_audit_exemption"] = make_claim_audit_exemption(
                answer,
                CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
            )
        return output

    # 生成 conclusion。
    @staticmethod
    def _generate_conclusion(state: dict, table_answer: str) -> str:
        # 结论只能复用已带引用的对比块，生成后还会单独过引用校验。
        is_local = state.get("is_local", False)
        query = state.get("query", "")

        llm = Generator._get_client_for_node("compare_conclusion", is_local=is_local)
        messages = [
            SystemMessage(content=COMPARE_CONCLUSION_SYSTEM_PROMPT),
            HumanMessage(
                content=COMPARE_CONCLUSION_USER_PROMPT_TEMPLATE.format(
                    query=query, table_answer=table_answer
                )
            ),
        ]
        return _normalize_evidence_citation_placement(
            llm.invoke(messages).content.strip()
        )

    # 校验 compare answer。
    @staticmethod
    def validate_compare_answer(state: dict) -> dict:
        # 结论单独校验，避免表格引用掩盖无引用结论。
        sources: List[str] = state.get("compare_sources", [])
        dimensions: List[CompareDimensionPlan] = state.get("compare_dimensions", [])
        profiles: List[DocumentProfile] = state.get("document_profiles", [])
        table_answer = state.get("compare_table_answer", state.get("answer", ""))
        conclusion = state.get("compare_conclusion", "")
        answer = state.get("answer", table_answer)
        generated_cells = [
            cell for cell in _profile_cells(profiles) if _is_generated_cell(cell)
        ]
        evidence_ledger = state.get("evidence_ledger")
        registry_updates = {}
        if evidence_ledger is None:
            if conclusion or generated_cells:
                docs_by_source: Dict[str, List[RetrievedDoc]] = state.get(
                    "compare_docs_by_source", {}
                )
                sources = sources or list(docs_by_source)
                annotated_by_source, evidence_ledger = ensure_compare_evidence_ids(
                    docs_by_source, sources
                )
                registry_updates = {
                    "compare_sources": sources,
                    "compare_docs_by_source": annotated_by_source,
                    "evidence_ledger": evidence_ledger,
                }
            else:
                evidence_ledger = []
                registry_updates = {"evidence_ledger": []}

        critiques = []
        ledger_sources = {
            str(entry.get("evidence_id") or ""): str(entry.get("source") or "")
            for entry in evidence_ledger
            if isinstance(entry, dict)
        }
        binding_errors = []
        for profile in profiles:
            for cell in profile.get("cells", []):
                if not _is_generated_cell(cell):
                    continue
                cell_source = str(cell.get("source") or profile.get("source") or "")
                wrong_ids = [
                    evidence_id
                    for evidence_id in extract_evidence_ids(cell.get("content", ""))
                    if ledger_sources.get(evidence_id)
                    and ledger_sources[evidence_id] != cell_source
                ]
                if wrong_ids:
                    binding_errors.append(
                        f"{cell_source}/{cell.get('dimension_id', '')}:"
                        + ",".join(wrong_ids)
                    )
        if binding_errors:
            critiques.append(
                "【单元格证据越界】对比单元只能引用当前文档的证据："
                + "；".join(binding_errors)
            )
        if conclusion:
            # 先查结论，防止表格里的合法引用掩盖结论缺引用。
            table_evidence_ids = set(extract_evidence_ids(table_answer))
            conclusion_ledger = [
                entry
                for entry in evidence_ledger
                if entry.get("evidence_id") in table_evidence_ids
            ]
            conclusion_check = validate_evidence_citations(
                conclusion, conclusion_ledger
            )
            if not conclusion_check["is_valid"]:
                critiques.append(conclusion_check["critique"])
                answer = table_answer

        # 只校验模型生成的 cell；no-evidence 和操作错误行是程序固定文案。
        if generated_cells:
            generated_answer = "\n".join(
                str(cell.get("content") or "") for cell in generated_cells
            )
            full_check = validate_evidence_citations(
                generated_answer, evidence_ledger
            )
            if not full_check["is_valid"]:
                critiques.append(full_check["critique"])
                answer = table_answer

        critique = "\n\n".join(critiques)
        if critique:
            answer = append_citation_warning(answer, critique, "单元格")

        return {
            **registry_updates,
            "answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "critique": critique,
            "claim_audit_projection": _compare_claim_audit_projection(
                answer,
                sources,
                dimensions,
                profiles,
                conclusion,
            ),
        }
