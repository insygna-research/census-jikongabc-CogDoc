import re
from collections.abc import Mapping, Sequence
from typing import List, Dict, Any

from cogdoc.tools.citation_ledger import (
    validate_evidence_citations as _validate_evidence_citations,
)
from cogdoc.tools.rust_core_loader import ensure_rust_core


# 门面只依赖原生校验器的结构化结果。
rust_core = ensure_rust_core("validate_citations_native")
_KNOWLEDGE_REF_RE = re.compile(
    r"[\[［]\s*knowledge\s*[:：]\s*([^\]］\s]+)\s*[\]］]", re.I
)


# 提取派生知识引用。
def _extract_knowledge_refs(answer: str) -> list[str]:
    return [
        match.group(1).strip() for match in _KNOWLEDGE_REF_RE.finditer(answer or "")
    ]


# 删除派生知识引用。
def _strip_knowledge_refs(answer: str) -> str:
    return _KNOWLEDGE_REF_RE.sub("", answer or "")


# 本轮允许引用的派生知识标识。
def _allowed_knowledge_ids(valid_docs: List[Dict[str, Any]]) -> set[str]:
    allowed = set()
    for doc in valid_docs:
        meta = doc.get("meta") or {}
        if meta.get("source_type") != "derived_knowledge":
            continue
        knowledge_id = meta.get("knowledge_id")
        if not knowledge_id and str(meta.get("chunk_id", "")).startswith("knowledge:"):
            knowledge_id = str(meta["chunk_id"]).split(":", 1)[1]
        if knowledge_id:
            allowed.add(str(knowledge_id))
    return allowed


# 过滤原始文档证据。
def _document_docs(valid_docs: List[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        doc
        for doc in valid_docs
        if (doc.get("meta") or {}).get("source_type") != "derived_knowledge"
    ]


# 定义引用校验器。
class CitationValidatorAgent:
    # 严格校验本次响应冻结的 Evidence ID；不接受物理页码引用。
    @staticmethod
    def validate_evidence_citations(
        answer: str, ledger: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        return _validate_evidence_citations(answer, ledger)

    # 校验引用。
    @staticmethod
    def validate_citations(
        answer: str, valid_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        allowed_knowledge = _allowed_knowledge_ids(valid_docs)
        knowledge_refs = _extract_knowledge_refs(answer)
        invalid_knowledge = [
            ref for ref in knowledge_refs if ref not in allowed_knowledge
        ]
        if invalid_knowledge:
            return {
                "is_valid": False,
                "critique": (
                    "【引用校验未通过】检测到派生知识引用不存在于本次上下文中："
                    + "，".join(f"[knowledge:{ref}]" for ref in invalid_knowledge)
                ),
            }

        document_answer = _strip_knowledge_refs(answer)
        document_docs = _document_docs(valid_docs)
        native_result = rust_core.validate_citations_native(
            document_answer or "", document_docs
        )
        if (
            native_result["missing_citations"]
            and knowledge_refs
            and not invalid_knowledge
        ):
            return {"is_valid": True, "critique": ""}
        if native_result["is_valid"]:
            return {"is_valid": True, "critique": ""}

        if native_result["missing_citations"]:
            return {
                "is_valid": False,
                "critique": (
                    "【引用校验未通过】你的回答中未包含任何引用标签。\n"
                    "要求：每陈述一处来自文档的事实，须在该句句尾附加 [文件名.pdf:P页码] 格式的引用标签，"
                    "其中文件名和页码直接取自对应 <Document> 标签的 source 和 page 属性。\n"
                    "请重新生成回答，确保每条事实都有对应的引用标注。"
                ),
            }

        critique_lines = [
            "【引用校验未通过】检测到以下引用存在错误，请逐一核查后重新生成："
        ]
        for err in native_result["invalid_sources"]:
            critique_lines.append(
                f"  - 文件名错误：[{err['source']}:P{err['page']}] (该文件根本不在本次检索的上下文中)"
            )
        for err in native_result["invalid_pages"]:
            valid_pages_str = ", ".join(f"P{p}" for p in err["valid_pages"])
            critique_lines.append(
                f"  - 页码错误：[{err['source']}:P{err['page']}] "
                f"(该文件本次仅召回了 {valid_pages_str}，你引用的页码属于捏造事实)"
            )
        critique_lines.append(
            "\n修正要求：所有引用的文件名和页码必须与 <Document> 标签中的 source 和 page 属性完全一致，"
            "不得引用未出现在参考资料中的文件或页码。"
        )

        return {"is_valid": False, "critique": "\n".join(critique_lines)}
