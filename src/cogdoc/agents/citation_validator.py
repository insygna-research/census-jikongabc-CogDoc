from typing import List, Dict, Any
from cogdoc.tools.rust_core_loader import ensure_rust_core


# Python 门面只依赖 native checker 的结构化结果。
rust_core = ensure_rust_core("validate_citations_native")


# 定义 CitationValidatorAgent 数据结构。
class CitationValidatorAgent:
    # 校验 citations。
    @staticmethod
    def validate_citations(
        answer: str, valid_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Rust 负责确定性规则校验，Python 负责生成 critique。
        native_result = rust_core.validate_citations_native(answer or "", valid_docs)
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
