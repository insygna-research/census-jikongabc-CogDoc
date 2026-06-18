import re
from typing import List, Dict, Any

class CitationValidatorAgent:
    @staticmethod
    def validate_citations(answer: str, valid_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 引用校验保持页级格式，chunk_id 只用于机器溯源。
        if not answer:
            return {"is_valid": True, "critique": ""}

        # allowed_registry 记录本轮检索实际允许引用的文件和页码。
        allowed_registry = {}
        for doc in valid_docs:
            meta = doc.get("meta", {})
            source = meta.get("source", "")
            page = meta.get("page")
            if source and page is not None:
                try:
                    allowed_registry.setdefault(source, set()).add(int(page))
                except (TypeError, ValueError):
                    continue
                
        if not allowed_registry:
            return {"is_valid": True, "critique": ""}

        allowed_pages_str_map = {
            file: ", ".join(f"P{p}" for p in sorted(pages))
            for file, pages in allowed_registry.items()
        }

        # 兜底回答不包含文档事实，允许不带引用。
        FALLBACK_MARKER = "在所提供的参考资料中未找到与该问题相关的内容"
        if FALLBACK_MARKER in answer:
            return {"is_valid": True, "critique": ""}

        citation_pattern = r'[\[［]\s*([^:：\]］]+?)\s*[:：]\s*[pPｐＰ]?\s*(\d+)\s*[\]］]'
        citations = re.findall(citation_pattern, answer, flags = re.IGNORECASE)

        if not citations:
            return {
                "is_valid": False,
                "critique": (
                    "【引用校验未通过】你的回答中未包含任何引用标签。\n"
                    "要求：每陈述一处来自文档的事实，须在该句句尾附加 [文件名.pdf:P页码] 格式的引用标签，"
                    "其中文件名和页码直接取自对应 <Document> 标签的 source 和 page 属性。\n"
                    "请重新生成回答，确保每条事实都有对应的引用标注。"
                )
            }

        invalid_sources = []
        invalid_pages = []

        for file_name, page_str in citations:
            file_name = file_name.strip()
            page_num = int(page_str)
            
            if file_name not in allowed_registry:
                invalid_sources.append(f"[{file_name}:P{page_num}] (该文件根本不在本次检索的上下文中)")
                continue
                
            if page_num not in allowed_registry[file_name]:
                valid_pages_str = allowed_pages_str_map[file_name]
                invalid_pages.append(
                    f"[{file_name}:P{page_num}] (该文件本次仅召回了 {valid_pages_str}，你引用的页码属于捏造事实)"
                )

        if invalid_sources or invalid_pages:
            critique_lines = ["【引用校验未通过】检测到以下引用存在错误，请逐一核查后重新生成："]
            for err in invalid_sources:
                critique_lines.append(f"  - 文件名错误：{err}")
            for err in invalid_pages:
                critique_lines.append(f"  - 页码错误：{err}")
            critique_lines.append(
                "\n修正要求：所有引用的文件名和页码必须与 <Document> 标签中的 source 和 page 属性完全一致，"
                "不得引用未出现在参考资料中的文件或页码。"
            )
            
            return {
                "is_valid": False,
                "critique": "\n".join(critique_lines)
            }

        return {"is_valid": True, "critique": ""}
