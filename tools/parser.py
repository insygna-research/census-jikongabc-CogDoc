import pymupdf as fitz  # 直接导入 pymupdf，避免被 PyPI 同名 stub 包 fitz 顶替
import re
import os
from typing import List
from graph.state import ParsedPage


# 解析逻辑变化时 bump：进入增量复用门控，避免未变文档复用旧解析结果。
PARSER_VERSION = "pymupdf_smart_parse_v1"


# 处理 smart parse 相关逻辑。
def smart_parse(pdf_path: str) -> List[ParsedPage]:
    source_name = os.path.basename(pdf_path)  # 文件名
    doc = fitz.open(pdf_path)
    parsed_pages: List[ParsedPage] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        text = page.get_text().strip()
        blocks = page.get_text("blocks")

        # 疑似扫描页，留给后续 OCR 处理
        if len(text) < 20 and len(blocks) <= 1:
            parsed_pages.append(
                {
                    "page": page_num,
                    "source": source_name,
                    "text": "",
                    "is_ocr_fallback": True,
                }
            )
            continue
        width = page.rect.width

        # 计算文本块中心点，用于判断单双栏
        center_xs = [(b[0] + b[2]) / 2 for b in blocks if len(b[4].strip()) > 5]

        left_count = sum(1 for cx in center_xs if cx < width * 0.4)
        right_count = sum(1 for cx in center_xs if cx > width * 0.6)

        if left_count > 3 and right_count > 3:
            # 双栏论文：左栏 -> 右栏
            left_blocks = sorted(
                [b for b in blocks if (b[0] + b[2]) / 2 <= width * 0.5],
                key=lambda x: x[1],
            )
            right_blocks = sorted(
                [b for b in blocks if (b[0] + b[2]) / 2 > width * 0.5],
                key=lambda x: x[1],
            )
            page_text = (
                "\n".join([b[4] for b in left_blocks])
                + "\n"
                + "\n".join([b[4] for b in right_blocks])
            )
        else:
            # 单栏：从上到下、从左到右
            blocks.sort(key=lambda x: (x[1], x[0]))
            page_text = "\n".join([b[4] for b in blocks])

        parsed_pages.append(
            {
                "page": page_num,
                "source": source_name,
                "text": re.sub(r"\n{3,}", "\n\n", page_text).strip(),
                "is_ocr_fallback": False,
            }
        )
    return parsed_pages
