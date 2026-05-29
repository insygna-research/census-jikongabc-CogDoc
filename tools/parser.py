import fitz
import re
import os
from typint import List
from graph.state import ParsedPage

def smart_parse(pdf_path: str) -> List[ParsedPage]:
    source_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    parsed_pages: List[ParsedPage] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        text = page.get_text().strip()
        blocks = page.get_text("blocks")

        if len(text) < 20 and len(blocks) <= 1:
            parsed_pages.append({
                "page": page_num,
                "source": source_name,
                "text": "",
                "is_ocr_fallback": True
            })
            continue
        width = page.rect.width
        center_xs = [(b[0] + b[2]) / 2 for b in blocks if len(b[4].strip()) > 5]
        left_count = sum(1 for cx in center_xs if cx < width * 0.4)
        right_count = sum(1 for cx in center_xs if cx > width * 0.6)
        if left_count > 3 and right_count > 3:
            left_blocks = sorted([b for b in blocks if (b[0] + b[2])/2 <= width * 0.5], key = lambda x: x[1])
            right_blocks = sorted([b for b in blocks if (b[0] + b[2])/2 > width * 0.5], key = lambda x: x[1])
            page_text = "\n".join([b[4] for b in left_blocks]) + "\n" + "\n".join([b[4] for b in right_blocks])
        else:
            blocks.sort(key  =lambda x: (x[1], x[0]))
            page_text = "\n".join([b[4] for b in blocks])
        parsed_pages.append({
            "page": page_num,
            "source": source_name,
            "text": re.sub(r'\n{3,}', '\n\n', page_text).strip(),
            "is_ocr_fallback": False
        })
    return parsed_pages
