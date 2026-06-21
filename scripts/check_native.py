#!/usr/bin/env python3
import os
import sys

# 校验 rust_core 是否已构建且运行链路依赖的 native 符号齐全。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    # 从任意目录调用时都优先导入仓库源码。
    sys.path.insert(0, ROOT)

from tools.rust_core_loader import ensure_rust_core

# 与运行链路里各处 ensure_rust_core 调用的符号并集保持一致。
REQUIRED_SYMBOLS = (
    "scan_pdf_manifest_native",
    "rrf_fusion_native",
    "validate_citations_native",
    "tokenize_mixed_text_native",
    "Bm25Index",
)


def main() -> int:
    try:
        rust_core = ensure_rust_core(*REQUIRED_SYMBOLS)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    location = getattr(rust_core, "__file__", "<unknown>")
    print("✅ rust_core 原生扩展已构建并加载正常。")
    print(f"   ↳ 模块路径: {location}")
    print(f"   ↳ 已校验符号: {', '.join(REQUIRED_SYMBOLS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
