import importlib
from types import ModuleType

def ensure_rust_core(*required: str) -> ModuleType:
    try:
        rust_core = importlib.import_module("rust_core")
    except ImportError as exc:
        raise RuntimeError("Rust 扩展 rust_core 未安装。请先运行: cd rust_core && maturin develop") from exc

    missing = [name for name in required if not hasattr(rust_core, name)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Rust 扩展 rust_core 未正确加载，缺少: {missing_str}。请先运行: cd rust_core && maturin develop")

    return rust_core
