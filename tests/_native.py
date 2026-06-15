import pytest

def require_rust_core(*required: str):
    rust_core = pytest.importorskip("rust_core")
    # importorskip 可能命中源码目录的 namespace package，必须继续校验 native 符号。
    missing = [name for name in required if not hasattr(rust_core, name)]
    if missing:
        pytest.skip(
            "native rust_core 未构建（缺少: {}）；请先运行 maturin develop --release".format(
                ", ".join(missing)
            ),
            allow_module_level=True,
        )
    return rust_core
