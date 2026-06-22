import pytest
from tools.reranker import BGEReranker


@pytest.fixture(autouse=True)
def restore_reranker_state():
    saved_device = BGEReranker.device
    saved_model = BGEReranker._model
    yield
    BGEReranker.device = saved_device
    BGEReranker._model = saved_model


def test_default_device_is_one_of_known_backends():
    # 验证默认 reranker 设备属于支持的后端。
    assert BGEReranker.default_device() in {"cuda", "mps", "cpu"}


def test_switch_to_new_device_invalidates_model_singleton():
    # 验证切换到新设备时会清空已加载模型单例。
    BGEReranker.device = "cuda"
    BGEReranker._model = object()

    BGEReranker.set_device("cpu")

    assert BGEReranker.device == "cpu"
    assert BGEReranker._model is None


def test_switch_to_same_device_keeps_model_singleton():
    # 验证切换到相同设备时保留已加载模型单例。
    BGEReranker.device = "cpu"
    sentinel = object()
    BGEReranker._model = sentinel

    BGEReranker.set_device("cpu")

    assert BGEReranker._model is sentinel


def test_local_then_cloud_switch_restores_default_device():
    # 验证本地 CPU 模式后还能切回默认设备。
    default = BGEReranker.default_device()

    BGEReranker.set_device("cpu")
    assert BGEReranker.device == "cpu"

    BGEReranker.set_device(default)
    assert BGEReranker.device == default
