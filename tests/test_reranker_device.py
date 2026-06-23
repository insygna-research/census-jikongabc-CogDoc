import pytest
from tools import device, reranker
from tools.reranker import BGEReranker


@pytest.fixture(autouse=True)
def restore_reranker_state():
    saved_device = BGEReranker.device
    saved_model = BGEReranker._model
    saved_tokenizer = BGEReranker._tokenizer
    yield
    BGEReranker.device = saved_device
    BGEReranker._model = saved_model
    BGEReranker._tokenizer = saved_tokenizer


class _FakeModel:
    def to(self, dev):
        self.device = dev
        return self

    def eval(self):
        return self


def _stub_model_loading(monkeypatch):
    monkeypatch.setattr(
        reranker.AutoTokenizer, "from_pretrained", lambda name: object()
    )
    fake = _FakeModel()
    monkeypatch.setattr(
        reranker.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda name: fake,
    )
    return fake


def test_default_device_is_one_of_known_backends():
    # 验证默认 reranker 设备属于支持的后端。
    assert BGEReranker.default_device() in {"cuda", "mps", "cpu"}


def test_default_device_uses_gpu_when_enough_free_memory(monkeypatch):
    # 空闲显存充足时走 cuda。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 6 * 1024 * 1024 * 1024)
    BGEReranker.device = "cpu"
    BGEReranker._model = None

    assert BGEReranker.default_device() == "cuda"


def test_default_device_falls_back_to_cpu_when_gpu_low(monkeypatch):
    # 显存被其它进程占满、空闲不足阈值时回落 CPU，避免 CUDA OOM。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 90 * 1024 * 1024)
    monkeypatch.setattr(device, "mps_available", lambda: False)
    BGEReranker.device = "cpu"
    BGEReranker._model = None

    assert BGEReranker.default_device() == "cpu"


def test_default_device_sticky_once_loaded_on_cuda(monkeypatch):
    # 已在 cuda 且模型在显存里：自身占用已计入空闲值，即便此刻空闲不足也不抖回 CPU。
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 0)
    BGEReranker.device = "cuda"
    BGEReranker._model = object()

    assert BGEReranker.default_device() == "cuda"


def test_get_resources_resolves_device_when_unset(monkeypatch):
    # 直连调用（未 set_device，device=None）按显存自动选设备，不退化成 CPU。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 6 * 1024 * 1024 * 1024)
    fake = _stub_model_loading(monkeypatch)
    BGEReranker.device = None
    BGEReranker._model = None
    BGEReranker._tokenizer = None

    BGEReranker._get_resources()

    assert BGEReranker.device == "cuda"
    assert fake.device == "cuda"


def test_get_resources_respects_explicit_cpu(monkeypatch):
    # 本地模式 set_device("cpu") 后即便显存充足也不被覆盖。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 6 * 1024 * 1024 * 1024)
    fake = _stub_model_loading(monkeypatch)
    BGEReranker._model = None
    BGEReranker._tokenizer = None
    BGEReranker.set_device("cpu")

    BGEReranker._get_resources()

    assert fake.device == "cpu"


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
