import pytest
from cogdoc.tools import device, embedder
from cogdoc.tools.embedder import Embedder


# 恢复 restore embedder state 相关逻辑。
@pytest.fixture(autouse=True)
def restore_embedder_state():
    saved_device = Embedder.device
    saved_model = Embedder._model
    yield
    Embedder.device = saved_device
    Embedder._model = saved_model


# 验证 get model uses gpu when enough free memory。
def test_get_model_uses_gpu_when_enough_free_memory(monkeypatch):
    # 模拟空闲显存充足，验证首次加载会选择 cuda。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 6 * 1024 * 1024 * 1024)
    captured = {}
    monkeypatch.setattr(
        embedder,
        "SentenceTransformer",
        lambda name, device, revision=None: captured.update(device=device),
    )
    Embedder._model = None

    Embedder.get_model()
    assert captured["device"] == "cuda"


# 验证 get model falls back to cpu when gpu low。
def test_get_model_falls_back_to_cpu_when_gpu_low(monkeypatch):
    # 模拟显存不足且无 MPS，验证首次加载安全回落 CPU。
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "cuda_free_bytes", lambda: 90 * 1024 * 1024)
    monkeypatch.setattr(device, "mps_available", lambda: False)
    captured = {}
    monkeypatch.setattr(
        embedder,
        "SentenceTransformer",
        lambda name, device, revision=None: captured.update(device=device),
    )
    Embedder._model = None

    Embedder.get_model()
    assert captured["device"] == "cpu"
