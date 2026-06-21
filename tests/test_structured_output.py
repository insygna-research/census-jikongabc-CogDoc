from langchain_core.messages import AIMessage
from pydantic import BaseModel

from agents.structured_output import _METHOD_CACHE, _extract_json_object, invoke_structured


class DemoSchema(BaseModel):
    value: str


class MethodAwareLLM:
    def __init__(self, supported_methods, raw_content = None, base_url = "https://unknown.example/v1", model = "demo"):
        self.supported_methods = set(supported_methods)
        self.raw_content = raw_content or '{"value":"raw"}'
        self.openai_api_base = base_url
        self.model_name = model
        self.methods_seen = []
        self.method = None

    def with_structured_output(self, schema, **kwargs):
        method = kwargs["method"]
        self.methods_seen.append(method)
        if method not in self.supported_methods:
            raise RuntimeError(f"{method} unsupported")

        child = MethodAwareLLM(self.supported_methods, self.raw_content, self.openai_api_base, self.model_name)
        child.methods_seen = self.methods_seen
        child.method = method
        return child

    def invoke(self, messages):
        if self.method:
            return DemoSchema(value = self.method)
        return AIMessage(content = self.raw_content)


def test_invoke_structured_uses_json_mode_first(monkeypatch):
    # 默认 auto 首选 json_mode，兼容 DeepSeek 的 json_object 输出。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM({"json_mode"})

    result = invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}])

    assert result.value == "json_mode"
    assert llm.methods_seen == ["json_mode"]


def test_invoke_structured_falls_back_to_json_schema(monkeypatch):
    # json_mode 不可用时继续尝试更严格的 json_schema。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM({"json_schema"})

    result = invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}])

    assert result.value == "json_schema"
    assert llm.methods_seen == ["json_mode", "json_schema"]


def test_invoke_structured_falls_back_to_raw_json(monkeypatch):
    # 所有结构化方法不可用时退回普通文本 JSON 解析。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM(set(), raw_content = '```json\n{"value":"plain"}\n```')

    result = invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}])

    assert result.value == "plain"
    assert llm.methods_seen == ["json_mode", "json_schema", "function_calling"]


def test_invoke_structured_respects_configured_method(monkeypatch):
    # 显式配置结构化输出方法时不再自动探测其它方法。
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "function_calling")
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM({"function_calling", "json_mode"})

    result = invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}])

    assert result.value == "function_calling"
    assert llm.methods_seen == ["function_calling"]


def test_invoke_structured_caches_successful_auto_method(monkeypatch):
    # auto 探测成功后，同一个 client 后续直接复用成功方法。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM({"raw_json"}, raw_content = '{"value":"plain"}')

    assert invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}]).value == "plain"
    assert llm.methods_seen == ["json_mode", "json_schema", "function_calling"]

    llm.methods_seen.clear()
    assert invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}]).value == "plain"
    assert llm.methods_seen == []


def test_invoke_structured_cache_survives_client_recreation(monkeypatch):
    # 方法缓存按后端和模型生效，client 重建后仍能复用。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    first = MethodAwareLLM({"json_schema"})
    second = MethodAwareLLM({"json_schema"})

    assert invoke_structured(first, DemoSchema, [{"role": "user", "content": "json"}]).value == "json_schema"
    assert first.methods_seen == ["json_mode", "json_schema"]

    assert invoke_structured(second, DemoSchema, [{"role": "user", "content": "json"}]).value == "json_schema"
    assert second.methods_seen == ["json_schema"]


def test_invoke_structured_prefers_raw_json_for_local_ollama(monkeypatch):
    # 本地 Ollama 默认优先 raw_json，避免 response_format 试错请求。
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising = False)
    _METHOD_CACHE.clear()
    llm = MethodAwareLLM(
        {"json_mode"},
        raw_content = '{"value":"local"}',
        base_url = "http://localhost:11434/v1",
        model = "qwen2.5:7b",
    )

    assert invoke_structured(llm, DemoSchema, [{"role": "user", "content": "json"}]).value == "local"
    assert llm.methods_seen == []


def test_extract_json_object_uses_balanced_object_not_last_brace():
    # raw_json 兜底按括号配平提取第一个完整 JSON 对象。
    content = '结果如下：{"value":"ok", "nested":{"x":"{literal}"}}。希望有帮助 {face}'

    assert _extract_json_object(content) == '{"value":"ok", "nested":{"x":"{literal}"}}'
