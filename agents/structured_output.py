import json
import os
from typing import Iterable, Type
from pydantic import BaseModel


STRUCTURED_OUTPUT_METHODS = ("json_mode", "json_schema", "function_calling", "raw_json")
_METHOD_CACHE: dict[str, str] = {}


def _llm_cache_key(llm) -> str:
    base_url = str(getattr(llm, "openai_api_base", "") or getattr(llm, "base_url", ""))
    model = str(getattr(llm, "model_name", "") or getattr(llm, "model", ""))
    return f"{llm.__class__.__module__}.{llm.__class__.__name__}:{base_url}:{model}"


def _auto_methods_for_llm(llm) -> list[str]:
    # 本地部署的兼容 OpenAI 接口的服务端，大多会拒绝各类 response_format 格式限定参数；直接返回原始 JSON 可避免无效请求损耗。
    base_url = str(getattr(llm, "openai_api_base", "") or getattr(llm, "base_url", "")).lower()
    model = str(getattr(llm, "model_name", "") or getattr(llm, "model", "")).lower()

    if "localhost" in base_url or "127.0.0.1" in base_url or "ollama" in base_url:
        return ["raw_json", "json_mode", "json_schema", "function_calling"]
    if "deepseek" in base_url or "deepseek" in model:
        return ["json_mode", "raw_json", "json_schema", "function_calling"]
    if "api.openai.com" in base_url:
        return ["json_schema", "json_mode", "function_calling", "raw_json"]

    return ["json_mode", "json_schema", "function_calling", "raw_json"]


def _configured_methods(llm = None) -> list[str]:
    # auto 优先走兼容面较广的 json_object，再尝试更严格/更旧的 LangChain 方法。
    configured = os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "auto").strip()
    if not configured or configured == "auto":
        return _auto_methods_for_llm(llm) if llm is not None else ["json_mode", "json_schema", "function_calling", "raw_json"]

    methods = [item.strip() for item in configured.split(",") if item.strip()]
    invalid = [method for method in methods if method not in STRUCTURED_OUTPUT_METHODS]
    if invalid:
        raise ValueError(
            "Unsupported LLM_STRUCTURED_OUTPUT_METHOD value(s): "
            f"{', '.join(invalid)}. Supported values: auto, "
            f"{', '.join(STRUCTURED_OUTPUT_METHODS)}"
        )
    return methods


def _message_content(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _extract_json_object(text: str) -> str:
    # raw_json 兜底时容忍模型包了一点解释或 Markdown。
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    for start, char in enumerate(stripped):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(stripped)):
            current = stripped[idx]

            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return candidate
                    break

    raise ValueError(f"No JSON object found in model output: {text[:200]}")


def _parse_schema(schema: Type[BaseModel], payload: str | dict) -> BaseModel:
    if isinstance(payload, dict):
        return schema.model_validate(payload)
    return schema.model_validate_json(payload)


def _invoke_raw_json(llm, schema: Type[BaseModel], messages: Iterable) -> BaseModel:
    response = llm.invoke(list(messages))
    content = _message_content(response)
    return _parse_schema(schema, _extract_json_object(content))


def invoke_structured(llm, schema: Type[BaseModel], messages: Iterable) -> BaseModel:
    # 统一兼容 OpenAI、DeepSeek、Ollama 等 OpenAI-compatible 后端的结构化输出差异。
    errors = []
    messages = list(messages)
    methods = _configured_methods(llm)
    cache_key = _llm_cache_key(llm)

    if os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "auto").strip() in ("", "auto"):
        cached_method = _METHOD_CACHE.get(cache_key)
        if cached_method in methods:
            methods = [cached_method] + [method for method in methods if method != cached_method]

    for method in methods:
        try:
            if method == "raw_json":
                output = _invoke_raw_json(llm, schema, messages)
                _METHOD_CACHE[cache_key] = method
                return output

            structured_llm = llm.with_structured_output(schema, method = method)
            output = structured_llm.invoke(messages)
            if isinstance(output, schema):
                _METHOD_CACHE[cache_key] = method
                return output
            if isinstance(output, dict):
                parsed = _parse_schema(schema, output)
                _METHOD_CACHE[cache_key] = method
                return parsed
            if isinstance(output, str):
                parsed = _parse_schema(schema, _extract_json_object(output))
                _METHOD_CACHE[cache_key] = method
                return parsed
            parsed = _parse_schema(schema, json.loads(_message_content(output)))
            _METHOD_CACHE[cache_key] = method
            return parsed
        except Exception as exc:
            errors.append(f"{method}: {exc}")

    raise RuntimeError("Structured output failed for all methods. " + " | ".join(errors))
