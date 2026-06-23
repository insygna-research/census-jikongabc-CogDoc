# 契约层零 web 框架依赖：不急加载 api.app，避免 import api.schemas 被迫拉起 fastapi。
from api.schemas import (
    API_SCHEMA_VERSION,
    ChatMode,
    ChatRequest,
    ChatResponse,
    ChatTask,
    Citation,
    ErrorCode,
    ErrorResponse,
    Evidence,
    build_error_response,
    chat_result_to_response,
)

__all__ = [
    "API_SCHEMA_VERSION",
    "ChatMode",
    "ChatRequest",
    "ChatResponse",
    "ChatTask",
    "Citation",
    "ErrorCode",
    "ErrorResponse",
    "Evidence",
    "build_error_response",
    "chat_result_to_response",
]
