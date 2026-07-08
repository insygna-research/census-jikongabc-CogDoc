from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import httpx

from cogdoc.api.schemas import DerivedKnowledge
from cogdoc.api.time_utils import now_iso
from cogdoc.config.settings import get_settings
from cogdoc.observability.logger import log_event


# 发送外部回调。
class WebhookDispatcher:
    def __init__(
        self,
        *,
        url: str | None = None,
        secret: str | None = None,
        timeout_seconds: float | None = None,
    ):
        settings = get_settings()
        self._url = (url if url is not None else settings.cogdoc_webhook_url).strip()
        self._secret = (
            secret if secret is not None else settings.cogdoc_webhook_secret
        ).strip()
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.cogdoc_webhook_timeout_seconds
        )

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    # 发送事件，失败只记日志。
    def emit(self, event: str, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        body = {
            "schema_version": "v1",
            "event_id": uuid4().hex,
            "event": event,
            "occurred_at": now_iso(),
            "payload": payload,
        }
        headers = {}
        if self._secret:
            headers["X-CogDoc-Webhook-Secret"] = self._secret
        try:
            response = httpx.post(
                self._url,
                json=body,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            log_event(
                "webhook",
                "webhook_emit_failed",
                {},
                level=logging.WARNING,
                event_type=event,
                error_class=type(exc).__name__,
            )
            return False
        log_event("webhook", "webhook_emit_succeeded", {}, event_type=event)
        return True


# 异步提交待审核知识通知。
def notify_pending_created(app, row: dict[str, Any], source: str) -> None:
    if row.get("status") != "pending":
        return
    dispatcher = getattr(app.state, "webhook_dispatcher", None)
    if dispatcher is None or not getattr(dispatcher, "enabled", False):
        return
    public_row = {
        key: value for key, value in row.items() if key in DerivedKnowledge.model_fields
    }
    payload = {
        "source": source,
        "knowledge": DerivedKnowledge.model_validate(public_row).model_dump(
            mode="json"
        ),
    }
    executor = getattr(app.state, "offload_executor", None)
    if executor is None:
        return
    try:
        executor.submit(dispatcher.emit, "knowledge.pending_created", payload)
    except RuntimeError as exc:
        level = logging.DEBUG if "shutdown" in str(exc).lower() else logging.WARNING
        log_event(
            "webhook",
            "webhook_submit_failed",
            {},
            level=level,
            event_type="knowledge.pending_created",
            error_class=type(exc).__name__,
        )
    except Exception as exc:
        log_event(
            "webhook",
            "webhook_submit_failed",
            {},
            level=logging.WARNING,
            event_type="knowledge.pending_created",
            error_class=type(exc).__name__,
        )
