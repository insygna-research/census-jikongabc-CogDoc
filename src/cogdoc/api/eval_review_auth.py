from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request


def _request_api_key(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def require_eval_reviewer(request: Request) -> str:
    """Authorize evidence-eval curation and return a non-secret actor identity."""

    configured = set(getattr(request.app.state, "eval_review_api_keys", set()))
    if not configured:
        raise HTTPException(status_code=403, detail="证据评测审核接口未启用")
    supplied = _request_api_key(request)
    if not supplied or not any(
        hmac.compare_digest(supplied, expected) for expected in configured
    ):
        raise HTTPException(status_code=403, detail="需要证据评测审核权限")
    fingerprint = hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:16]
    return f"eval-review:{fingerprint}"
