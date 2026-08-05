from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


CLAIM_AUDIT_PROJECTION_STATE_KEY = "claim_audit_projection"
CLAIM_AUDIT_PROJECTION_VERSION = "v1"

_MAX_SEGMENTS = 512
_MAX_SEGMENT_ID_CHARS = 256
_MAX_SEGMENT_CONTENT_CHARS = 50_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ClaimAuditProjectionStatus(str, Enum):
    """How one structured output segment participates in claim audit."""

    GENERATED = "generated"
    DETERMINISTIC = "deterministic"
    OPERATIONAL = "operational"


class ClaimAuditProjectionError(ValueError):
    """A fail-closed projection contract violation with a stable reason code."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _canonical_answer(answer: Any) -> str:
    return str(answer or "").strip()


def _answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _required_text(value: Any, *, name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ClaimAuditProjectionError(
            "claim_audit_projection_segment_invalid", f"{name} must be a string"
        )
    normalized = value.strip()
    if not normalized:
        raise ClaimAuditProjectionError(
            "claim_audit_projection_segment_invalid", f"{name} is required"
        )
    if len(normalized) > max_chars:
        raise ClaimAuditProjectionError(
            "claim_audit_projection_segment_invalid",
            f"{name} exceeds {max_chars} characters",
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ClaimAuditProjectionSegment:
    """One rendered structured unit and its trusted generation disposition."""

    segment_id: str
    content: str
    status: ClaimAuditProjectionStatus
    source_status: str = ""
    obligation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_id",
            _required_text(
                self.segment_id,
                name="segment_id",
                max_chars=_MAX_SEGMENT_ID_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "content",
            _required_text(
                self.content,
                name="content",
                max_chars=_MAX_SEGMENT_CONTENT_CHARS,
            ),
        )
        if not isinstance(self.status, ClaimAuditProjectionStatus):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "status must be a ClaimAuditProjectionStatus",
            )
        if not isinstance(self.source_status, str):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "source_status must be a string",
            )
        object.__setattr__(self, "source_status", self.source_status.strip())
        if not isinstance(self.obligation_ids, tuple):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "obligation_ids must be a tuple",
            )
        normalized_obligation_ids = tuple(
            _required_text(
                obligation_id,
                name="obligation_id",
                max_chars=_MAX_SEGMENT_ID_CHARS,
            )
            for obligation_id in self.obligation_ids
        )
        if len(normalized_obligation_ids) != len(set(normalized_obligation_ids)):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_duplicate_obligation_id"
            )
        object.__setattr__(self, "obligation_ids", normalized_obligation_ids)

    @classmethod
    def generated(
        cls,
        segment_id: str,
        content: str,
        *,
        source_status: str = "generated",
        obligation_ids: Sequence[str] = (),
    ) -> ClaimAuditProjectionSegment:
        return cls(
            segment_id=segment_id,
            content=content,
            status=ClaimAuditProjectionStatus.GENERATED,
            source_status=source_status,
            obligation_ids=tuple(obligation_ids),
        )

    @classmethod
    def deterministic(
        cls,
        segment_id: str,
        content: str,
        *,
        source_status: str,
        obligation_ids: Sequence[str] = (),
    ) -> ClaimAuditProjectionSegment:
        return cls(
            segment_id=segment_id,
            content=content,
            status=ClaimAuditProjectionStatus.DETERMINISTIC,
            source_status=source_status,
            obligation_ids=tuple(obligation_ids),
        )

    @classmethod
    def operational(
        cls,
        segment_id: str,
        content: str,
        *,
        source_status: str,
        obligation_ids: Sequence[str] = (),
    ) -> ClaimAuditProjectionSegment:
        return cls(
            segment_id=segment_id,
            content=content,
            status=ClaimAuditProjectionStatus.OPERATIONAL,
            source_status=source_status,
            obligation_ids=tuple(obligation_ids),
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "content": self.content,
            "status": self.status.value,
            "source_status": self.source_status,
            "obligation_ids": list(self.obligation_ids),
        }


@dataclass(frozen=True, slots=True)
class ClaimAuditProjection:
    """Answer-bound ordered projection of structured generation output."""

    answer_sha256: str
    segments: tuple[ClaimAuditProjectionSegment, ...]
    version: str = CLAIM_AUDIT_PROJECTION_VERSION

    def __post_init__(self) -> None:
        if self.version != CLAIM_AUDIT_PROJECTION_VERSION:
            raise ClaimAuditProjectionError(
                "claim_audit_projection_version_invalid", str(self.version)
            )
        if not isinstance(self.answer_sha256, str) or not _SHA256_RE.fullmatch(
            self.answer_sha256
        ):
            raise ClaimAuditProjectionError("claim_audit_projection_digest_invalid")
        if not isinstance(self.segments, tuple) or not all(
            isinstance(segment, ClaimAuditProjectionSegment)
            for segment in self.segments
        ):
            raise ClaimAuditProjectionError("claim_audit_projection_segments_invalid")
        if not self.segments:
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segments_invalid",
                "at least one segment is required",
            )
        if len(self.segments) > _MAX_SEGMENTS:
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segments_invalid",
                f"at most {_MAX_SEGMENTS} segments are allowed",
            )
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_duplicate_segment_id"
            )
        obligation_ids = [
            obligation_id
            for segment in self.segments
            for obligation_id in segment.obligation_ids
        ]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_duplicate_obligation_id"
            )

    @property
    def generated_segments(self) -> tuple[ClaimAuditProjectionSegment, ...]:
        return tuple(
            segment
            for segment in self.segments
            if segment.status is ClaimAuditProjectionStatus.GENERATED
        )

    @property
    def audit_text(self) -> str:
        """Return only model-generated content, preserving structured order."""

        return "\n\n".join(segment.content for segment in self.generated_segments)

    @property
    def has_generated_content(self) -> bool:
        return bool(self.generated_segments)

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(
            obligation_id
            for segment in self.segments
            for obligation_id in segment.obligation_ids
        )

    @property
    def metrics(self) -> dict[str, int]:
        counts = Counter(segment.status.value for segment in self.segments)
        return {
            "segment_count": len(self.segments),
            "generated_count": counts[ClaimAuditProjectionStatus.GENERATED.value],
            "deterministic_count": counts[
                ClaimAuditProjectionStatus.DETERMINISTIC.value
            ],
            "operational_count": counts[ClaimAuditProjectionStatus.OPERATIONAL.value],
            "obligation_count": len(self.obligation_ids),
        }

    def validate_for_answer(self, answer: Any) -> ClaimAuditProjection:
        canonical_answer = _canonical_answer(answer)
        if not canonical_answer:
            raise ClaimAuditProjectionError("claim_audit_projection_answer_empty")
        if _answer_sha256(canonical_answer) != self.answer_sha256:
            raise ClaimAuditProjectionError("claim_audit_projection_answer_mismatch")

        cursor = 0
        for segment in self.segments:
            position = canonical_answer.find(segment.content, cursor)
            if position < 0:
                reason_code = (
                    "claim_audit_projection_segment_order_invalid"
                    if segment.content in canonical_answer
                    else "claim_audit_projection_segment_missing"
                )
                raise ClaimAuditProjectionError(reason_code, segment.segment_id)
            cursor = position + len(segment.content)
        return self

    def to_state(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "answer_sha256": self.answer_sha256,
            "segments": [segment.to_state() for segment in self.segments],
        }


def build_claim_audit_projection(
    answer: Any,
    segments: Sequence[ClaimAuditProjectionSegment],
) -> ClaimAuditProjection:
    """Build and immediately validate a projection against its final answer."""

    if isinstance(segments, (str, bytes, bytearray)) or not isinstance(
        segments, Sequence
    ):
        raise ClaimAuditProjectionError("claim_audit_projection_segments_invalid")
    canonical_answer = _canonical_answer(answer)
    if not canonical_answer:
        raise ClaimAuditProjectionError("claim_audit_projection_answer_empty")
    projection = ClaimAuditProjection(
        answer_sha256=_answer_sha256(canonical_answer),
        segments=tuple(segments),
    )
    return projection.validate_for_answer(canonical_answer)


def load_claim_audit_projection(
    value: Any,
    *,
    answer: Any,
) -> ClaimAuditProjection:
    """Strictly parse persisted state and verify its answer binding and order."""

    if isinstance(value, ClaimAuditProjection):
        return value.validate_for_answer(answer)
    if not isinstance(value, Mapping):
        raise ClaimAuditProjectionError("claim_audit_projection_invalid")

    allowed_projection_keys = {"version", "answer_sha256", "segments"}
    if set(value) != allowed_projection_keys:
        raise ClaimAuditProjectionError(
            "claim_audit_projection_invalid", "projection keys do not match v1"
        )
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments, (str, bytes, bytearray)
    ):
        raise ClaimAuditProjectionError("claim_audit_projection_segments_invalid")

    segments: list[ClaimAuditProjectionSegment] = []
    allowed_segment_keys = {
        "segment_id",
        "content",
        "status",
        "source_status",
        "obligation_ids",
    }
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping) or set(raw_segment) != (
            allowed_segment_keys
        ):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "segment keys do not match v1",
            )
        try:
            status = ClaimAuditProjectionStatus(raw_segment.get("status"))
        except (TypeError, ValueError) as exc:
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "unknown projection status",
            ) from exc
        raw_obligation_ids = raw_segment["obligation_ids"]
        if not isinstance(raw_obligation_ids, list):
            raise ClaimAuditProjectionError(
                "claim_audit_projection_segment_invalid",
                "obligation_ids must be a list in state",
            )
        segments.append(
            ClaimAuditProjectionSegment(
                segment_id=raw_segment["segment_id"],
                content=raw_segment["content"],
                status=status,
                source_status=raw_segment["source_status"],
                obligation_ids=tuple(raw_obligation_ids),
            )
        )

    projection = ClaimAuditProjection(
        version=value["version"],
        answer_sha256=value["answer_sha256"],
        segments=tuple(segments),
    )
    return projection.validate_for_answer(answer)


__all__ = [
    "CLAIM_AUDIT_PROJECTION_STATE_KEY",
    "CLAIM_AUDIT_PROJECTION_VERSION",
    "ClaimAuditProjection",
    "ClaimAuditProjectionError",
    "ClaimAuditProjectionSegment",
    "ClaimAuditProjectionStatus",
    "build_claim_audit_projection",
    "load_claim_audit_projection",
]
