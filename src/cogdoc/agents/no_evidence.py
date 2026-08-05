from __future__ import annotations

import re


NO_EVIDENCE_MARKER = "文档中未明确说明"
_NO_EVIDENCE_STATEMENT_RE = re.compile(rf"{re.escape(NO_EVIDENCE_MARKER)}[。.!！?？]?")


def is_no_evidence_statement(content: object) -> bool:
    """Match only the deterministic no-evidence sentence emitted by agents."""

    return bool(_NO_EVIDENCE_STATEMENT_RE.fullmatch(str(content or "").strip()))
