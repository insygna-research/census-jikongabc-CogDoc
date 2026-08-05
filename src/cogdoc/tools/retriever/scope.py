from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """Task-independent retrieval boundaries applied before channel top-k.

    An empty ``allowed_sources`` tuple means the whole knowledge base.  A
    non-empty tuple is an exact source-name allowlist.  Derived knowledge is an
    independent channel switch and, when source-scoped, is matched through its
    ``related_source`` binding rather than its synthetic ``knowledge:*`` source.
    """

    allowed_sources: tuple[str, ...] = ()
    include_derived_knowledge: bool = True

    def __post_init__(self) -> None:
        raw_sources: Sequence[Any] = self.allowed_sources
        if isinstance(raw_sources, (str, bytes, bytearray)):
            raise TypeError("allowed_sources must be a sequence of source names")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_source in raw_sources:
            # Source names are document identities and therefore use exact
            # spelling.  Do not case-fold or trim a valid filesystem name.
            source = "" if raw_source is None else str(raw_source)
            if source and source not in seen:
                seen.add(source)
                normalized.append(source)
        if not isinstance(self.include_derived_knowledge, bool):
            raise TypeError("include_derived_knowledge must be a boolean")
        object.__setattr__(self, "allowed_sources", tuple(normalized))

    @property
    def is_source_restricted(self) -> bool:
        return bool(self.allowed_sources)

    def allows_source(self, source: Any) -> bool:
        if not self.allowed_sources:
            return True
        normalized = "" if source is None else str(source)
        return normalized in self.allowed_sources

    def allows_document(self, doc: Mapping[str, Any]) -> bool:
        meta_value = doc.get("meta")
        meta = meta_value if isinstance(meta_value, Mapping) else {}
        if meta.get("source_type") == "derived_knowledge":
            return self.include_derived_knowledge and self.allows_source(
                meta.get("related_source")
            )
        return self.allows_source(meta.get("source"))
