from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
from typing import Any, Callable
import unicodedata

import pymupdf as fitz


MAX_OCR_OUTPUT_BYTES = 4 * 1024 * 1024
_LANGUAGES_RE = re.compile(r"^[A-Za-z0-9_+-]+$")


class OcrError(RuntimeError):
    pass


class OcrUnavailableError(OcrError):
    pass


class OcrTimeoutError(OcrError):
    pass


class OcrPageLimitError(OcrError):
    pass


class OcrExecutionError(OcrError):
    pass


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool = False
    provider: str = "tesseract"
    binary: str = "tesseract"
    languages: str = "eng+chi_sim"
    dpi: int = 300
    min_native_chars: int = 40
    max_pages: int = 100
    page_timeout_seconds: float = 30.0
    required: bool = False

    def __post_init__(self) -> None:
        if self.provider != "tesseract":
            raise ValueError(f"unsupported OCR provider: {self.provider}")
        if not self.binary.strip():
            raise ValueError("OCR binary must not be blank")
        if not _LANGUAGES_RE.fullmatch(self.languages):
            raise ValueError("OCR languages contain unsupported characters")
        if not 72 <= self.dpi <= 600:
            raise ValueError("OCR DPI must be between 72 and 600")
        if not 0 <= self.min_native_chars <= 5000:
            raise ValueError("OCR native-text threshold is outside its valid range")
        if not 1 <= self.max_pages <= 10000:
            raise ValueError("OCR page limit is outside its valid range")
        if not 0.1 <= self.page_timeout_seconds <= 300:
            raise ValueError("OCR page timeout is outside its valid range")

    @classmethod
    def from_settings(cls, settings: Any) -> "OcrConfig":
        return cls(
            enabled=settings.cogdoc_ocr_enabled,
            provider=settings.cogdoc_ocr_provider,
            binary=settings.cogdoc_ocr_binary,
            languages=settings.cogdoc_ocr_languages,
            dpi=settings.cogdoc_ocr_dpi,
            min_native_chars=settings.cogdoc_ocr_min_native_chars,
            max_pages=settings.cogdoc_ocr_max_pages,
            page_timeout_seconds=settings.cogdoc_ocr_page_timeout_seconds,
            required=settings.cogdoc_ocr_required,
        )


@dataclass(frozen=True)
class OcrDependencyStatus:
    enabled: bool
    available: bool
    provider: str
    reason: str


@dataclass(frozen=True)
class OcrPageResult:
    text: str
    provider: str


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text.replace("\x0c", ""))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in normalized.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def ocr_config_signature(settings: Any) -> str:
    config = OcrConfig.from_settings(settings)
    payload = {
        "enabled": config.enabled,
        "provider": config.provider,
        "binary": config.binary,
        "languages": config.languages,
        "dpi": config.dpi,
        "min_native_chars": config.min_native_chars,
        "max_pages": config.max_pages,
        "page_timeout_seconds": config.page_timeout_seconds,
        "required": config.required,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def probe_ocr_dependency(
    config: OcrConfig, *, which: Callable[[str], str | None] | None = None
) -> OcrDependencyStatus:
    if not config.enabled:
        return OcrDependencyStatus(False, True, config.provider, "disabled")
    resolver = which or shutil.which
    resolved = resolver(config.binary)
    if resolved is None and os.path.isfile(config.binary) and os.access(config.binary, os.X_OK):
        resolved = config.binary
    return OcrDependencyStatus(
        True,
        resolved is not None,
        config.provider,
        "available" if resolved is not None else "binary_not_found",
    )


class TesseractOcrEngine:
    def __init__(
        self,
        config: OcrConfig,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.config = config
        self._runner = runner
        self._dependency = probe_ocr_dependency(config)

    def extract(self, page: Any) -> OcrPageResult:
        if not self._dependency.available:
            raise OcrUnavailableError("configured OCR binary is unavailable")

        scale = self.config.dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = bytes(pixmap.tobytes("png"))
        command = [
            self.config.binary,
            "stdin",
            "stdout",
            "-l",
            self.config.languages,
            "--dpi",
            str(self.config.dpi),
        ]
        try:
            completed = self._runner(
                command,
                input=image,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.page_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise OcrUnavailableError("configured OCR binary is unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrTimeoutError("OCR page timeout exceeded") from exc

        if completed.returncode != 0:
            raise OcrExecutionError(
                f"OCR process exited with status {completed.returncode}"
            )
        raw = completed.stdout
        if isinstance(raw, str):
            encoded_size = len(raw.encode("utf-8"))
            text = raw
        else:
            encoded_size = len(raw)
            text = raw.decode("utf-8", errors="replace")
        if encoded_size > MAX_OCR_OUTPUT_BYTES:
            raise OcrExecutionError("OCR page output exceeds the safety limit")
        return OcrPageResult(normalize_ocr_text(text), self.config.provider)
