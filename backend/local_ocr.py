"""Optional local OCR support for Feishu image ingestion."""

from __future__ import annotations

import asyncio
import importlib
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional


class LocalOCRUnavailable(RuntimeError):
    """Raised when the optional local OCR runtime cannot be used."""


class LocalOCRInvalidResult(RuntimeError):
    """Raised when OCR completed but did not produce useful text."""


def _clean_text(value: Any) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value or ""))
    return " ".join(text.split()).strip()


def is_meaningful_ocr_text(text: str, scores: Iterable[float] = ()) -> bool:
    """Reject empty, symbol-only and consistently low-confidence OCR output."""
    visible = [char for char in text if not char.isspace()]
    meaningful = [char for char in visible if char.isalnum()]
    if len(meaningful) < 2:
        return False
    if len(meaningful) / max(len(visible), 1) < 0.25:
        return False
    score_list = list(scores)
    return not score_list or sum(score_list) / len(score_list) >= 0.45


def _score(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def _extract_items(result: Any) -> tuple[list[str], list[float]]:
    """Read RapidOCR 3.x output and the legacy ``(items, elapsed)`` shape."""
    if hasattr(result, "txts"):
        texts = list(getattr(result, "txts", None) or [])
        raw_scores = list(getattr(result, "scores", None) or [])
        lines: list[str] = []
        scores: list[float] = []
        for index, value in enumerate(texts):
            text = _clean_text(value)
            score = _score(raw_scores[index]) if index < len(raw_scores) else None
            if text and (score is None or score >= 0.35):
                lines.append(text)
                if score is not None:
                    scores.append(score)
        return lines, scores

    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if result is None:
        return [], []
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return [], []

    lines = []
    scores = []
    for item in result:
        text: Any = ""
        score: Optional[float] = None
        if isinstance(item, dict):
            text = item.get("text") or item.get("txt") or ""
            score = _score(item.get("score"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) >= 3:
                text = item[1]
                score = _score(item[2])
            elif len(item) == 2 and isinstance(item[0], str):
                text = item[0]
                score = _score(item[1])
        elif isinstance(item, str):
            text = item
        cleaned = _clean_text(text)
        if cleaned and (score is None or score >= 0.35):
            lines.append(cleaned)
            if score is not None:
                scores.append(score)
    return lines, scores


class LocalRapidOCR:
    """Lazy, reusable RapidOCR adapter whose blocking work runs off-loop."""

    def __init__(self, engine_factory: Optional[Callable[[], Any]] = None):
        self._engine_factory = engine_factory
        self._engine: Any = None
        self._engine_error = ""
        self._initialization_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def initialized(self) -> bool:
        return self._engine is not None

    def _create_engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        try:
            module = importlib.import_module("rapidocr")
            engine_class = getattr(module, "RapidOCR")
            return engine_class()
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise LocalOCRUnavailable(
                "本地 OCR 可选依赖未安装，请安装 rapidocr 与 onnxruntime"
            ) from exc
        except Exception as exc:
            raise LocalOCRUnavailable(f"本地 OCR 初始化失败：{str(exc)[:160]}") from exc

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_error:
            raise LocalOCRUnavailable(self._engine_error)
        with self._initialization_lock:
            if self._engine is not None:
                return self._engine
            try:
                self._engine = self._create_engine()
            except LocalOCRUnavailable as exc:
                self._engine_error = str(exc)
                raise
        return self._engine

    def _recognize_sync(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise LocalOCRInvalidResult("本地 OCR 未收到图片内容")
        engine = self._get_engine()
        try:
            with self._inference_lock:
                result = engine(image_bytes)
        except Exception as exc:
            raise RuntimeError(f"本地 OCR 识别失败：{str(exc)[:160]}") from exc

        lines, scores = _extract_items(result)
        text = "\n".join(lines).strip()
        if not is_meaningful_ocr_text(text, scores):
            raise LocalOCRInvalidResult("本地 OCR 未识别到有效文字")
        return text

    async def recognize(self, image_bytes: bytes) -> str:
        return await asyncio.to_thread(self._recognize_sync, bytes(image_bytes))


local_rapidocr = LocalRapidOCR()
