"""OCR-aware document preparation for restart-safe knowledge ingestion jobs."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from backend.knowledge_base.document_loader import load_document
from backend.local_ocr import LocalOCRInvalidResult, LocalOCRUnavailable, local_rapidocr


IMAGE_DOCUMENT_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


class DocumentOCRFailure(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _has_meaningful_text(value: str, minimum: int = 12) -> bool:
    normalized = re.sub(r"\s+", "", str(value or ""))
    return len([char for char in normalized if char.isalnum()]) >= minimum


def render_pdf_pages(file_path: str, page_indexes: Sequence[int]) -> Dict[int, bytes]:
    """Render selected PDF pages to PNG without requiring a system Poppler install."""
    try:
        import pymupdf
    except (ImportError, ModuleNotFoundError) as exc:
        raise DocumentOCRFailure(
            "扫描 PDF 需要安装 PyMuPDF 渲染依赖",
            code="DOCUMENT_OCR_RENDERER_UNAVAILABLE", retryable=False,
        ) from exc

    rendered: Dict[int, bytes] = {}
    try:
        document = pymupdf.open(file_path)
        try:
            for page_index in page_indexes:
                page = document.load_page(int(page_index))
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                rendered[int(page_index)] = pixmap.tobytes("png")
        finally:
            document.close()
    except DocumentOCRFailure:
        raise
    except Exception as exc:
        raise DocumentOCRFailure(
            f"扫描 PDF 页面渲染失败：{str(exc)[:160]}",
            code="DOCUMENT_OCR_RENDER_FAILED", retryable=True,
        ) from exc
    return rendered


def _summary(
    *, status: str, method: str, page_count: int, ocr_pages: Sequence[int],
    failed_pages: Sequence[int] = (), warning: str = "",
) -> Dict[str, Any]:
    return {
        "extraction_status": status,
        "extraction_method": method,
        "page_count": int(page_count),
        "ocr_page_count": len(ocr_pages),
        "ocr_pages": [int(index) for index in ocr_pages],
        "ocr_failed_pages": [int(index) for index in failed_pages],
        "extraction_warning": str(warning or "")[:500],
    }


async def prepare_document_for_ingestion(
    file_path: str,
    original_filename: str,
    *,
    report: Optional[Callable[[str, int, str], None]] = None,
    ocr: Any = local_rapidocr,
    document_loader: Callable[[str, str], List[Document]] = load_document,
    pdf_renderer: Callable[[str, Sequence[int]], Dict[int, bytes]] = render_pdf_pages,
    max_ocr_pages: int = 100,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Load a document and OCR only image files or PDF pages lacking native text."""
    extension = Path(original_filename).suffix.lower().lstrip(".")
    file_name = Path(original_filename).name

    if extension in IMAGE_DOCUMENT_EXTENSIONS:
        if report:
            report("document_ocr_recognize", 35, "正在识别图片文档中的文字")
        try:
            image_bytes = await asyncio.to_thread(Path(file_path).read_bytes)
            text = await ocr.recognize(image_bytes)
        except LocalOCRUnavailable as exc:
            raise DocumentOCRFailure(
                str(exc), code="DOCUMENT_OCR_UNAVAILABLE", retryable=False,
            ) from exc
        except LocalOCRInvalidResult as exc:
            raise DocumentOCRFailure(
                "图片文档未识别到有效文字", code="DOCUMENT_OCR_EMPTY", retryable=False,
            ) from exc
        except Exception as exc:
            raise DocumentOCRFailure(
                f"图片文档识别失败：{str(exc)[:160]}",
                code="DOCUMENT_OCR_FAILED", retryable=True,
            ) from exc
        document = Document(
            page_content=text,
            metadata={
                "source_file": file_name, "source": file_name, "page": 0,
                "ocr_applied": True, "extraction_method": "local_rapidocr",
            },
        )
        return [document], _summary(
            status="completed", method="local_rapidocr", page_count=1, ocr_pages=[0],
        )

    if report:
        report("document_parse", 20, "正在读取文档结构与原生文本")
    documents = await asyncio.to_thread(document_loader, file_path, original_filename)
    if extension != "pdf":
        return documents, _summary(
            status="native", method="native_text", page_count=len(documents), ocr_pages=[],
        )

    sparse_indexes = [
        index for index, document in enumerate(documents)
        if not _has_meaningful_text(document.page_content)
    ]
    if not sparse_indexes:
        return documents, _summary(
            status="native", method="pdf_text", page_count=len(documents), ocr_pages=[],
        )

    page_limit = max(1, int(max_ocr_pages))
    target_indexes = sparse_indexes[:page_limit]
    skipped_indexes = sparse_indexes[page_limit:]
    if report:
        report("document_ocr_render", 30, f"正在渲染 {len(target_indexes)} 个扫描页面")
    try:
        rendered = await asyncio.to_thread(pdf_renderer, file_path, target_indexes)
    except DocumentOCRFailure:
        if any(_has_meaningful_text(document.page_content) for document in documents):
            return documents, _summary(
                status="partial", method="pdf_text", page_count=len(documents), ocr_pages=[],
                failed_pages=sparse_indexes, warning="扫描页面未能渲染，已保留原生文本页面",
            )
        raise

    recognized_pages: List[int] = []
    failed_pages: List[int] = list(skipped_indexes)
    errors: List[Exception] = []
    if report:
        report("document_ocr_recognize", 40, f"正在识别 {len(target_indexes)} 个扫描页面")
    for position, page_index in enumerate(target_indexes, start=1):
        if report:
            progress = 40 + min(10, int(position * 10 / max(len(target_indexes), 1)))
            report(
                "document_ocr_recognize", progress,
                f"正在识别扫描页面 {position} / {len(target_indexes)}",
            )
        image_bytes = rendered.get(page_index, b"")
        try:
            text = await ocr.recognize(image_bytes)
            documents[page_index].page_content = text
            documents[page_index].metadata.update({
                "ocr_applied": True, "extraction_method": "local_rapidocr",
            })
            recognized_pages.append(page_index)
        except Exception as exc:
            failed_pages.append(page_index)
            errors.append(exc)

    has_text = any(_has_meaningful_text(document.page_content) for document in documents)
    if not has_text:
        unavailable = next((error for error in errors if isinstance(error, LocalOCRUnavailable)), None)
        invalid_only = bool(errors) and all(isinstance(error, LocalOCRInvalidResult) for error in errors)
        if unavailable:
            raise DocumentOCRFailure(
                str(unavailable), code="DOCUMENT_OCR_UNAVAILABLE", retryable=False,
            ) from unavailable
        if invalid_only:
            raise DocumentOCRFailure(
                "扫描 PDF 未识别到有效文字", code="DOCUMENT_OCR_EMPTY", retryable=False,
            )
        detail = str(errors[0])[:160] if errors else "页面渲染结果为空"
        raise DocumentOCRFailure(
            f"扫描 PDF 识别失败：{detail}", code="DOCUMENT_OCR_FAILED", retryable=True,
        )

    status = "partial" if failed_pages else "completed"
    method = "pdf_text+local_rapidocr" if recognized_pages else "pdf_text"
    warning = ""
    if skipped_indexes:
        warning = f"超过单任务 {page_limit} 页 OCR 上限，其余扫描页面待拆分后重新上传"
    elif failed_pages:
        warning = f"第 {', '.join(str(index + 1) for index in failed_pages)} 页未识别到有效文字"
    return documents, _summary(
        status=status, method=method, page_count=len(documents),
        ocr_pages=recognized_pages, failed_pages=failed_pages, warning=warning,
    )


__all__ = [
    "DocumentOCRFailure", "IMAGE_DOCUMENT_EXTENSIONS", "prepare_document_for_ingestion",
    "render_pdf_pages",
]
