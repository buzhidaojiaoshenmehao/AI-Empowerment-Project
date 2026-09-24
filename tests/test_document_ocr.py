import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from backend.document_ocr import DocumentOCRFailure, prepare_document_for_ingestion
from backend.local_ocr import LocalOCRInvalidResult, LocalOCRUnavailable


class _OCR:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def recognize(self, image_bytes):
        self.calls.append(image_bytes)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class DocumentOCRTest(unittest.TestCase):
    def test_text_pdf_skips_render_and_ocr(self):
        ocr = _OCR([])
        renderer_called = []

        def loader(path, filename):
            return [Document(page_content="这是包含足够有效文字的原生 PDF 页面内容", metadata={"page": 0})]

        def renderer(path, indexes):
            renderer_called.append(indexes)
            return {}

        documents, summary = asyncio.run(prepare_document_for_ingestion(
            "/tmp/staged.pending", "guide.pdf", ocr=ocr,
            document_loader=loader, pdf_renderer=renderer,
        ))
        self.assertEqual(summary["extraction_status"], "native")
        self.assertEqual(summary["extraction_method"], "pdf_text")
        self.assertEqual(documents[0].metadata["page"], 0)
        self.assertFalse(renderer_called)
        self.assertFalse(ocr.calls)

    def test_scanned_pdf_page_is_rendered_and_recognized(self):
        ocr = _OCR(["扫描页面识别内容包含足够多的有效文字"])

        documents, summary = asyncio.run(prepare_document_for_ingestion(
            "/tmp/staged.pending", "scan.pdf", ocr=ocr,
            document_loader=lambda path, filename: [Document(page_content="", metadata={"page": 0})],
            pdf_renderer=lambda path, indexes: {0: b"rendered-page"},
        ))
        self.assertEqual(documents[0].page_content, "扫描页面识别内容包含足够多的有效文字")
        self.assertTrue(documents[0].metadata["ocr_applied"])
        self.assertEqual(summary["extraction_status"], "completed")
        self.assertEqual(summary["ocr_pages"], [0])
        self.assertEqual(ocr.calls, [b"rendered-page"])

    def test_mixed_pdf_keeps_native_text_and_marks_failed_ocr_page_partial(self):
        ocr = _OCR([LocalOCRInvalidResult("empty")])
        documents, summary = asyncio.run(prepare_document_for_ingestion(
            "/tmp/staged.pending", "mixed.pdf", ocr=ocr,
            document_loader=lambda path, filename: [
                Document(page_content="这是原生文本页面，内容足够继续完成知识入库", metadata={"page": 0}),
                Document(page_content="", metadata={"page": 1}),
            ],
            pdf_renderer=lambda path, indexes: {1: b"blank-page"},
        ))
        self.assertEqual(summary["extraction_status"], "partial")
        self.assertEqual(summary["ocr_failed_pages"], [1])
        self.assertIn("第 2 页", summary["extraction_warning"])
        self.assertTrue(documents[0].page_content)

    def test_image_document_becomes_searchable_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "upload.pending"
            path.write_bytes(b"image-bytes")
            ocr = _OCR(["图片文档识别后形成可检索的项目知识内容"])
            documents, summary = asyncio.run(prepare_document_for_ingestion(
                str(path), "whiteboard.png", ocr=ocr,
            ))
        self.assertEqual(documents[0].metadata["source_file"], "whiteboard.png")
        self.assertEqual(documents[0].metadata["extraction_method"], "local_rapidocr")
        self.assertEqual(summary["extraction_status"], "completed")
        self.assertEqual(summary["page_count"], 1)

    def test_scanned_pdf_page_limit_marks_remaining_pages_partial(self):
        ocr = _OCR([
            "第一页扫描内容已经成功识别并可检索",
            "第二页扫描内容已经成功识别并可检索",
        ])
        documents, summary = asyncio.run(prepare_document_for_ingestion(
            "/tmp/staged.pending", "large-scan.pdf", ocr=ocr, max_ocr_pages=2,
            document_loader=lambda path, filename: [
                Document(page_content="", metadata={"page": index}) for index in range(3)
            ],
            pdf_renderer=lambda path, indexes: {index: f"page-{index}".encode() for index in indexes},
        ))
        self.assertEqual(summary["extraction_status"], "partial")
        self.assertEqual(summary["ocr_pages"], [0, 1])
        self.assertEqual(summary["ocr_failed_pages"], [2])
        self.assertIn("2 页 OCR 上限", summary["extraction_warning"])
        self.assertFalse(documents[2].page_content)

    def test_fully_scanned_pdf_without_ocr_runtime_fails_explicitly(self):
        ocr = _OCR([LocalOCRUnavailable("依赖未安装")])
        with self.assertRaises(DocumentOCRFailure) as raised:
            asyncio.run(prepare_document_for_ingestion(
                "/tmp/staged.pending", "scan.pdf", ocr=ocr,
                document_loader=lambda path, filename: [Document(page_content="", metadata={"page": 0})],
                pdf_renderer=lambda path, indexes: {0: b"rendered-page"},
            ))
        self.assertEqual(raised.exception.code, "DOCUMENT_OCR_UNAVAILABLE")
        self.assertFalse(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
