import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, UploadFile

from backend import main
from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity, get_current_identity
from backend.document_ocr import DocumentOCRFailure
from backend.processing_jobs import ProcessingJobError
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class _Context:
    def __init__(self):
        self.events = []

    def report(self, stage, progress, message):
        self.events.append((stage, progress, message))


class DocumentIngestionJobTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.staging = self.root / ".staging"
        self.staging.mkdir()
        self.repository = StorageRepository(Database(self.root / "app.db"))
        member = self.repository.provision_project_member(
            "uploader@example.com", "上传者", "project_member", status="active",
        )
        self.identity = Identity(
            user_id=member["user_id"], email=member["email"], display_name=member["display_name"],
            organization_id="org_default", project_id=member["project_id"],
            project_name=member["project_name"], role=member["role"],
            permissions=ROLE_PERMISSIONS[member["role"]],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_async_upload_persists_private_staging_file_and_returns_job(self):
        service = Mock()
        service.enqueue.return_value = {
            "job_id": "job_upload", "status": "queued", "created": True,
        }
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploaded = UploadFile(filename="guide.md", file=io.BytesIO(b"project guide"))
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_document_upload(request, uploaded, category="项目规范"))
        self.assertTrue(result["success"])
        self.assertEqual(result["job"]["job_id"], "job_upload")
        staged = list(self.staging.iterdir())
        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].read_bytes(), b"project guide")
        payload = service.enqueue.call_args.args[1]
        self.assertNotIn("project guide", str(payload))
        self.assertEqual(service.enqueue.call_args.args[0], "document_ingestion")

    def test_async_upload_accepts_image_for_background_ocr(self):
        service = Mock()
        service.enqueue.return_value = {
            "job_id": "job_image", "status": "queued", "created": True,
        }
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploaded = UploadFile(filename="whiteboard.png", file=io.BytesIO(b"png bytes"))
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_document_upload(request, uploaded))
        self.assertTrue(result["success"])
        payload = service.enqueue.call_args.args[1]
        self.assertEqual(payload["original_filename"], "whiteboard.png")
        self.assertNotIn("png bytes", str(payload))

    def test_batch_upload_creates_independent_jobs_and_private_staging_files(self):
        service = Mock()
        service.enqueue.side_effect = [
            {"job_id": "job_one", "status": "queued", "created": True},
            {"job_id": "job_two", "status": "queued", "created": True},
        ]
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploads = [
            UploadFile(filename="guide.md", file=io.BytesIO(b"project guide")),
            UploadFile(filename="rules.txt", file=io.BytesIO(b"project rules")),
        ]
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_document_upload_batch(
                request, uploads, category="项目规范",
            ))
        self.assertEqual(result["submitted_count"], 2)
        self.assertEqual(result["accepted_count"], 2)
        self.assertEqual([job["job_id"] for job in result["jobs"]], ["job_one", "job_two"])
        staged = sorted(self.staging.iterdir())
        self.assertEqual(len(staged), 2)
        self.assertEqual({path.read_bytes() for path in staged}, {b"project guide", b"project rules"})
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in staged))
        self.assertEqual(service.enqueue.call_count, 2)
        for call in service.enqueue.call_args_list:
            self.assertEqual(call.args[0], "document_ingestion")
            self.assertNotIn("project guide", str(call.args[1]))
            self.assertNotIn("project rules", str(call.args[1]))

    def test_batch_upload_rejects_duplicate_names_before_staging(self):
        service = Mock()
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploads = [
            UploadFile(filename="Guide.md", file=io.BytesIO(b"first")),
            UploadFile(filename="guide.md", file=io.BytesIO(b"second")),
        ]
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.enqueue_document_upload_batch(request, uploads))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(list(self.staging.iterdir()))
        service.enqueue.assert_not_called()

    def test_batch_upload_rejects_file_count_before_reading_or_staging(self):
        service = Mock()
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploads = [
            UploadFile(filename="one.md", file=io.BytesIO(b"one")),
            UploadFile(filename="two.md", file=io.BytesIO(b"two")),
        ]
        with patch.object(main.settings, "MAX_BATCH_UPLOAD_FILES", 1), \
                patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.enqueue_document_upload_batch(request, uploads))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(list(self.staging.iterdir()))
        service.enqueue.assert_not_called()

    def test_batch_upload_keeps_accepted_item_when_another_cannot_enqueue(self):
        service = Mock()
        service.enqueue.side_effect = [
            {"job_id": "job_one", "status": "queued", "created": True},
            RuntimeError("queue unavailable"),
        ]
        request = SimpleNamespace(state=SimpleNamespace(identity=self.identity))
        uploads = [
            UploadFile(filename="one.md", file=io.BytesIO(b"one")),
            UploadFile(filename="two.md", file=io.BytesIO(b"two")),
        ]
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_document_upload_batch(request, uploads))
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["items"][1]["error"], "任务受理失败，请重新提交该文件")
        self.assertEqual(len(list(self.staging.iterdir())), 1)

    def test_document_job_rechecks_identity_and_removes_staging_after_publish(self):
        staging_name = "upload_test.pending"
        (self.staging / staging_name).write_bytes(b"project guide")
        context = _Context()
        published = {
            "success": True, "filename": "guide.md", "chunks": 2,
            "asset": {"asset_id": "asset_1"},
        }
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "prepare_document_for_ingestion", new=AsyncMock(return_value=([], {
                    "extraction_status": "native", "extraction_method": "native_text",
                }))), \
                patch.object(main, "_upload_document_impl", new=AsyncMock(return_value=published)) as upload:
            result = asyncio.run(main._job_document_ingestion(context, {
                "staging_name": staging_name, "original_filename": "guide.md",
                "actor_user_id": self.identity.user_id, "category": "项目规范",
            }))
        self.assertTrue(result["_job_commit_on_cancel"])
        self.assertFalse((self.staging / staging_name).exists())
        self.assertIsNone(get_current_identity())
        self.assertEqual(upload.await_count, 1)
        self.assertEqual([item[0] for item in context.events], ["document_validate", "document_ingest"])

    def test_document_job_keeps_staging_file_when_publish_fails_for_retry(self):
        staging_name = "upload_retry.pending"
        (self.staging / staging_name).write_bytes(b"retry body")
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "prepare_document_for_ingestion", new=AsyncMock(return_value=([], {
                    "extraction_status": "native", "extraction_method": "native_text",
                }))), \
                patch.object(main, "_upload_document_impl", new=AsyncMock(side_effect=RuntimeError("temporary"))):
            with self.assertRaises(RuntimeError):
                asyncio.run(main._job_document_ingestion(_Context(), {
                    "staging_name": staging_name, "original_filename": "retry.md",
                    "actor_user_id": self.identity.user_id,
                }))
        self.assertTrue((self.staging / staging_name).exists())
        self.assertIsNone(get_current_identity())

    def test_document_job_maps_ocr_failure_and_keeps_staging_for_manual_retry(self):
        staging_name = "upload_scan.pending"
        (self.staging / staging_name).write_bytes(b"scanned pdf")
        failure = DocumentOCRFailure(
            "扫描 PDF 需要安装 OCR", code="DOCUMENT_OCR_UNAVAILABLE", retryable=False,
        )
        with patch.object(main, "UPLOAD_STAGING_DIR", self.staging), \
                patch.object(main, "get_repository", return_value=self.repository), \
                patch.object(main, "prepare_document_for_ingestion", new=AsyncMock(side_effect=failure)):
            with self.assertRaises(ProcessingJobError) as raised:
                asyncio.run(main._job_document_ingestion(_Context(), {
                    "staging_name": staging_name, "original_filename": "scan.pdf",
                    "actor_user_id": self.identity.user_id,
                }))
        self.assertEqual(raised.exception.code, "DOCUMENT_OCR_UNAVAILABLE")
        self.assertFalse(raised.exception.retryable)
        self.assertTrue((self.staging / staging_name).exists())
        self.assertIsNone(get_current_identity())


if __name__ == "__main__":
    unittest.main()
