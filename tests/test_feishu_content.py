import json
import tempfile
import unittest
from pathlib import Path

from backend.feishu_content import FeishuContentProcessor, flatten_rich_text, message_content
from backend.feishu_workspace import evaluate_content


class FeishuRichTextTest(unittest.TestCase):
    def test_rich_text_keeps_structure_links_mentions_and_images(self):
        parsed = flatten_rich_text({
            "title": "发布结论",
            "content": [
                [
                    {"tag": "text", "text": "结论：采用灰度发布"},
                    {"tag": "a", "text": "查看方案", "href": "https://example.com/plan"},
                ],
                [
                    {"tag": "at", "user_name": "示例用户"},
                    {"tag": "img", "image_key": "img_a"},
                    {"tag": "code_block", "text": "deploy --gray"},
                ],
            ],
        })

        self.assertIn("# 发布结论", parsed["text"])
        self.assertIn("[查看方案](https://example.com/plan)", parsed["text"])
        self.assertIn("@示例用户", parsed["text"])
        self.assertIn("deploy --gray", parsed["text"])
        self.assertEqual(parsed["image_keys"], ["img_a"])
        self.assertEqual(parsed["links"][0]["href"], "https://example.com/plan")


class FeishuImageProcessingTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_is_stored_and_extracted(self):
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"test-image"

        async def download(message_id, resource_key):
            self.assertEqual((message_id, resource_key), ("om_image", "img_a"))
            return {"content": image_bytes, "content_type": "image/png"}

        async def extract(data, content_type):
            self.assertEqual(data, image_bytes)
            self.assertEqual(content_type, "image/png")
            return "### 识别文字\n\n发布风险需要确认回滚方案", "feishu_ocr"

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = FeishuContentProcessor(download, extract, Path(temp_dir))
            result = await processor.process_message({
                "message_id": "om_image",
                "msg_type": "image",
                "body": {"content": json.dumps({"image_key": "img_a"})},
            })
            content = message_content(result)

            self.assertEqual(content["extraction_status"], "completed")
            self.assertEqual(content["extraction_method"], "feishu_ocr")
            self.assertIn("发布风险", content["text"])
            self.assertEqual(content["resource_count"], 1)
            self.assertTrue((Path(temp_dir) / content["resource_files"][0]["stored_file"]).is_file())

    async def test_image_failure_is_explicit_and_requires_review(self):
        async def download(message_id, resource_key):
            raise RuntimeError("缺少 im:resource 权限")

        async def extract(data, content_type):
            raise AssertionError("下载失败后不应调用识别")

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = FeishuContentProcessor(download, extract, Path(temp_dir))
            result = await processor.process_message({
                "message_id": "om_failed",
                "msg_type": "image",
                "body": {"content": json.dumps({"image_key": "img_failed"})},
            })
            content = message_content(result)
            assessment = evaluate_content({
                "message_type": "image",
                "content": content["text"],
                "extraction_status": content["extraction_status"],
            }, {"collection_mode": "auto", "confidence_threshold": 0.85})

            self.assertEqual(content["extraction_status"], "failed")
            self.assertIn("im:resource", content["extraction_error"])
            self.assertEqual(assessment["content_status"], "review_required")
            self.assertIn("图片内容识别失败", assessment["reason"])

    async def test_original_image_is_kept_when_recognition_fails(self):
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"unreadable"

        async def download(message_id, resource_key):
            return {"content": image_bytes, "content_type": "image/png"}

        async def extract(data, content_type):
            raise RuntimeError("OCR 与多模态模型均不可用")

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = FeishuContentProcessor(download, extract, Path(temp_dir))
            result = await processor.process_message({
                "message_id": "om_keep",
                "msg_type": "image",
                "body": {"content": json.dumps({"image_key": "img_keep"})},
            })
            content = message_content(result)

            self.assertEqual(content["extraction_status"], "failed")
            self.assertEqual(content["resource_count"], 1)
            stored_file = content["resource_files"][0]["stored_file"]
            self.assertTrue((Path(temp_dir) / stored_file).is_file())


if __name__ == "__main__":
    unittest.main()
