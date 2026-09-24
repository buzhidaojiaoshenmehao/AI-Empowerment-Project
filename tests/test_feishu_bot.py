import hashlib
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend import main
from backend.config import settings
from backend.feishu_bot import (
    FeishuBot,
    _reset_webhook_replay_cache_for_tests,
    handle_feishu_webhook,
)
from backend.local_ocr import LocalOCRUnavailable
from backend.knowledge_base.llm_service import LLMService


class FeishuBotMessageTest(unittest.TestCase):
    def test_direct_message_keeps_text_and_chat_type(self):
        message = FeishuBot().parse_feishu_message({
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_member"}},
                "message": {
                    "message_id": "om_direct",
                    "chat_id": "oc_direct",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": '{"text":"项目风险有哪些？"}',
                },
            },
        })
        self.assertEqual(message["chat_type"], "p2p")
        self.assertEqual(message["text"], "项目风险有哪些？")
        self.assertFalse(message["mention_bot"])

    def test_group_mention_is_removed_from_query(self):
        message = FeishuBot().parse_feishu_message({
            "event": {
                "message": {
                    "message_id": "om_group",
                    "chat_id": "oc_group",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text":"@_user_1 新人先读哪些资料？"}',
                },
            },
        })
        self.assertTrue(message["mention_bot"])
        self.assertEqual(message["text"], "新人先读哪些资料？")

    def test_post_message_is_flattened_for_reply_and_archiving(self):
        message = FeishuBot().parse_feishu_message({
            "event": {
                "message": {
                    "message_id": "om_post",
                    "chat_id": "oc_group",
                    "chat_type": "group",
                    "message_type": "post",
                    "content": '{"title":"风险同步","content":[[{"tag":"text","text":"结论：回滚方案待确认"},{"tag":"img","image_key":"img_a"}]]}',
                },
            },
        })
        self.assertEqual(message["message_type"], "post")
        self.assertIn("风险同步", message["text"])
        self.assertEqual(message["image_keys"], ["img_a"])


class FeishuBotApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_unmapped_sender_is_denied_before_rag(self):
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_unknown"}},
                "message": {
                    "message_id": "om_denied", "chat_id": "oc_restricted", "chat_type": "group",
                    "message_type": "text", "content": '{"text":"@_user_1 项目风险有哪些？"}',
                },
            },
        }
        group = {
            "chat_id": "oc_restricted", "collection_mode": "off", "bot_enabled": True,
            "visibility": "restricted", "access_policy_id": "policy_group",
        }
        with (
            patch("backend.feishu_workspace.feishu_workspace.get_group", return_value=group),
            patch("backend.feishu_workspace.feishu_workspace.record_event") as record_event,
            patch("backend.feishu_bot._resolve_feishu_rag_identity", new=AsyncMock(return_value=None)),
            patch.object(FeishuBot, "reply_text_message", new=AsyncMock(return_value=True)),
            patch("backend.knowledge_base.llm_service.llm_service.chat", side_effect=AssertionError("RAG 不应执行")),
        ):
            result = await handle_feishu_webhook(payload)
        self.assertEqual(result["action"], "rag_access_denied")
        self.assertEqual(record_event.call_args.args[1], "rag_access_denied")

    async def test_chat_history_uses_feishu_chat_container_type(self):
        calls = []

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"code": 0, "data": {"items": [], "page_token": ""}}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, **kwargs):
                calls.append((url, kwargs))
                return Response()

        bot = FeishuBot()

        async def access_token():
            return "test-token"

        bot._get_app_access_token = access_token
        bot._get_client = lambda timeout=10.0: Client()

        messages, next_page_token = await bot.list_chat_messages("oc_project")

        self.assertEqual(messages, [])
        self.assertEqual(next_page_token, "")
        self.assertEqual(calls[0][1]["params"]["container_id_type"], "chat")
        self.assertEqual(calls[0][1]["params"]["container_id"], "oc_project")

    async def test_message_image_download_uses_resource_endpoint(self):
        calls = []

        class Response:
            status_code = 200
            content = b"\x89PNG\r\n\x1a\ncontent"
            text = ""
            headers = {"content-type": "image/png"}

            @staticmethod
            def json():
                return {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, **kwargs):
                calls.append((url, kwargs))
                return Response()

        bot = FeishuBot()

        async def access_token():
            return "test-token"

        bot._get_app_access_token = access_token
        bot._get_client = lambda timeout=10.0: Client()
        result = await bot.download_message_resource("om_image", "img_a")

        self.assertTrue(calls[0][0].endswith("/messages/om_image/resources/img_a"))
        self.assertEqual(calls[0][1]["params"], {"type": "image"})
        self.assertEqual(result["content_type"], "image/png")

    async def test_feishu_ocr_returns_joined_text_regions(self):
        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"code": 0, "data": {"text_list": ["发布结论", "回滚方案待确认"]}}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs
                return Response()

        client = Client()
        bot = FeishuBot()

        async def access_token():
            return "test-token"

        bot._get_app_access_token = access_token
        bot._get_client = lambda timeout=10.0: client
        result = await bot.recognize_image_text(b"small-image")

        self.assertIn("optical_char_recognition", client.url)
        self.assertIn("image", client.kwargs["json"])
        self.assertEqual(result, "发布结论\n回滚方案待确认")

    async def test_feishu_ocr_retries_frequency_limit_then_succeeds(self):
        class Response:
            status_code = 200
            text = ""

            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        class Client:
            def __init__(self):
                self.calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return Response({"code": 99991400, "msg": "request trigger frequency limit"})
                return Response({"code": 0, "data": {"text_list": ["重试成功"]}})

        client = Client()
        bot = FeishuBot()

        async def access_token():
            return "test-token"

        bot._get_app_access_token = access_token
        bot._get_client = lambda timeout=10.0: client
        with patch("backend.feishu_bot.asyncio.sleep", new=AsyncMock()):
            result = await bot.recognize_image_text(b"small-image")

        self.assertEqual(client.calls, 2)
        self.assertEqual(result, "重试成功")

    async def test_local_ocr_is_used_before_remote_channels(self):
        bot = FeishuBot()
        bot._local_ocr = type("LocalOCR", (), {
            "recognize": AsyncMock(return_value="发布结论\n回滚方案待确认"),
        })()
        bot.recognize_image_text = AsyncMock(side_effect=AssertionError("不应调用飞书 OCR"))

        text, method = await bot.extract_image_content(b"image", "image/png")

        self.assertEqual(method, "local_rapidocr")
        self.assertIn("回滚方案待确认", text)
        bot.recognize_image_text.assert_not_awaited()

    async def test_missing_local_ocr_falls_back_to_feishu_ocr(self):
        bot = FeishuBot()
        bot._local_ocr = type("LocalOCR", (), {
            "recognize": AsyncMock(side_effect=LocalOCRUnavailable("本地 OCR 可选依赖未安装")),
        })()
        bot.recognize_image_text = AsyncMock(return_value="飞书识别成功")

        text, method = await bot.extract_image_content(b"image", "image/png")

        self.assertEqual(method, "feishu_ocr")
        self.assertIn("飞书识别成功", text)

    async def test_failed_ocr_channels_fall_back_to_vision_model(self):
        bot = FeishuBot()
        bot._local_ocr = type("LocalOCR", (), {
            "recognize": AsyncMock(side_effect=RuntimeError("本地识别失败")),
        })()
        bot.recognize_image_text = AsyncMock(side_effect=RuntimeError("飞书限流"))

        with patch(
            "backend.knowledge_base.llm_service.llm_service.extract_image_knowledge",
            new=AsyncMock(return_value="### 图片说明\n\n这是发布流程截图"),
        ):
            text, method = await bot.extract_image_content(b"image", "image/png")

        self.assertEqual(method, "vision_model")
        self.assertIn("发布流程截图", text)

    async def test_invalid_feishu_ocr_result_falls_back_to_vision_model(self):
        bot = FeishuBot()
        bot._local_ocr = type("LocalOCR", (), {
            "recognize": AsyncMock(side_effect=RuntimeError("本地未识别")),
        })()
        bot.recognize_image_text = AsyncMock(return_value="... ...")

        with patch(
            "backend.knowledge_base.llm_service.llm_service.extract_image_knowledge",
            new=AsyncMock(return_value="### 图片说明\n\n这是风险列表截图"),
        ):
            text, method = await bot.extract_image_content(b"image", "image/png")

        self.assertEqual(method, "vision_model")
        self.assertIn("风险列表截图", text)

    async def test_all_channel_errors_are_aggregated_without_secrets(self):
        original_api_key = settings.OPENAI_API_KEY
        original_secret = settings.FEISHU_APP_SECRET
        try:
            settings.OPENAI_API_KEY = "sk-sensitive-value"
            settings.FEISHU_APP_SECRET = "feishu-sensitive-value"
            bot = FeishuBot()
            bot._local_ocr = type("LocalOCR", (), {
                "recognize": AsyncMock(side_effect=RuntimeError("本地失败 sk-sensitive-value")),
            })()
            bot.recognize_image_text = AsyncMock(
                side_effect=RuntimeError("飞书失败 feishu-sensitive-value")
            )

            with patch(
                "backend.knowledge_base.llm_service.llm_service.extract_image_knowledge",
                new=AsyncMock(side_effect=RuntimeError("Bearer sk-sensitive-value")),
            ):
                with self.assertRaises(RuntimeError) as context:
                    await bot.extract_image_content(b"image", "image/png")

            detail = str(context.exception)
            self.assertIn("本地失败", detail)
            self.assertIn("飞书失败", detail)
            self.assertNotIn("sk-sensitive-value", detail)
            self.assertNotIn("feishu-sensitive-value", detail)
        finally:
            settings.OPENAI_API_KEY = original_api_key
            settings.FEISHU_APP_SECRET = original_secret


class FeishuWebhookSecurityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_secret = settings.FEISHU_WEBHOOK_SECRET
        self.original_token = settings.FEISHU_VERIFICATION_TOKEN
        _reset_webhook_replay_cache_for_tests()

    async def asyncTearDown(self):
        settings.FEISHU_WEBHOOK_SECRET = self.original_secret
        settings.FEISHU_VERIFICATION_TOKEN = self.original_token
        _reset_webhook_replay_cache_for_tests()

    @staticmethod
    def _request(raw_body: bytes, headers: dict):
        class Request:
            method = "POST"

            async def body(self):
                return raw_body

        request = Request()
        request.headers = headers
        return request

    async def test_signed_webhook_uses_raw_body_and_rejects_replay(self):
        settings.FEISHU_WEBHOOK_SECRET = "callback-secret"
        settings.FEISHU_VERIFICATION_TOKEN = ""
        raw = json.dumps({"challenge": "ok"}, separators=(",", ":")).encode("utf-8")
        timestamp = str(int(time.time()))
        nonce = "nonce-1"
        signature = hashlib.sha256(
            f"{timestamp}{nonce}callback-secret".encode("utf-8") + raw
        ).hexdigest()
        request = self._request(raw, {
            "X-Lark-Request-Timestamp": timestamp,
            "X-Lark-Request-Nonce": nonce,
            "X-Lark-Signature": signature,
        })
        with patch("backend.feishu_bot.handle_feishu_webhook", new=AsyncMock(return_value={"challenge": "ok"})) as handle:
            result = await main.feishu_webhook(request)
            self.assertEqual(result["challenge"], "ok")
            handle.assert_awaited_once()
            with self.assertRaises(HTTPException) as replay:
                await main.feishu_webhook(request)
        self.assertEqual(replay.exception.status_code, 401)
        self.assertIn("已处理", replay.exception.detail)

    async def test_webhook_is_fail_closed_when_unconfigured_or_expired(self):
        raw = b'{"event":{}}'
        settings.FEISHU_WEBHOOK_SECRET = ""
        settings.FEISHU_VERIFICATION_TOKEN = ""
        with patch("backend.feishu_bot.handle_feishu_webhook", new=AsyncMock()) as handle:
            with self.assertRaises(HTTPException) as missing:
                await main.feishu_webhook(self._request(raw, {}))
            handle.assert_not_awaited()
        self.assertEqual(missing.exception.status_code, 401)

        settings.FEISHU_WEBHOOK_SECRET = "callback-secret"
        timestamp = str(int(time.time()) - 301)
        nonce = "expired"
        signature = hashlib.sha256(
            f"{timestamp}{nonce}callback-secret".encode("utf-8") + raw
        ).hexdigest()
        with self.assertRaises(HTTPException) as expired:
            await main.feishu_webhook(self._request(raw, {
                "X-Lark-Request-Timestamp": timestamp,
                "X-Lark-Request-Nonce": nonce,
                "X-Lark-Signature": signature,
            }))
        self.assertEqual(expired.exception.status_code, 401)
        self.assertIn("过期", expired.exception.detail)

    async def test_official_verification_token_is_accepted_and_replayed_token_is_denied(self):
        settings.FEISHU_WEBHOOK_SECRET = ""
        settings.FEISHU_VERIFICATION_TOKEN = "official-token"
        raw = b'{"token":"official-token","challenge":"verified"}'
        request = self._request(raw, {})
        with patch("backend.feishu_bot.handle_feishu_webhook", new=AsyncMock(return_value={"challenge": "verified"})):
            result = await main.feishu_webhook(request)
            self.assertEqual(result["challenge"], "verified")
            with self.assertRaises(HTTPException) as replay:
                await main.feishu_webhook(request)
        self.assertIn("已处理", replay.exception.detail)


class ImageModelCapabilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_image_request_is_rejected_before_api_call(self):
        original_model = settings.LLM_MODEL
        original_base_url = settings.OPENAI_BASE_URL
        try:
            settings.LLM_MODEL = "deepseek-v4-flash"
            settings.OPENAI_BASE_URL = "https://api.deepseek.com"
            service = LLMService()
            service._non_stream_call = AsyncMock()

            with self.assertRaisesRegex(RuntimeError, "不支持图片输入"):
                await service.extract_image_knowledge(b"image", "image/jpeg")

            service._non_stream_call.assert_not_awaited()
        finally:
            settings.LLM_MODEL = original_model
            settings.OPENAI_BASE_URL = original_base_url


if __name__ == "__main__":
    unittest.main()
