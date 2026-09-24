"""
飞书集成模块 —— 与飞书进行消息、文档、群聊联动

功能：
1. 飞书机器人 Webhook 接收消息 → 调用 AI 知识库检索
2. 飞书消息卡片推送（知识日报、项目周报）
3. 飞书文档导入（通过飞书开放 API 读取云文档）
4. 飞书群聊自动回复（@机器人自动检索知识库）
"""
import asyncio
import base64
import json
import hashlib
import hmac
import logging
import re
import time
import threading
from typing import Optional, Dict, Any, Union
from datetime import datetime

import httpx
from backend.config import settings
from backend.feishu_content import (
    FeishuContentProcessor,
    flatten_rich_text,
    message_content,
    replace_message_content,
)
from backend.local_ocr import is_meaningful_ocr_text, local_rapidocr

logger = logging.getLogger(__name__)

_WEBHOOK_MAX_AGE_SECONDS = 300
_webhook_replay_lock = threading.Lock()
_webhook_replay_cache: Dict[str, float] = {}


class FeishuWebhookVerificationError(ValueError):
    pass


def _reset_webhook_replay_cache_for_tests() -> None:
    with _webhook_replay_lock:
        _webhook_replay_cache.clear()


def _claim_webhook_request(key: str, now: float) -> None:
    with _webhook_replay_lock:
        expired = [item for item, expires_at in _webhook_replay_cache.items() if expires_at <= now]
        for item in expired:
            _webhook_replay_cache.pop(item, None)
        if key in _webhook_replay_cache:
            raise FeishuWebhookVerificationError("飞书回调请求已处理")
        _webhook_replay_cache[key] = now + _WEBHOOK_MAX_AGE_SECONDS


def verify_feishu_webhook_request(
    raw_body: bytes,
    headers: Any,
    payload: Dict[str, Any],
    *,
    now: Optional[float] = None,
) -> str:
    """Verify an HTTP callback before any event parsing or side effect."""
    current = float(time.time() if now is None else now)
    normalized_headers = {str(key).lower(): str(value) for key, value in dict(headers).items()}
    timestamp = normalized_headers.get("x-lark-request-timestamp", "").strip()
    nonce = normalized_headers.get("x-lark-request-nonce", "").strip()
    signature = normalized_headers.get("x-lark-signature", "").strip()
    signing_secret = str(settings.FEISHU_WEBHOOK_SECRET or "").strip()
    verification_token = str(settings.FEISHU_VERIFICATION_TOKEN or "").strip()
    if not signing_secret and not verification_token:
        raise FeishuWebhookVerificationError("飞书回调验证未配置")

    if timestamp or nonce or signature:
        if not (timestamp and nonce and signature and signing_secret):
            raise FeishuWebhookVerificationError("飞书回调签名信息不完整")
        try:
            request_time = float(timestamp)
        except ValueError as exc:
            raise FeishuWebhookVerificationError("飞书回调时间戳无效") from exc
        if abs(current - request_time) > _WEBHOOK_MAX_AGE_SECONDS:
            raise FeishuWebhookVerificationError("飞书回调已过期")
        signed = f"{timestamp}{nonce}{signing_secret}".encode("utf-8") + raw_body
        expected = hashlib.sha256(signed).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise FeishuWebhookVerificationError("飞书回调签名验证失败")
        _claim_webhook_request(f"signature:{timestamp}:{nonce}:{signature}", current)
        return "signature"

    supplied_token = str(
        payload.get("token") or (payload.get("header") or {}).get("token") or ""
    ).strip()
    if not verification_token or not supplied_token or not hmac.compare_digest(
        verification_token, supplied_token,
    ):
        raise FeishuWebhookVerificationError("飞书回调 verification token 验证失败")
    event_id = str((payload.get("header") or {}).get("event_id") or "").strip()
    replay_key = event_id or hashlib.sha256(raw_body).hexdigest()
    _claim_webhook_request(f"token:{replay_key}", current)
    return "verification_token"


def _safe_error_message(exc: Exception) -> str:
    """Return a bounded integration error without credentials or bearer tokens."""
    detail = str(exc)
    for secret in (
        settings.OPENAI_API_KEY,
        settings.FEISHU_APP_SECRET,
        settings.FEISHU_WEBHOOK_SECRET,
        settings.FEISHU_VERIFICATION_TOKEN,
    ):
        if secret:
            detail = detail.replace(str(secret), "***")
    detail = re.sub(
        r"(?i)(api[_ -]?key|app[_ -]?secret|authorization|bearer)(\s*[:=]?\s*)[^\s,;，；]+",
        r"\1\2***",
        detail,
    )
    return detail[:240]


class FeishuBot:
    """飞书机器人集成"""

    def __init__(self):
        self._app_access_token: Optional[str] = None
        self._token_expires_at: int = 0
        self._ocr_lock: Optional[asyncio.Lock] = None
        self._ocr_lock_loop = None
        self._last_ocr_request_at = 0.0
        self._local_ocr = local_rapidocr

    def _current_ocr_lock(self) -> asyncio.Lock:
        """Create the throttle lock lazily for the currently running event loop."""
        loop = asyncio.get_running_loop()
        if self._ocr_lock is None or self._ocr_lock_loop is not loop:
            self._ocr_lock = asyncio.Lock()
            self._ocr_lock_loop = loop
        return self._ocr_lock

    def _get_client(self, timeout: float = 10.0) -> httpx.AsyncClient:
        """创建带代理配置的 HTTP 客户端"""
        kwargs: dict = dict(timeout=timeout)
        if settings.HTTP_PROXY_ENABLED and settings.HTTP_PROXY_URL:
            kwargs["proxy"] = settings.HTTP_PROXY_URL
        if not settings.HTTP_VERIFY_SSL:
            kwargs["verify"] = False
        return httpx.AsyncClient(**kwargs)

    # ── 飞书 API 鉴权 ──────────────────────────

    async def _get_app_access_token(self) -> str:
        """获取飞书应用访问令牌（自动缓存）"""
        if self._app_access_token and datetime.now().timestamp() < self._token_expires_at - 60:
            return self._app_access_token

        if not settings.FEISHU_APP_ID or not settings.FEISHU_APP_SECRET:
            return ""

        async with self._get_client() as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": settings.FEISHU_APP_ID,
                    "app_secret": settings.FEISHU_APP_SECRET,
                },
            )
            data = resp.json()
            self._app_access_token = data.get("tenant_access_token", "")
            self._token_expires_at = datetime.now().timestamp() + data.get("expire", 6000)
            return self._app_access_token

    async def get_user_info(self, open_id: str) -> Dict[str, Any]:
        """Resolve a Feishu sender to verified tenant profile fields."""
        token = await self._get_app_access_token()
        if not token or not open_id:
            return {}
        async with self._get_client() as client:
            response = await client.get(
                f"https://open.feishu.cn/open-apis/contact/v3/users/{open_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"user_id_type": "open_id"},
            )
        data = response.json()
        if response.status_code >= 400 or data.get("code", 0) != 0:
            raise RuntimeError(data.get("msg") or f"飞书成员查询失败: HTTP {response.status_code}")
        return dict((data.get("data") or {}).get("user") or {})

    # ── Webhook 签名验证 ───────────────────────

    def verify_webhook_signature(self, timestamp: str, nonce: str, body: Union[str, bytes], signature: str) -> bool:
        """验证飞书 Webhook 回调签名"""
        if not settings.FEISHU_WEBHOOK_SECRET:
            return False

        raw_body = body if isinstance(body, bytes) else body.encode("utf-8")
        string_to_sign = f"{timestamp}{nonce}{settings.FEISHU_WEBHOOK_SECRET}".encode("utf-8") + raw_body
        sign = hashlib.sha256(string_to_sign).hexdigest()
        return hmac.compare_digest(sign, signature)

    # ── 消息处理 ───────────────────────────────

    def parse_feishu_message(self, payload: dict) -> dict:
        """解析飞书回调消息，提取文本内容和上下文"""
        result = {
            "text": "",
            "sender": "",
            "chat_id": "",
            "mention_bot": False,
            "message_id": "",
            "message_type": "text",
            "chat_type": "",
            "root_id": "",
            "parent_id": "",
            "thread_id": "",
            "create_time": "",
            "update_time": "",
        }

        try:
            # 事件类型
            event = payload.get("event", payload)
            result["sender"] = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
            result["chat_id"] = event.get("message", {}).get("chat_id", "")
            message = event.get("message", {})
            result["message_id"] = message.get("message_id", "")
            result["chat_type"] = message.get("chat_type", "")
            result["root_id"] = message.get("root_id", "")
            result["parent_id"] = message.get("parent_id", "")
            result["thread_id"] = message.get("thread_id", "") or message.get("root_id", "")
            result["create_time"] = message.get("create_time", "")
            result["update_time"] = message.get("update_time", "")

            # 消息内容
            msg_type = message.get("message_type", "text")
            result["message_type"] = msg_type
            content_str = message.get("content", "{}")
            content = json.loads(content_str) if isinstance(content_str, str) else content_str

            if msg_type == "text":
                text = content.get("text", "")
                # 检查是否 @ 了机器人
                if re.search(r"@_user_\d+", text):
                    result["mention_bot"] = True
                    # 移除 @ 标记
                    text = re.sub(r"@_user_\d+", "", text).strip()
                result["text"] = text

            elif msg_type == "file":
                result["file_key"] = content.get("file_key", "")
                result["file_name"] = content.get("file_name", "")
                result["text"] = result["file_name"]
            elif msg_type == "post":
                parsed = flatten_rich_text(content if isinstance(content, dict) else {})
                result["text"] = parsed["text"]
                result["image_keys"] = parsed["image_keys"]
                result["mention_bot"] = bool(message.get("mentions")) or "@_user_" in result["text"]
                result["text"] = re.sub(r"@_user_\d+", "", result["text"]).strip()
            elif msg_type == "image":
                result["image_key"] = content.get("image_key", "")
        except Exception as e:
            logger.warning("解析飞书消息失败: %s", e)

        return result

    async def download_message_resource(self, message_id: str, resource_key: str) -> Dict[str, Any]:
        """下载用户消息或富文本中的图片资源。"""
        token = await self._get_app_access_token()
        if not token:
            raise RuntimeError("飞书机器人未配置 App ID 或 App Secret")
        if not message_id or not resource_key:
            raise ValueError("缺少飞书消息 ID 或图片资源 Key")

        async with self._get_client(timeout=30.0) as client:
            resp = await client.get(
                f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{resource_key}",
                params={"type": "image"},
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code != 200:
            detail = resp.text[:300]
            try:
                data = resp.json()
                detail = str(data.get("msg") or data.get("message") or detail)
            except Exception:
                pass
            raise RuntimeError(f"飞书图片下载失败：{detail}")
        return {
            "content": bytes(resp.content),
            "content_type": str(resp.headers.get("content-type") or "application/octet-stream"),
        }

    async def recognize_image_text(self, image_bytes: bytes) -> str:
        """调用飞书 OCR；需要 optical_char_recognition:image 权限。"""
        if len(image_bytes) > 5 * 1024 * 1024:
            raise RuntimeError("图片超过飞书 OCR 的 5 MB 上限")
        token = await self._get_app_access_token()
        if not token:
            raise RuntimeError("飞书机器人未配置 App ID 或 App Secret")
        payload = {"image": base64.b64encode(image_bytes).decode("ascii")}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        async with self._current_ocr_lock():
            async with self._get_client(timeout=30.0) as client:
                for attempt in range(3):
                    interval = time.monotonic() - self._last_ocr_request_at
                    if interval < 1.0:
                        await asyncio.sleep(1.0 - interval)
                    resp = await client.post(
                        "https://open.feishu.cn/open-apis/optical_char_recognition/v1/image/basic_recognize",
                        headers=headers,
                        json=payload,
                    )
                    self._last_ocr_request_at = time.monotonic()
                    try:
                        data = resp.json()
                    except Exception as exc:
                        raise RuntimeError(f"飞书 OCR 返回无法解析：{resp.text[:200]}") from exc
                    if resp.status_code == 200 and data.get("code", 0) == 0:
                        text_list = (data.get("data") or {}).get("text_list") or []
                        return "\n".join(str(item).strip() for item in text_list if str(item).strip())

                    detail = str(data.get("msg") or resp.text[:200])
                    rate_limited = resp.status_code == 429 or "frequency limit" in detail.lower()
                    if not rate_limited:
                        raise RuntimeError(f"飞书 OCR 失败：{detail}")
                    if attempt < 2:
                        await asyncio.sleep(float(attempt + 1))
                        continue
                    raise RuntimeError("飞书 OCR 触发频率限制，已自动重试 3 次，请稍后重新识别")
        raise RuntimeError("飞书 OCR 未返回识别结果")

    async def extract_image_content(self, image_bytes: bytes, content_type: str) -> tuple[str, str]:
        """依次使用本地 OCR、飞书 OCR 和当前多模态模型识别图片。"""
        errors = []
        try:
            text = await self._local_ocr.recognize(image_bytes)
            if text.strip():
                return f"### 识别文字\n\n{text.strip()}", "local_rapidocr"
            errors.append("本地 OCR 未识别到有效文字")
        except Exception as exc:
            errors.append(_safe_error_message(exc))

        try:
            text = await self.recognize_image_text(image_bytes)
            if is_meaningful_ocr_text(text):
                return f"### 识别文字\n\n{text.strip()}", "feishu_ocr"
            errors.append("飞书 OCR 未识别到有效文字")
        except Exception as exc:
            detail = _safe_error_message(exc)
            if "unknown variant `image_url`" in detail or "expected `text`" in detail:
                detail = f"当前模型 {settings.LLM_MODEL or '未命名模型'} 不支持图片输入"
            errors.append(detail)

        try:
            from backend.knowledge_base.llm_service import llm_service

            text = await llm_service.extract_image_knowledge(image_bytes, content_type)
            if text.strip():
                return text.strip(), "vision_model"
            errors.append("多模态模型未返回内容")
        except Exception as exc:
            errors.append(_safe_error_message(exc))
        raise RuntimeError("；".join(errors)[:500])

    async def enrich_messages(
        self,
        messages: list[dict],
        skip_message_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """解析新消息中的富文本与图片，已归档消息不重复下载和识别。"""
        processor = FeishuContentProcessor(
            resource_downloader=self.download_message_resource,
            image_extractor=self.extract_image_content,
        )
        return await processor.process_messages(messages, skip_message_ids=skip_message_ids)

    async def retry_enrich_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """使用已保存的原始内容重新下载并识别一条图片或富文本消息。"""
        original_content = message.get("original_content") or {
            "text": message.get("content") or "",
            "title": message.get("title") or "",
            "image_key": message.get("image_key") or "",
            "image_keys": message.get("image_keys") or [],
        }
        raw_message = {
            "message_id": message.get("message_id") or "",
            "msg_type": message.get("message_type") or "text",
            "body": {"content": json.dumps(original_content, ensure_ascii=False)},
            "sender": {"id": message.get("sender") or "未知成员"},
            "create_time": message.get("create_time") or "",
            "update_time": message.get("update_time") or "",
            "root_id": message.get("root_id") or "",
            "parent_id": message.get("parent_id") or "",
            "thread_id": message.get("thread_id") or "",
            "event_id": message.get("event_id") or "",
        }
        enriched = await self.enrich_messages([raw_message])
        return enriched[0]

    async def send_text_message(self, chat_id: str, text: str) -> bool:
        """向飞书群聊发送文本消息"""
        token = await self._get_app_access_token()
        if not token:
            logger.warning("飞书未配置，跳过消息发送")
            return False

        async with self._get_client() as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                },
            )
            if resp.status_code != 200:
                logger.warning("飞书文本消息发送失败: %s", resp.text)
                return False
            data = resp.json()
            if data.get("code", 0) != 0:
                logger.warning("飞书文本消息发送失败: %s", data.get("msg") or resp.text)
                return False
            return True

    async def reply_text_message(self, message_id: str, text: str) -> bool:
        """回复到指定飞书消息下，保留群聊中的问题上下文。"""
        token = await self._get_app_access_token()
        if not token or not message_id:
            return False

        async with self._get_client() as client:
            resp = await client.post(
                f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "msg_type": "text",
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                },
            )
            if resp.status_code != 200:
                logger.warning("飞书消息回复失败: %s", resp.text)
                return False
            data = resp.json()
            if data.get("code", 0) != 0:
                logger.warning("飞书消息回复失败: %s", data.get("msg") or resp.text)
                return False
            return True

    async def list_chat_messages(
        self,
        chat_id: str,
        page_size: int = 50,
        page_token: str = "",
    ) -> tuple[list[dict], str]:
        """读取机器人已加入群聊的消息历史，返回消息列表与下一页标记。"""
        token = await self._get_app_access_token()
        if not token:
            raise RuntimeError("飞书机器人未配置 App ID 或 App Secret")
        if not chat_id:
            raise ValueError("缺少群聊 Chat ID")

        params = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "page_size": max(1, min(int(page_size), 50)),
            "sort_type": "ByCreateTimeDesc",
        }
        if page_token:
            params["page_token"] = page_token

        async with self._get_client() as client:
            resp = await client.get(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
        data = resp.json()
        if resp.status_code != 200 or data.get("code", 0) != 0:
            logger.warning("飞书群消息同步失败: %s", resp.text)
            reason = data.get("msg") or "请确认机器人已加入群聊且已开通群消息权限"
            raise RuntimeError(f"飞书群消息读取失败：{reason}（code {data.get('code', resp.status_code)}）")
        payload = data.get("data") or {}
        return payload.get("items") or [], str(payload.get("page_token") or "")

    async def list_chats(self, page_size: int = 100, page_token: str = "") -> tuple[list[dict], str]:
        """读取机器人当前所在群聊。"""
        token = await self._get_app_access_token()
        if not token:
            raise RuntimeError("飞书机器人未配置 App ID 或 App Secret")
        params = {
            "page_size": max(1, min(int(page_size), 100)),
            "user_id_type": "open_id",
        }
        if page_token:
            params["page_token"] = page_token
        async with self._get_client() as client:
            resp = await client.get(
                "https://open.feishu.cn/open-apis/im/v1/chats",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
        data = resp.json()
        if resp.status_code != 200 or data.get("code", 0) != 0:
            reason = data.get("msg") or "请确认已开通读取群信息权限"
            raise RuntimeError(f"飞书群列表读取失败：{reason}（code {data.get('code', resp.status_code)}）")
        payload = data.get("data") or {}
        return payload.get("items") or [], str(payload.get("page_token") or "")

    async def get_chat_info(self, chat_id: str) -> dict:
        """按 Chat ID 读取群名称、头像和描述。"""
        token = await self._get_app_access_token()
        if not token:
            raise RuntimeError("飞书机器人未配置 App ID 或 App Secret")
        async with self._get_client() as client:
            resp = await client.get(
                f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}",
                params={"user_id_type": "open_id"},
                headers={"Authorization": f"Bearer {token}"},
            )
        data = resp.json()
        if resp.status_code != 200 or data.get("code", 0) != 0:
            reason = data.get("msg") or "群信息不可用"
            raise RuntimeError(f"飞书群信息读取失败：{reason}（code {data.get('code', resp.status_code)}）")
        return data.get("data") or {}

    async def send_card_message(self, chat_id: str, title: str, content: str,
                                 buttons: list = None) -> bool:
        """发送飞书消息卡片（富文本）"""
        token = await self._get_app_access_token()
        if not token:
            return False

        # 构建消息卡片
        elements = [
            {
                "tag": "markdown",
                "content": content[:2000],  # 飞书单字段限制
            }
        ]

        if buttons:
            elements.append({
                "tag": "action",
                "actions": [
                    {"tag": "button", "text": {"tag": "plain_text", "content": b["text"]},
                     "type": b.get("type", "default"),
                     "url": b.get("url", "")}
                    for b in buttons
                ],
            })

        card = {
            "header": {
                "title": {"tag": "plain_text", "content": title[:100]},
                "template": "blue",
            },
            "elements": elements,
        }

        async with self._get_client() as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "receive_id": chat_id,
                    "msg_type": "interactive",
                    "content": json.dumps(card, ensure_ascii=False),
                },
            )
            ok = resp.status_code == 200
            if not ok:
                logger.warning("飞书卡片发送失败: %s", resp.text)
            return ok

    # ── 知识推送功能 ───────────────────────────

    async def push_daily_knowledge(self, chat_id: str, knowledge_items: list) -> bool:
        """推送每日知识卡片到指定群聊"""
        if not knowledge_items:
            return False

        content_lines = ["**📚 今日项目知识推荐**\n"]
        for i, item in enumerate(knowledge_items[:5], 1):
            content_lines.append(f"{i}. **{item.get('title', '未命名')}**")
            content_lines.append(f"   {item.get('summary', '')[:100]}")
            if item.get("tags"):
                content_lines.append(f"   `{'` `'.join(item['tags'])}`")
            content_lines.append("")

        return await self.send_card_message(
            chat_id=chat_id,
            title="📚 每日知识推送",
            content="\n".join(content_lines),
            buttons=[
                {"text": "🔍 进入知识库", "type": "primary",
                 "url": f"{settings.FRONTEND_URL or 'http://localhost:8000'}"},
            ],
        )

    async def send_resignation_guide(self, chat_id: str, sender_name: str = "") -> bool:
        """发送离职流程引导卡片（含前端表单链接）"""
        frontend_url = settings.FRONTEND_URL or "http://localhost:8000"
        content = (
            f"**📋 离职流程指引**\n\n"
            f"您好，检测到您咨询离职相关事宜，以下为离职流程：\n\n"
            f"**步骤一：填写离职信息**\n"
            f"请点击下方按钮，填写您的姓名和职能信息。\n\n"
            f"**步骤二：上传交接资料**\n"
            f"在表单中上传您的工作交接文档（支持 PDF/TXT/MD/DOCX）。\n\n"
            f"**步骤三：提交审核**\n"
            f"提交后系统会自动将您的交接资料归档至知识库，"
            f"便于后续同事查阅。\n\n"
            f"如有疑问可继续@我咨询。"
        )

        return await self.send_card_message(
            chat_id=chat_id,
            title="📋 离职流程指引",
            content=content,
            buttons=[
                {"text": "📝 填写离职表单", "type": "primary",
                 "url": f"{frontend_url}/?tab=resignation"},
                {"text": "💬 咨询 AI 助手", "type": "default",
                 "url": f"{frontend_url}/?tab=chat"},
            ],
        )

    async def push_project_weekly(self, chat_id: str, weekly_data: dict) -> bool:
        """推送项目知识周报"""
        content_lines = [
            f"**📊 {weekly_data.get('project_name', '项目')} - 知识周报**\n",
            f"📄 本周新增文档: {weekly_data.get('new_docs', 0)} 篇",
            f"🧩 新增知识块: {weekly_data.get('new_chunks', 0)} 块",
            f"💡 新增决策记录: {weekly_data.get('new_decisions', 0)} 条",
            f"👥 本周活跃成员: {weekly_data.get('active_members', 0)} 人",
            "",
            f"**🏆 热门知识**:",
        ]
        for item in weekly_data.get("hot_items", []):
            content_lines.append(f"- {item}")

        return await self.send_card_message(
            chat_id=chat_id,
            title=f"📊 项目知识周报",
            content="\n".join(content_lines),
        )


# ── 飞书 Webhook API 处理函数 ──────────────

async def _resolve_feishu_rag_identity(bot: FeishuBot, sender_open_id: str, group: Dict[str, Any]):
    """Map Feishu identity to an active project member and verify group scope."""
    from backend.auth import ROLE_PERMISSIONS
    from backend.auth_context import Identity
    from backend.storage import get_repository

    repository = get_repository()
    record = repository.resolve_external_identity("feishu", sender_open_id)
    if not record:
        try:
            profile = await bot.get_user_info(sender_open_id)
        except Exception as exc:
            logger.warning("飞书成员身份解析失败: %s", _safe_error_message(exc))
            profile = {}
        record = repository.resolve_external_identity(
            "feishu", sender_open_id, verified_email=str(profile.get("email") or ""),
        )
    if not record:
        return None
    role = str(record.get("role") or "")
    identity = Identity(
        user_id=str(record["user_id"]),
        email=str(record.get("email") or ""),
        display_name=str(record.get("display_name") or ""),
        organization_id=str(record.get("organization_id") or ""),
        project_id=str(record.get("project_id") or ""),
        project_name=str(record.get("project_name") or ""),
        role=role,
        permissions=ROLE_PERMISSIONS.get(role, frozenset()),
        session_id=f"feishu:{sender_open_id}",
    )
    if not identity.can("knowledge.read"):
        return None
    if not repository.identity_can_access_scope(
        identity,
        visibility=str(group.get("visibility") or "project"),
        access_policy_id=str(group.get("access_policy_id") or ""),
        owner_user_id=str(group.get("owner_user_id") or ""),
    ):
        return None
    return identity

async def handle_feishu_webhook(payload: dict) -> dict:
    """处理飞书回调 Webhook（消息事件等）

    当用户在群里 @机器人 时，使用与大模型对话（RAG 增强），
    与前端的 /api/chat 体验一致。
    """
    bot = FeishuBot()

    # 验证事件类型
    event_type = payload.get("header", {}).get("event_type", payload.get("type", ""))

    if "im.message.receive_v1" in event_type or event_type == "event_callback":
        # 解析消息
        msg = bot.parse_feishu_message(payload)

        # 已授权采集源先归档，再依据群策略自动评估、聚合和沉淀。
        from backend.feishu_workspace import feishu_workspace
        group = feishu_workspace.get_group(msg["chat_id"])
        collected_without_reply = False
        enrichment_job = None
        if msg["chat_type"] == "p2p" and not group and msg["chat_id"]:
            sender_suffix = msg["sender"][-6:] if msg["sender"] else "成员"
            group = feishu_workspace.upsert_group({
                "chat_id": msg["chat_id"],
                "name": f"机器人单聊 · {sender_suffix}",
                "chat_type": "p2p",
                "collection_mode": "archive_only",
                "default_category": "proj_communication",
                "retention_days": 30,
                "bot_enabled": True,
            })

        capture_commands = ("保存聊天记录", "保存群聊记录", "同步聊天记录", "开始记录群聊")
        is_capture_command = (
            msg["chat_type"] == "group"
            and any(command in msg["text"] for command in capture_commands)
        )
        if is_capture_command and msg["chat_id"]:
            try:
                chat_info = await bot.get_chat_info(msg["chat_id"])
                group = feishu_workspace.upsert_group({
                    "chat_id": msg["chat_id"],
                    "name": chat_info.get("name") or "飞书项目群",
                    "description": chat_info.get("description") or "",
                    "avatar": chat_info.get("avatar") or "",
                    "chat_type": "group",
                    "collection_mode": "auto",
                    "default_category": "proj_communication",
                    "retention_days": 90,
                    "bot_enabled": True,
                })
                history, next_page_token = await bot.list_chat_messages(msg["chat_id"], page_size=50)
                history = await bot.enrich_messages(
                    history,
                    skip_message_ids=feishu_workspace.existing_message_ids(),
                )
                added = feishu_workspace.upsert_messages(msg["chat_id"], history)
                reply = (
                    f"已接入群聊“{group['name']}”，同步最近 {len(history)} 条记录，"
                    f"新增 {added} 条成员消息。\n"
                    "系统会自动归档并聚合有价值内容，只有异常项需要人工处理。"
                )
                from backend.feishu_knowledge import feishu_knowledge_service
                ready_ids = [
                    item["candidate_id"]
                    for item in feishu_workspace.list_candidates(status="ready_for_auto", chat_id=msg["chat_id"])
                ]
                publish_result = feishu_knowledge_service.publish_ready_candidates(ready_ids)
                if next_page_token:
                    reply += " 更早的记录可在项目工作台继续同步。"
                delivered = await bot.reply_text_message(msg["message_id"], reply)
                if not delivered:
                    delivered = await bot.send_text_message(msg["chat_id"], reply)
                action = "group_connected" if delivered else "reply_failed"
                feishu_workspace.record_event(event_type, action, msg["message_id"])
                return {
                    "success": delivered,
                    "action": action,
                    "message_id": msg["message_id"],
                    "group": group,
                    "received": len(history),
                    "added": added,
                    "published": len(publish_result["published"]),
                }
            except Exception as exc:
                error_text = str(exc)[:240]
                reply = f"群聊接入失败：{error_text}"
                delivered = await bot.reply_text_message(msg["message_id"], reply)
                if not delivered:
                    await bot.send_text_message(msg["chat_id"], reply)
                feishu_workspace.record_event(event_type, "group_connect_failed", msg["message_id"], error_text)
                return {"success": False, "action": "group_connect_failed", "message_id": msg["message_id"]}

        if group and group.get("collection_mode") != "off" and msg["message_id"]:
            event = payload.get("event", payload)
            raw_message = dict(event.get("message", {}))
            raw_message["message_id"] = msg["message_id"]
            raw_message["event_id"] = str(payload.get("header", {}).get("event_id") or "")
            raw_message["msg_type"] = raw_message.get("message_type") or msg["message_type"]
            raw_message["sender"] = {"id": msg["sender"]}
            raw_message["root_id"] = raw_message.get("root_id") or msg.get("root_id") or ""
            raw_message["parent_id"] = raw_message.get("parent_id") or msg.get("parent_id") or ""
            raw_message["thread_id"] = raw_message.get("thread_id") or msg.get("thread_id") or ""
            requires_enrichment = (
                msg["message_type"] == "image"
                or (msg["message_type"] == "post" and bool(msg.get("image_keys")))
            )
            if requires_enrichment:
                existing_message = feishu_workspace.get_message(msg["message_id"])
                already_enriched = bool(
                    existing_message
                    and existing_message.get("extraction_status") in {"completed", "partial"}
                )
                if not already_enriched:
                    original_content = message_content(raw_message)
                    if msg["message_type"] == "post":
                        parsed = flatten_rich_text(original_content)
                        queued_content = {
                            **parsed,
                            "original_content": original_content,
                            "extraction_status": "queued",
                            "extraction_method": "rich_text_parser",
                        }
                    else:
                        queued_content = {
                            "text": "",
                            "image_key": str(original_content.get("image_key") or msg.get("image_key") or ""),
                            "image_keys": [str(original_content.get("image_key") or msg.get("image_key") or "")],
                            "original_content": original_content,
                            "extraction_status": "queued",
                            "extraction_method": "",
                        }
                    queued_content["image_keys"] = [item for item in queued_content.get("image_keys", []) if item]
                    queued_message = replace_message_content(raw_message, queued_content)
                    feishu_workspace.upsert_messages(msg["chat_id"], [queued_message])
                    from backend.processing_jobs import processing_job_service
                    enrichment_job = processing_job_service.enqueue(
                        "feishu_message_enrichment",
                        {"message_id": msg["message_id"], "trigger": "webhook"},
                        created_by="user_system",
                        idempotency_key=f"feishu-enrichment:{msg['chat_id']}:{msg['message_id']}",
                        source_id=msg["chat_id"],
                        priority=80,
                    )
                    feishu_workspace.record_event(
                        event_type, "enrichment_queued", msg["message_id"],
                    )
                collected_without_reply = True
            else:
                enriched = await bot.enrich_messages(
                    [raw_message],
                    skip_message_ids=feishu_workspace.existing_message_ids(),
                )
                raw_message = enriched[0]
                added = feishu_workspace.upsert_messages(msg["chat_id"], [raw_message])
                collected_without_reply = bool(added or feishu_workspace.get_message(msg["message_id"]))
                if added and group.get("collection_mode") == "auto":
                    from backend.feishu_knowledge import feishu_knowledge_service
                    ready_ids = [
                        item["candidate_id"]
                        for item in feishu_workspace.list_candidates(status="ready_for_auto", chat_id=msg["chat_id"])
                    ]
                    feishu_knowledge_service.publish_ready_candidates(ready_ids)

        should_reply = msg["chat_type"] == "p2p" or msg["mention_bot"]
        if group and group.get("bot_enabled") is False:
            should_reply = False

        if not should_reply:
            action = "collected_no_reply" if collected_without_reply else "ignored_not_mentioned"
            detail = "消息已按采集规则归档，未触发机器人回复" if collected_without_reply else "未授权采集且未触发机器人回复"
            feishu_workspace.record_event(event_type, action, msg["message_id"], detail)
            return {
                "success": True, "action": action,
                "job_id": str((enrichment_job or {}).get("job_id") or ""),
            }

        feishu_workspace.record_event(event_type, "processing", msg["message_id"])

        if msg["message_type"] == "image" and not msg["text"].strip():
            stored = feishu_workspace.get_message(msg["message_id"]) or {}
            if stored.get("extraction_status") == "queued":
                reply = "图片已接收，正在后台识别；完成后会按当前采集规则归档或形成知识。"
            elif stored.get("extraction_status") == "failed":
                reply = "图片原件已保存，但内容识别未完成。请在飞书内容中心检查资源与 OCR 权限后重新识别。"
            else:
                reply = "图片及识别内容已保存，系统会按当前采集规则归档或形成知识。"
            delivered = await bot.reply_text_message(msg["message_id"], reply)
            if not delivered:
                delivered = await bot.send_text_message(msg["chat_id"], reply)
            action = "image_enrichment_queued" if delivered and enrichment_job else "image_collected" if delivered else "reply_failed"
            feishu_workspace.record_event(event_type, action, msg["message_id"])
            return {
                "success": delivered, "action": action, "message_id": msg["message_id"],
                "job_id": str((enrichment_job or {}).get("job_id") or ""),
            }

        # ── 离职关键词检测（保持原有功能）──
        resignation_keywords = [
            "离职", "辞职", "离开公司", "离职流程", "辞职流程",
            "离岗", "离职手续", "离职申请", "我要离职", "办理离职",
            "resign", "quit", "leave",
        ]
        text_lower = msg["text"].lower()
        is_resignation = any(kw in text_lower for kw in resignation_keywords)

        if is_resignation:
            await bot.send_resignation_guide(msg["chat_id"])
            feishu_workspace.record_event(event_type, "resignation_guide", msg["message_id"])
            return {"success": True, "action": "resignation_guide", "message_id": msg["message_id"]}

        # ── 调用 LLM 对话（RAG 增强），与前端的 /api/chat 一致 ──
        from backend.knowledge_base.llm_service import llm_service
        from backend.auth_context import reset_current_identity, set_current_identity

        rag_identity = await _resolve_feishu_rag_identity(bot, msg["sender"], group or {})
        if rag_identity is None:
            reply = "当前飞书账号尚未绑定为本项目有效成员，或无权访问该群知识。为保护项目数据，本次未执行知识库检索。"
            delivered = await bot.reply_text_message(msg["message_id"], reply)
            if not delivered:
                delivered = await bot.send_text_message(msg["chat_id"], reply)
            feishu_workspace.record_event(
                event_type, "rag_access_denied", msg["message_id"], "发送者或群聊权限校验未通过",
            )
            return {"success": delivered, "action": "rag_access_denied", "message_id": msg["message_id"]}

        token = set_current_identity(rag_identity)
        try:
            # 收集流式回复为完整文本（飞书消息需一次性发送）
            full_reply = ""
            async for chunk in llm_service.chat(query=msg["text"], use_rag=True):
                if chunk:
                    full_reply += chunk

            if full_reply:
                # 飞书文本消息有 ~2000 字符限制，超出则截断并加提示
                MAX_LEN = 1800
                if len(full_reply) > MAX_LEN:
                    full_reply = full_reply[:MAX_LEN] + "\n\n……（回复过长已截断，请在前端查看完整内容）"
                reply = f"🤖 **AI 助手**\n\n{full_reply}"
            else:
                reply = f"🤖 抱歉，我没有生成有效的回答，请稍后换个方式提问。"
        except Exception as e:
            logger.error("飞书 LLM 对话失败: %s", e)
            reply = (
                f"🤖 **AI 助手**\n\n"
                f"抱歉，AI 服务暂时不可用，请稍后重试。\n"
                f"（错误: {_safe_error_message(e)[:150]}）"
            )
        finally:
            reset_current_identity(token)

        delivered = await bot.reply_text_message(msg["message_id"], reply)
        if not delivered:
            delivered = await bot.send_text_message(msg["chat_id"], reply)
        action = "replied" if delivered else "reply_failed"
        feishu_workspace.record_event(
            event_type,
            action,
            msg["message_id"],
            "飞书发送接口未成功" if not delivered else "",
        )
        return {"success": delivered, "action": action, "message_id": msg["message_id"]}

    # URL 验证（飞书首次配置时需要）
    challenge = payload.get("challenge")
    if challenge:
        from backend.feishu_workspace import feishu_workspace
        feishu_workspace.record_event("url_verification", "challenge_verified")
        return {"challenge": challenge}

    return {"success": True, "action": "unhandled"}


# 全局单例
feishu_bot = FeishuBot()
