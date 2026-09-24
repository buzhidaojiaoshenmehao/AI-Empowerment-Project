"""飞书富文本与图片消息的规范化、资源留存和内容提取。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


ResourceDownloader = Callable[[str, str], Awaitable[Dict[str, Any]]]
ImageExtractor = Callable[[bytes, str], Awaitable[Tuple[str, str]]]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def decode_message_content(value: Any) -> Dict[str, Any]:
    """兼容事件回调、历史消息以及已规范化的内容字段。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"text": value}
        except json.JSONDecodeError:
            return {"text": value}
    return {}


def _localized_post(content: Dict[str, Any]) -> Dict[str, Any]:
    if "content" in content or "title" in content:
        return content
    for locale in ("zh_cn", "zh_hk", "en_us", "ja_jp"):
        localized = content.get(locale)
        if isinstance(localized, dict):
            return localized
    return content


def flatten_rich_text(content: Dict[str, Any]) -> Dict[str, Any]:
    """将飞书 post 消息转成适合检索的 Markdown，并提取内嵌资源。"""
    post = _localized_post(_as_dict(content))
    title = str(post.get("title") or "").strip()
    paragraphs: List[str] = []
    image_keys: List[str] = []
    links: List[Dict[str, str]] = []

    for row in _as_list(post.get("content")):
        fragments: List[str] = []
        for raw_element in _as_list(row):
            element = _as_dict(raw_element)
            tag = str(element.get("tag") or "text")
            text = str(element.get("text") or element.get("content") or "").strip()
            if tag in {"text", "code_block"}:
                if text:
                    fragments.append(text)
            elif tag == "a":
                href = str(element.get("href") or "").strip()
                label = text or href
                if href:
                    links.append({"text": label, "href": href})
                    fragments.append(f"[{label}]({href})" if label else href)
            elif tag == "at":
                name = str(element.get("user_name") or element.get("user_id") or "成员").strip()
                fragments.append(f"@{name}")
            elif tag == "img":
                image_key = str(element.get("image_key") or "").strip()
                if image_key:
                    image_keys.append(image_key)
                    fragments.append("[图片]")
            elif tag == "media":
                image_key = str(element.get("image_key") or "").strip()
                if image_key:
                    image_keys.append(image_key)
                fragments.append("[媒体附件]")
            elif tag == "emotion":
                emotion = str(element.get("emoji_type") or element.get("emotion_type") or "表情").strip()
                fragments.append(f"[{emotion}]")
            elif tag == "hr":
                fragments.append("---")
            elif text:
                fragments.append(text)
        paragraph = " ".join(part for part in fragments if part).strip()
        if paragraph:
            paragraphs.append(paragraph)

    parts = [f"# {title}" if title else "", *paragraphs]
    text = "\n\n".join(part for part in parts if part).strip()
    return {
        "title": title,
        "text": text,
        "image_keys": list(dict.fromkeys(image_keys)),
        "links": links,
    }


def message_content(message: Dict[str, Any]) -> Dict[str, Any]:
    body = _as_dict(message.get("body"))
    return decode_message_content(body.get("content", message.get("content", {})))


def replace_message_content(message: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(message)
    encoded = json.dumps(content, ensure_ascii=False)
    if isinstance(result.get("body"), dict):
        result["body"] = {**result["body"], "content": encoded}
    else:
        result["content"] = encoded
    return result


def _image_format(data: bytes, hinted_content_type: str = "") -> Tuple[str, str]:
    hinted = str(hinted_content_type or "").split(";", 1)[0].strip().lower()
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
        (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
        (b"GIF87a", "image/gif", ".gif"),
        (b"GIF89a", "image/gif", ".gif"),
        (b"BM", "image/bmp", ".bmp"),
    )
    for signature, content_type, suffix in signatures:
        if data.startswith(signature):
            return content_type, suffix
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    suffixes = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
    }
    if hinted in suffixes:
        return hinted, suffixes[hinted]
    raise ValueError("下载结果不是受支持的图片格式")


class FeishuContentProcessor:
    """把飞书消息扩展成可归档、可预览、可检索的标准内容。"""

    def __init__(
        self,
        resource_downloader: ResourceDownloader,
        image_extractor: ImageExtractor,
        resource_dir: Optional[Path] = None,
        max_image_bytes: int = 10 * 1024 * 1024,
        concurrency: int = 3,
    ) -> None:
        self.resource_downloader = resource_downloader
        self.image_extractor = image_extractor
        self.resource_dir = Path(resource_dir or "uploads/feishu_resources")
        self.max_image_bytes = max_image_bytes
        self.concurrency = max(1, min(int(concurrency), 5))

    def _store_image(
        self,
        message_id: str,
        resource_key: str,
        data: bytes,
        hinted_content_type: str,
    ) -> Dict[str, Any]:
        if not data:
            raise ValueError("图片资源为空")
        if len(data) > self.max_image_bytes:
            raise ValueError(f"图片超过 {self.max_image_bytes // 1024 // 1024} MB 处理上限")
        content_type, suffix = _image_format(data, hinted_content_type)
        digest = hashlib.sha256(f"{message_id}:{resource_key}".encode("utf-8")).hexdigest()[:24]
        stored_file = f"{digest}{suffix}"
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        target = self.resource_dir / stored_file
        if not target.exists() or target.stat().st_size != len(data):
            temporary = target.with_suffix(f"{target.suffix}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        return {
            "resource_key": resource_key,
            "stored_file": stored_file,
            "content_type": content_type,
            "size": len(data),
            "status": "stored",
        }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        message_type = str(message.get("msg_type") or message.get("message_type") or "text").lower()
        if message_type not in {"image", "post"}:
            return message

        original_content = message_content(message)
        base_text = str(original_content.get("text") or "").strip()
        image_keys = [str(item).strip() for item in _as_list(original_content.get("image_keys")) if str(item).strip()]
        title = str(original_content.get("title") or "").strip()
        links = _as_list(original_content.get("links"))
        extraction_methods: List[str] = []

        if message_type == "post":
            rich_source = _as_dict(original_content.get("original_content")) or original_content
            parsed = flatten_rich_text(rich_source)
            base_text = parsed["text"] or base_text
            title = parsed["title"] or title
            links = parsed["links"] or links
            image_keys.extend(parsed["image_keys"])
            extraction_methods.append("rich_text_parser")
        else:
            image_key = str(original_content.get("image_key") or "").strip()
            if image_key:
                image_keys.append(image_key)

        image_keys = list(dict.fromkeys(image_keys))
        message_id = str(message.get("message_id") or message.get("id") or "").strip()
        resource_files: List[Dict[str, Any]] = []
        extracted_sections: List[str] = []
        errors: List[str] = []

        for index, resource_key in enumerate(image_keys, 1):
            stored: Dict[str, Any] = {}
            try:
                payload = await self.resource_downloader(message_id, resource_key)
                data = payload.get("content") or b""
                stored = self._store_image(
                    message_id,
                    resource_key,
                    data,
                    str(payload.get("content_type") or ""),
                )
                extracted_text, method = await self.image_extractor(data, stored["content_type"])
                extracted_text = str(extracted_text or "").strip()
                if not extracted_text:
                    raise RuntimeError("图片识别未返回有效内容")
                stored.update({"status": "extracted", "extraction_method": method})
                resource_files.append(stored)
                extraction_methods.append(method)
                heading = "图片内容" if len(image_keys) == 1 else f"图片 {index} 内容"
                extracted_sections.append(f"## {heading}\n\n{extracted_text}")
            except Exception as exc:
                errors.append(f"{resource_key}: {str(exc)[:240]}")
                resource_files.append({
                    "resource_key": resource_key,
                    "stored_file": str(stored.get("stored_file") or ""),
                    "content_type": str(stored.get("content_type") or ""),
                    "size": int(stored.get("size") or 0),
                    "status": "failed",
                    "error": str(exc)[:240],
                })

        parts = [base_text, *extracted_sections]
        text = "\n\n".join(part for part in parts if part).strip()
        if image_keys and errors and extracted_sections:
            extraction_status = "partial"
        elif image_keys and errors and not extracted_sections:
            extraction_status = "failed"
        else:
            extraction_status = "completed"

        enriched_content = {
            "text": text,
            "title": title,
            "image_key": image_keys[0] if message_type == "image" and image_keys else "",
            "image_keys": image_keys,
            "links": links,
            "resource_files": resource_files,
            "resource_count": sum(1 for item in resource_files if item.get("stored_file")),
            "extraction_status": extraction_status,
            "extraction_method": ",".join(dict.fromkeys(extraction_methods)),
            "extraction_error": "；".join(errors)[:500],
            "original_content": _as_dict(original_content.get("original_content")) or original_content,
        }
        return replace_message_content(message, enriched_content)

    async def process_messages(
        self,
        messages: Iterable[Dict[str, Any]],
        skip_message_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        skipped = set(skip_message_ids or [])
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process(message: Dict[str, Any]) -> Dict[str, Any]:
            message_id = str(message.get("message_id") or message.get("id") or "")
            if message_id in skipped:
                return message
            async with semaphore:
                return await self.process_message(message)

        return list(await asyncio.gather(*(process(dict(message)) for message in messages)))
