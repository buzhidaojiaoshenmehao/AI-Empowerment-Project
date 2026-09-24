"""飞书知识自动化的本地状态、规则评估与可追溯归档。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.feishu_content import flatten_rich_text
from backend.storage.repository import StorageRepository


WORKSPACE_VERSION = 4
COLLECTION_MODES = {"off", "review", "auto", "archive_only"}
REVIEW_STATUSES = {"pending", "approved", "ignored"}
CANDIDATE_STATUSES = {
    "ready_for_auto", "review_required", "published", "failed", "excluded", "reverted",
}

LOW_VALUE_PHRASES = {
    "ok", "okay", "收到", "好的", "好", "可以", "明白", "了解", "谢谢", "感谢",
    "辛苦了", "辛苦", "赞", "+1", "哈哈", "嗯", "行", "没问题", "已阅",
}
KNOWLEDGE_KEYWORDS = {
    "决策", "结论", "风险", "问题", "原因", "方案", "流程", "规范", "要求", "需求",
    "架构", "接口", "部署", "发布", "验收", "复盘", "经验", "教训", "交接", "负责人",
    "截止", "里程碑", "变更", "影响", "待办", "操作步骤", "注意事项", "解决办法", "故障",
    "decision", "risk", "requirement", "architecture", "deploy", "release", "handover",
}
EXPLICIT_CAPTURE_KEYWORDS = {"沉淀这段", "保存聊天记录", "保存群聊记录", "同步聊天记录", "开始记录群聊"}
SENSITIVE_PATTERNS = (
    re.compile(r"(?:密码|口令|密钥|secret|token|api\s*key)\s*[:：=]", re.IGNORECASE),
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"\b(?:\d[ -]*?){16,19}\b"),
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _message_time(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.isdigit():
            timestamp = int(raw)
            return datetime.fromtimestamp(timestamp / 1000 if timestamp > 9_999_999_999 else timestamp)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            # 前端通过 toISOString() 传入的是 UTC 时间（带 Z/时区），
            # 而飞书毫秒时间戳经 fromtimestamp 得到的是本地时间。
            # 这里先换算到本地时区再去除 tzinfo，保证两者比较基准一致，
            # 否则会因时区偏移把最近数小时（含今天）的消息误判为超出结束时间而被过滤掉。
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except (OverflowError, OSError, ValueError):
        return None


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _candidate_id(aggregation_key: str) -> str:
    digest = hashlib.sha256(aggregation_key.encode("utf-8")).hexdigest()[:18]
    return f"fc_{digest}"


def _asset_id(candidate_id: str) -> str:
    return f"fa_{candidate_id.removeprefix('fc_')}"


def _legacy_candidate_id(message_id: str) -> str:
    return _candidate_id(f"legacy:{message_id}")


def _content_hash(chat_id: str, message_id: str, content: str) -> str:
    return hashlib.sha256(f"{chat_id}\n{message_id}\n{content}".encode("utf-8")).hexdigest()


def _compat_review_status(content_status: str) -> str:
    if content_status == "published":
        return "approved"
    if content_status == "review_required":
        return "pending"
    return "ignored"


def evaluate_content(message: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    """使用确定性规则评估内容，不在事件接收路径调用 LLM。"""
    mode = str(group.get("collection_mode") or "auto")
    text = str(message.get("content") or "").strip()
    message_type = str(message.get("message_type") or "text").lower()
    extraction_status = str(message.get("extraction_status") or "").lower()
    threshold = _clamp_float(group.get("confidence_threshold"), 0.85, 0.5, 0.99)

    if mode == "off":
        return {"content_status": "not_collected", "confidence": 1.0, "reason": "采集已关闭", "valuable": False}
    if mode == "archive_only":
        return {"content_status": "archived", "confidence": 1.0, "reason": "采集源设置为仅归档", "valuable": False}

    lowered = text.lower().strip("。.!！?？ ")
    if message_type in {"system", "reaction", "audio", "sticker"}:
        return {"content_status": "excluded", "confidence": 0.98, "reason": "消息类型不参与知识沉淀", "valuable": False}
    if message_type in {"image", "post"} and extraction_status == "queued":
        return {"content_status": "review_required", "confidence": 0.5, "reason": "图片内容已归档，正在后台识别", "valuable": True}
    if message_type in {"image", "post"} and extraction_status == "failed":
        return {"content_status": "review_required", "confidence": 0.35, "reason": "图片内容识别失败，需要重新识别或人工处理", "valuable": True}
    if message_type in {"image", "post"} and extraction_status == "partial":
        return {"content_status": "review_required", "confidence": 0.68, "reason": "正文已提取，但部分图片识别失败", "valuable": True}
    if not text or text.startswith("[暂不支持直接预览"):
        return {"content_status": "review_required", "confidence": 0.35, "reason": "内容无法解析，需要人工处理", "valuable": True}
    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        return {"content_status": "review_required", "confidence": 0.45, "reason": "可能包含敏感信息", "valuable": True}
    if message_type in {"file", "media", "share_chat", "share_calendar_event"}:
        return {"content_status": "review_required", "confidence": 0.72, "reason": "共享内容需要验证读取和解析结果", "valuable": True}
    if lowered in LOW_VALUE_PHRASES or (len(lowered) <= 8 and not any(key in lowered for key in KNOWLEDGE_KEYWORDS)):
        return {"content_status": "excluded", "confidence": 0.96, "reason": "简短确认或日常寒暄", "valuable": False}

    explicit_capture = any(keyword in text for keyword in EXPLICIT_CAPTURE_KEYWORDS)
    keyword_hits = [keyword for keyword in KNOWLEDGE_KEYWORDS if keyword in lowered]
    has_structure = bool(re.search(r"(?:^|\n)\s*(?:\d+[.、]|[-*]|结论[:：]|原因[:：]|方案[:：])", text))
    length_score = min(0.24, max(0.0, (len(text) - 16) / 240))
    confidence = min(0.98, 0.62 + min(0.2, len(keyword_hits) * 0.06) + length_score + (0.1 if has_structure else 0.0))
    if explicit_capture:
        confidence = max(confidence, 0.95)
    elif len(keyword_hits) >= 2 or (keyword_hits and has_structure):
        confidence = max(confidence, 0.88)
    valuable = explicit_capture or bool(keyword_hits) or (len(text) >= 80 and has_structure)

    if not valuable:
        return {"content_status": "archived", "confidence": 0.64, "reason": "未命中知识价值规则，仅保留原始记录", "valuable": False}
    reason = "用户明确要求沉淀" if explicit_capture else f"命中知识主题：{'、'.join(keyword_hits[:3])}"
    if mode == "review" or confidence < threshold:
        return {"content_status": "review_required", "confidence": round(confidence, 2), "reason": reason, "valuable": True}
    return {"content_status": "candidate", "confidence": round(confidence, 2), "reason": reason, "valuable": True}


class FeishuWorkspace:
    """保存飞书采集规则、原始内容、知识候选、资产与审计记录。"""

    def __init__(self, repository: Optional[StorageRepository] = None) -> None:
        self._storage_repository = repository
        self._lock = RLock()

    @property
    def repository(self) -> StorageRepository:
        if self._storage_repository is None:
            from backend.storage import get_repository

            self._storage_repository = get_repository()
        return self._storage_repository

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": WORKSPACE_VERSION,
            "groups": [],
            "messages": [],
            "candidates": [],
            "assets": [],
            "audit": [],
            "diagnostics": {
                "last_event_at": "", "last_event_type": "", "last_action": "",
                "last_message_id": "", "last_error": "",
            },
            "updated_at": "",
        }

    def _migrate(self, source: Dict[str, Any]) -> Dict[str, Any]:
        empty = self._empty()
        groups = []
        for raw_group in _as_list(source.get("groups")):
            group = _as_dict(raw_group)
            external = bool(group.get("external", False))
            mode = str(group.get("collection_mode") or ("archive_only" if external else "auto"))
            if mode not in COLLECTION_MODES:
                mode = "review"
            groups.append({
                **group,
                "collection_mode": mode,
                "external": external,
                "confidence_threshold": _clamp_float(group.get("confidence_threshold"), 0.85, 0.5, 0.99),
                "aggregation_window_minutes": max(5, min(120, int(group.get("aggregation_window_minutes") or 30))),
                "retention_days": max(7, min(365, int(group.get("retention_days") or 90))),
                "bot_enabled": bool(group.get("bot_enabled", True)),
            })

        messages = []
        for raw_message in _as_list(source.get("messages")):
            message = _as_dict(raw_message)
            message_type = str(message.get("message_type") or "text")
            extraction_status = str(message.get("extraction_status") or "")
            if message_type in {"image", "post"} and not extraction_status:
                extraction_status = "legacy_pending"
            review_status = str(message.get("review_status") or "pending")
            content_status = str(message.get("content_status") or "")
            if not content_status:
                content_status = {"pending": "review_required", "approved": "published", "ignored": "excluded"}.get(review_status, "archived")
            messages.append({
                **message,
                "content_status": content_status,
                "confidence": float(message.get("confidence") or (0.5 if content_status == "review_required" else 1.0)),
                "reason": str(message.get("reason") or "由旧版工作台迁移"),
                "source_url": str(message.get("source_url") or ""),
                "root_id": str(message.get("root_id") or ""),
                "parent_id": str(message.get("parent_id") or ""),
                "thread_id": str(message.get("thread_id") or message.get("root_id") or ""),
                "title": str(message.get("title") or ""),
                "image_key": str(message.get("image_key") or ""),
                "image_keys": [str(item) for item in _as_list(message.get("image_keys")) if str(item)],
                "links": [dict(item) for item in _as_list(message.get("links")) if isinstance(item, dict)],
                "resource_files": [dict(item) for item in _as_list(message.get("resource_files")) if isinstance(item, dict)],
                "resource_count": int(message.get("resource_count") or 0),
                "extraction_status": extraction_status,
                "extraction_method": str(message.get("extraction_method") or ""),
                "extraction_error": str(message.get("extraction_error") or ""),
                "original_content": _as_dict(message.get("original_content")),
                "content_hash": str(message.get("content_hash") or _content_hash(
                    str(message.get("chat_id") or ""), str(message.get("message_id") or ""), str(message.get("content") or ""),
                )),
            })

        candidates = [dict(item) for item in _as_list(source.get("candidates")) if isinstance(item, dict)]
        assets = [dict(item) for item in _as_list(source.get("assets")) if isinstance(item, dict)]
        known_candidate_ids = {str(item.get("candidate_id") or "") for item in candidates}
        known_asset_candidates = {str(item.get("candidate_id") or "") for item in assets}
        for message in messages:
            message_id = str(message.get("message_id") or "")
            content_status = str(message.get("content_status") or "")
            if not message_id or content_status not in {"review_required", "published"}:
                continue
            candidate_id = str(message.get("candidate_id") or _legacy_candidate_id(message_id))
            message["candidate_id"] = candidate_id
            if candidate_id not in known_candidate_ids:
                candidates.append({
                    "candidate_id": candidate_id, "aggregation_key": f"legacy:{message_id}",
                    "chat_id": str(message.get("chat_id") or ""), "title": self._candidate_title([message]),
                    "source_message_ids": [message_id], "message_count": 1, "category": "proj_communication",
                    "confidence": float(message.get("confidence") or 0.5),
                    "reasons": [str(message.get("reason") or "由旧版工作台迁移")],
                    "status": "published" if content_status == "published" else "review_required",
                    "asset_id": "", "created_at": str(message.get("synced_at") or _now()), "updated_at": _now(),
                })
                known_candidate_ids.add(candidate_id)
            if content_status == "published" and candidate_id not in known_asset_candidates:
                asset_id = _asset_id(candidate_id)
                stored_file = f"feishu_{message_id.replace('/', '_')}.md"
                asset = {
                    "asset_id": asset_id, "candidate_id": candidate_id,
                    "source_file": str(message.get("knowledge_file") or stored_file), "stored_file": stored_file,
                    "source_message_ids": [message_id], "status": "published",
                    "published_at": str(message.get("reviewed_at") or _now()), "reverted_at": "",
                }
                assets.append(asset)
                message["asset_id"] = asset_id
                for candidate in candidates:
                    if candidate.get("candidate_id") == candidate_id:
                        candidate["asset_id"] = asset_id
                known_asset_candidates.add(candidate_id)

        return {
            **empty,
            "groups": groups, "messages": messages, "candidates": candidates, "assets": assets,
            "audit": [dict(item) for item in _as_list(source.get("audit")) if isinstance(item, dict)][-2000:],
            "diagnostics": {**empty["diagnostics"], **_as_dict(source.get("diagnostics"))},
            "updated_at": str(source.get("updated_at") or ""),
            "_revision": int(source.get("_revision") or 0),
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            data = self.repository.load_feishu_workspace()
            return self._migrate(data) if isinstance(data, dict) else self._empty()

    def save(self, data: Dict[str, Any]) -> None:
        with self._lock:
            data["version"] = WORKSPACE_VERSION
            data["updated_at"] = _now()
            data["_revision"] = self.repository.save_feishu_workspace(data)

    def revision(self) -> int:
        return self.repository.feishu_revision()

    def list_groups(self) -> List[Dict[str, Any]]:
        return sorted(self.load()["groups"], key=lambda group: str(group.get("updated_at") or ""), reverse=True)

    def upsert_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chat_id = str(payload.get("chat_id") or "").strip()
        if not chat_id:
            raise ValueError("请填写群聊 Chat ID")
        data = self.load()
        groups = data["groups"]
        current = next((group for group in groups if group.get("chat_id") == chat_id), {})
        external = bool(payload.get("external", current.get("external", False)))
        default_mode = "archive_only" if external else "auto"
        mode = str(payload.get("collection_mode") or current.get("collection_mode") or default_mode)
        if mode not in COLLECTION_MODES:
            raise ValueError("无效的采集模式")
        visibility = str(payload.get("visibility") or current.get("visibility") or "project").strip()
        if visibility not in {"project", "organization", "restricted", "private"}:
            raise ValueError("无效的知识可见范围")
        access_policy_id = str(payload.get("access_policy_id") or current.get("access_policy_id") or "").strip()
        if visibility == "restricted" and not access_policy_id:
            raise ValueError("受限采集源必须绑定访问策略")
        group = {
            **current,
            "chat_id": chat_id, "name": str(payload.get("name") or current.get("name") or "未命名项目群").strip(),
            "collection_mode": mode,
            "default_category": str(payload.get("default_category") or current.get("default_category") or "proj_communication").strip(),
            "retention_days": max(7, min(365, int(payload.get("retention_days") or current.get("retention_days") or 90))),
            "bot_enabled": bool(payload.get("bot_enabled", current.get("bot_enabled", True))),
            "chat_type": str(payload.get("chat_type") or current.get("chat_type") or "group"), "external": external,
            "description": str(payload.get("description") or current.get("description") or "").strip(),
            "avatar": str(payload.get("avatar") or current.get("avatar") or "").strip(),
            "confidence_threshold": _clamp_float(payload.get("confidence_threshold", current.get("confidence_threshold")), 0.85, 0.5, 0.99),
            "aggregation_window_minutes": max(5, min(120, int(payload.get("aggregation_window_minutes") or current.get("aggregation_window_minutes") or 30))),
            "visibility": visibility,
            "access_policy_id": access_policy_id if visibility == "restricted" else "",
            "last_synced_at": current.get("last_synced_at", ""), "last_sync_count": int(current.get("last_sync_count") or 0),
            "created_at": current.get("created_at") or _now(), "updated_at": _now(),
        }
        data["groups"] = [item for item in groups if item.get("chat_id") != chat_id] + [group]
        self._append_audit(data, "source_policy_saved", "group", chat_id, f"采集模式：{mode}")
        self.save(data)
        return group

    def get_group(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return next((group for group in self.list_groups() if group.get("chat_id") == chat_id), None)

    def normalize_message(self, chat_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        body = _as_dict(message.get("body"))
        raw_content = body.get("content", message.get("content", "{}"))
        if isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                content = {"text": raw_content}
        else:
            content = _as_dict(raw_content)
        message_type = str(message.get("msg_type") or message.get("message_type") or "text")
        if message_type == "post" and not content.get("extraction_status"):
            parsed = flatten_rich_text(content)
            content = {**content, **parsed, "original_content": content, "extraction_status": "completed", "extraction_method": "rich_text_parser"}
        text = str(content.get("text") or content.get("title") or content.get("file_name") or "").strip()
        if not text and message_type == "file":
            text = str(content.get("file_name") or "[文件消息]")
        if not text:
            text = "[暂不支持直接预览的消息]"
        sender = _as_dict(message.get("sender"))
        sender_id = str(sender.get("id") or sender.get("open_id") or sender.get("sender_id") or "未知成员")
        message_id = str(message.get("message_id") or message.get("id") or "")
        root_id = str(message.get("root_id") or "")
        parent_id = str(message.get("parent_id") or "")
        thread_id = str(message.get("thread_id") or root_id or "")
        return {
            "message_id": message_id, "event_id": str(message.get("event_id") or ""),
            "provider_account_id": str(message.get("provider_account_id") or "feishu_default"), "chat_id": chat_id,
            "sender": sender_id, "message_type": message_type, "content": text[:5000],
            "file_key": str(content.get("file_key") or ""), "file_name": str(content.get("file_name") or ""),
            "title": str(content.get("title") or "")[:500],
            "image_key": str(content.get("image_key") or ""),
            "image_keys": [str(item) for item in _as_list(content.get("image_keys")) if str(item)],
            "links": [dict(item) for item in _as_list(content.get("links")) if isinstance(item, dict)],
            "resource_files": [dict(item) for item in _as_list(content.get("resource_files")) if isinstance(item, dict)],
            "resource_count": int(content.get("resource_count") or 0),
            "extraction_status": str(content.get("extraction_status") or ""),
            "extraction_method": str(content.get("extraction_method") or ""),
            "extraction_error": str(content.get("extraction_error") or "")[:500],
            "original_content": _as_dict(content.get("original_content")),
            "create_time": str(message.get("create_time") or _now()), "update_time": str(message.get("update_time") or ""),
            "root_id": root_id, "parent_id": parent_id, "thread_id": thread_id,
            "source_url": str(message.get("source_url") or ""), "content_hash": _content_hash(chat_id, message_id, text),
            "review_status": "pending", "knowledge_file": "", "candidate_id": "", "asset_id": "", "synced_at": _now(),
        }

    def _aggregation_key(self, group: Dict[str, Any], message: Dict[str, Any]) -> str:
        chat_id = str(message.get("chat_id") or group.get("chat_id") or "")
        thread_id = str(message.get("thread_id") or message.get("root_id") or "")
        if thread_id:
            return f"thread:{chat_id}:{thread_id}"
        window_minutes = max(5, min(120, int(group.get("aggregation_window_minutes") or 30)))
        created_at = _message_time(message.get("create_time")) or datetime.now()
        bucket = int(created_at.timestamp()) // (window_minutes * 60)
        return f"window:{chat_id}:{window_minutes}:{bucket}"

    @staticmethod
    def _candidate_title(messages: Sequence[Dict[str, Any]]) -> str:
        for message in messages:
            text = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
            if text and not text.startswith("["):
                return text if len(text) <= 48 else f"{text[:47]}…"
        return "飞书项目沟通知识"

    def _upsert_candidate(self, data: Dict[str, Any], group: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        aggregation_key = self._aggregation_key(group, message)
        candidate_id = _candidate_id(aggregation_key)
        linked_candidate_id = str(message.get("candidate_id") or "")
        candidate = next((
            item for item in data["candidates"]
            if linked_candidate_id
            and item.get("candidate_id") == linked_candidate_id
            and message.get("message_id") in item.get("source_message_ids", [])
        ), None)
        if candidate is not None:
            candidate_id = str(candidate.get("candidate_id") or candidate_id)
            aggregation_key = str(candidate.get("aggregation_key") or aggregation_key)
        else:
            candidate = next((item for item in data["candidates"] if item.get("candidate_id") == candidate_id), None)
        if candidate is None:
            candidate = {
                "candidate_id": candidate_id, "aggregation_key": aggregation_key,
                "chat_id": str(message.get("chat_id") or ""), "title": "", "source_message_ids": [],
                "message_count": 0, "category": str(group.get("default_category") or "proj_communication"),
                "confidence": 0.0, "reasons": [], "status": "review_required", "asset_id": "",
                "created_at": _now(), "updated_at": _now(),
            }
            data["candidates"].append(candidate)
        source_ids = list(dict.fromkeys([*candidate.get("source_message_ids", []), message["message_id"]]))
        candidate_messages = [item for item in data["messages"] if item.get("message_id") in source_ids]
        if not any(item.get("message_id") == message["message_id"] for item in candidate_messages):
            candidate_messages.append(message)
        candidate_messages.sort(key=lambda item: str(item.get("create_time") or ""))
        reasons = list(dict.fromkeys(str(item.get("reason") or "符合知识规则") for item in candidate_messages))
        confidences = [float(item.get("confidence") or 0.0) for item in candidate_messages]
        requires_review = any(item.get("content_status") == "review_required" for item in candidate_messages)
        previous_status = str(candidate.get("status") or "")
        all_published = bool(candidate_messages) and all(item.get("content_status") == "published" for item in candidate_messages)
        if all_published:
            status = "published"
        else:
            status = "review_required" if requires_review or str(group.get("collection_mode")) != "auto" else "ready_for_auto"
        candidate.update({
            "title": self._candidate_title(candidate_messages), "source_message_ids": source_ids,
            "message_count": len(source_ids), "confidence": round(sum(confidences) / max(1, len(confidences)), 2),
            "reasons": reasons, "status": status, "updated_at": _now(),
            "update_existing_asset": previous_status == "published" or bool(candidate.get("asset_id")),
        })
        message["candidate_id"] = candidate_id
        self._append_audit(data, "candidate_created" if len(source_ids) == 1 else "candidate_updated", "candidate", candidate_id, reasons[-1])
        return candidate

    def upsert_messages(self, chat_id: str, messages: Iterable[Dict[str, Any]]) -> int:
        data = self.load()
        group = next((item for item in data["groups"] if item.get("chat_id") == chat_id), None)
        if not group or group.get("collection_mode") == "off":
            return 0
        existing = {str(item.get("message_id")): item for item in data["messages"]}
        existing_events = {
            (str(item.get("provider_account_id") or "feishu_default"), str(item.get("event_id") or "")): item
            for item in data["messages"] if item.get("event_id")
        }
        added = 0
        for raw_message in messages:
            sender = _as_dict(raw_message.get("sender"))
            message_type = str(raw_message.get("msg_type") or raw_message.get("message_type") or "")
            if sender.get("sender_type") == "app" or message_type == "system":
                continue
            normalized = self.normalize_message(chat_id, raw_message)
            message_id = normalized["message_id"]
            if not message_id:
                continue
            event_key = (
                str(normalized.get("provider_account_id") or "feishu_default"),
                str(normalized.get("event_id") or ""),
            )
            previous = existing.get(message_id) or (existing_events.get(event_key) if event_key[1] else None)
            if previous:
                if previous.get("message_id") != message_id:
                    message_id = str(previous.get("message_id") or message_id)
                    normalized["message_id"] = message_id
                refreshing_legacy = (
                    previous.get("extraction_status") == "legacy_pending"
                    and normalized.get("extraction_status") in {"completed", "partial", "failed"}
                )
                lifecycle = {
                    key: previous.get(key)
                    for key in ("content_status", "confidence", "reason", "review_status", "knowledge_file", "candidate_id", "asset_id", "reviewed_at")
                    if key in previous
                }
                if previous.get("extraction_status") and not normalized.get("extraction_status"):
                    enrichment = {
                        key: previous.get(key)
                        for key in (
                            "content", "title", "image_key", "image_keys", "links", "resource_files",
                            "resource_count", "extraction_status", "extraction_method", "extraction_error",
                            "original_content", "content_hash",
                        )
                        if key in previous
                    }
                    normalized.update(enrichment)
                if refreshing_legacy:
                    assessment = evaluate_content(normalized, group)
                    normalized = {
                        **previous, **normalized, **assessment,
                        "review_status": _compat_review_status(assessment["content_status"]),
                        "candidate_id": previous.get("candidate_id") or "",
                        "asset_id": previous.get("asset_id") or "",
                    }
                    self._append_audit(data, "item_enriched", "message", message_id, assessment["reason"])
                else:
                    normalized = {**previous, **normalized, **lifecycle}
            else:
                refreshing_legacy = False
                assessment = evaluate_content(normalized, group)
                normalized.update(assessment)
                normalized["review_status"] = _compat_review_status(assessment["content_status"])
                added += 1
                self._append_audit(data, "item_received", "message", message_id, assessment["reason"])
            existing[message_id] = normalized
            if event_key[1]:
                existing_events[event_key] = normalized
            if (not previous or refreshing_legacy) and normalized.get("content_status") in {"candidate", "review_required"}:
                data["messages"] = list(existing.values())
                self._upsert_candidate(data, group, normalized)

        retention_by_chat = {str(item.get("chat_id") or ""): int(item.get("retention_days") or 90) for item in data["groups"]}
        now = datetime.now()
        retained_messages = []
        for message in existing.values():
            retention_days = retention_by_chat.get(str(message.get("chat_id") or ""))
            created_at = _message_time(message.get("create_time"))
            if retention_days and created_at and created_at < now - timedelta(days=retention_days):
                continue
            retained_messages.append(message)
        data["messages"] = sorted(
            retained_messages,
            key=lambda item: (str(item.get("create_time") or ""), str(item.get("message_id") or "")),
            reverse=True,
        )[:5000]
        retained_ids = {item.get("message_id") for item in data["messages"]}
        for candidate in data["candidates"]:
            candidate["source_message_ids"] = [item for item in candidate.get("source_message_ids", []) if item in retained_ids]
            candidate["message_count"] = len(candidate["source_message_ids"])
        data["candidates"] = [item for item in data["candidates"] if item.get("source_message_ids")]
        for item in data["groups"]:
            if item.get("chat_id") == chat_id:
                item["last_synced_at"] = _now()
                item["last_sync_count"] = added
                item["updated_at"] = _now()
        self.save(data)
        return added

    def list_messages(self, chat_id: str = "", keyword: str = "", limit: int = 120) -> List[Dict[str, Any]]:
        return self.list_items(chat_id=chat_id, keyword=keyword, limit=limit)

    def list_items(self, chat_id: str = "", keyword: str = "", content_status: str = "", message_type: str = "", limit: int = 300) -> List[Dict[str, Any]]:
        keyword = keyword.strip().lower()
        filtered = []
        for message in self.load()["messages"]:
            if chat_id and message.get("chat_id") != chat_id:
                continue
            if content_status and message.get("content_status") != content_status:
                continue
            if message_type and message.get("message_type") != message_type:
                continue
            haystack = " ".join(str(message.get(key) or "") for key in ("content", "sender", "file_name", "message_type", "reason")).lower()
            if keyword and keyword not in haystack:
                continue
            filtered.append(message)
        return filtered[:max(1, min(limit, 500))]

    def query_items(
        self,
        chat_id: str = "",
        keyword: str = "",
        content_statuses: Sequence[str] = (),
        excluded_statuses: Sequence[str] = (),
        message_type: str = "",
        date_from: str = "",
        date_to: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页查询内容中心消息，避免将完整历史一次性发送到浏览器。"""
        keyword = keyword.strip().lower()
        allowed_statuses = {str(item).strip() for item in content_statuses if str(item).strip()}
        blocked_statuses = {str(item).strip() for item in excluded_statuses if str(item).strip()}
        start_at = _message_time(date_from)
        end_at = _message_time(date_to)
        filtered = []
        for message in self.load()["messages"]:
            if chat_id and message.get("chat_id") != chat_id:
                continue
            status = str(message.get("content_status") or "")
            if allowed_statuses and status not in allowed_statuses:
                continue
            if blocked_statuses and status in blocked_statuses:
                continue
            if message_type and message.get("message_type") != message_type:
                continue
            created_at = _message_time(message.get("create_time") or message.get("synced_at"))
            if start_at and (not created_at or created_at < start_at):
                continue
            if end_at and (not created_at or created_at > end_at):
                continue
            haystack = " ".join(
                str(message.get(key) or "")
                for key in ("content", "title", "sender", "file_name", "message_type", "reason")
            ).lower()
            if keyword and keyword not in haystack:
                continue
            filtered.append(message)

        filtered.sort(
            key=lambda item: (
                _message_time(item.get("create_time") or item.get("synced_at")) or datetime.min,
                str(item.get("message_id") or ""),
            ),
            reverse=True,
        )
        safe_page_size = max(1, min(int(page_size or 20), 100))
        safe_page = max(1, int(page or 1))
        total = len(filtered)
        pages = max(1, math.ceil(total / safe_page_size))
        safe_page = min(safe_page, pages)
        offset = (safe_page - 1) * safe_page_size
        return {
            "items": filtered[offset:offset + safe_page_size],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
            "pages": pages,
        }

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.load()["messages"] if item.get("message_id") == message_id), None)

    def existing_message_ids(self) -> set[str]:
        return {
            str(item.get("message_id") or "")
            for item in self.load()["messages"]
            if item.get("message_id") and item.get("extraction_status") != "legacy_pending"
        }

    def replace_enriched_message(self, chat_id: str, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """替换图片/富文本的提取结果，并重新计算知识候选状态。"""
        data = self.load()
        group = next((item for item in data["groups"] if item.get("chat_id") == chat_id), None)
        if not group:
            return None
        normalized = self.normalize_message(chat_id, raw_message)
        message_id = normalized.get("message_id")
        previous = next((item for item in data["messages"] if item.get("message_id") == message_id), None)
        if not previous:
            self.upsert_messages(chat_id, [raw_message])
            return self.get_message(str(message_id or ""))
        if previous.get("content_status") == "published":
            raise ValueError("已入库内容不能直接重新识别，请先撤销对应知识资产")

        assessment = evaluate_content(normalized, group)
        normalized = {
            **previous,
            **normalized,
            **assessment,
            "review_status": _compat_review_status(assessment["content_status"]),
            "candidate_id": previous.get("candidate_id") or "",
            "asset_id": previous.get("asset_id") or "",
        }
        data["messages"] = [normalized if item.get("message_id") == message_id else item for item in data["messages"]]

        candidate_id = str(previous.get("candidate_id") or "")
        candidate = next((item for item in data["candidates"] if item.get("candidate_id") == candidate_id), None)
        if normalized.get("content_status") in {"candidate", "review_required"}:
            self._upsert_candidate(data, group, normalized)
        elif candidate:
            candidate["source_message_ids"] = [item for item in candidate.get("source_message_ids", []) if item != message_id]
            candidate["message_count"] = len(candidate["source_message_ids"])
            if not candidate["source_message_ids"]:
                data["candidates"] = [item for item in data["candidates"] if item.get("candidate_id") != candidate_id]

        self._append_audit(data, "item_enriched", "message", str(message_id or ""), assessment["reason"])
        self.save(data)
        return self.get_message(str(message_id or ""))

    def list_candidates(self, status: str = "", chat_id: str = "", limit: int = 300) -> List[Dict[str, Any]]:
        data = self.load()
        messages = {str(item.get("message_id") or ""): item for item in data["messages"]}
        result = []
        for candidate in sorted(data["candidates"], key=lambda item: str(item.get("updated_at") or ""), reverse=True):
            if status and candidate.get("status") != status:
                continue
            if chat_id and candidate.get("chat_id") != chat_id:
                continue
            source_messages = [messages[item] for item in candidate.get("source_message_ids", []) if item in messages]
            result.append({**candidate, "messages": source_messages})
        return result[:max(1, min(limit, 500))]

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.list_candidates(limit=500) if item.get("candidate_id") == candidate_id), None)

    def ensure_candidate_for_messages(self, message_ids: Sequence[str], category: str = "") -> Optional[Dict[str, Any]]:
        data = self.load()
        selected = [item for item in data["messages"] if item.get("message_id") in set(message_ids)]
        if not selected:
            return None
        group = next((item for item in data["groups"] if item.get("chat_id") == selected[0].get("chat_id")), {
            "chat_id": selected[0].get("chat_id"), "collection_mode": "review",
            "default_category": category or "proj_communication", "aggregation_window_minutes": 30,
        })
        candidate = None
        for message in sorted(selected, key=lambda item: str(item.get("create_time") or "")):
            message["content_status"] = "review_required"
            message["reason"] = message.get("reason") or "由兼容审核入口创建"
            message["confidence"] = float(message.get("confidence") or 0.5)
            message["review_status"] = "pending"
            candidate = self._upsert_candidate(data, group, message)
        if candidate and category:
            candidate["category"] = category
        self.save(data)
        return self.get_candidate(str(candidate.get("candidate_id"))) if candidate else None

    def set_review_status(self, message_id: str, status: str, knowledge_file: str = "") -> Optional[Dict[str, Any]]:
        if status not in REVIEW_STATUSES:
            raise ValueError("无效的审核状态")
        data = self.load()
        for message in data["messages"]:
            if message.get("message_id") == message_id:
                message["review_status"] = status
                message["content_status"] = {"pending": "review_required", "approved": "published", "ignored": "excluded"}[status]
                message["reviewed_at"] = _now()
                if knowledge_file:
                    message["knowledge_file"] = knowledge_file
                self._append_audit(data, f"message_{status}", "message", message_id, "兼容单条审核入口")
                self.save(data)
                return message
        return None

    def update_candidate(
        self,
        candidate_id: str,
        action: str,
        category: str = "",
        error: str = "",
        preserve_published_ids: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        data = self.load()
        candidate = next((item for item in data["candidates"] if item.get("candidate_id") == candidate_id), None)
        if not candidate:
            return None
        if action == "exclude":
            candidate["status"] = "excluded"
        elif action == "retry":
            candidate["status"] = "ready_for_auto"
            candidate["last_error"] = ""
        elif action == "category":
            if not category:
                raise ValueError("请选择知识分类")
            candidate["category"] = category
        elif action == "failed":
            candidate["status"] = "failed"
            candidate["last_error"] = error[:500]
        else:
            raise ValueError("无效的候选操作")
        candidate["updated_at"] = _now()
        preserved_ids = set(preserve_published_ids or [])
        for message in data["messages"]:
            if message.get("message_id") in candidate.get("source_message_ids", []):
                if action == "exclude":
                    message["content_status"] = "excluded"
                    message["review_status"] = "ignored"
                elif action == "retry":
                    message["content_status"] = "candidate"
                elif action == "failed":
                    if message.get("message_id") in preserved_ids:
                        message["content_status"] = "published"
                        message["review_status"] = "approved"
                    else:
                        message["content_status"] = "failed"
        self._append_audit(data, f"candidate_{action}", "candidate", candidate_id, category or error or action)
        self.save(data)
        return self.get_candidate(candidate_id)

    def publish_candidate(self, candidate_id: str, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self.load()
        candidate = next((item for item in data["candidates"] if item.get("candidate_id") == candidate_id), None)
        if not candidate:
            return None
        asset_id = str(asset.get("asset_id") or _asset_id(candidate_id))
        previous_asset = next((item for item in data["assets"] if item.get("asset_id") == asset_id), {})
        asset_record = {
            **previous_asset, **asset, "asset_id": asset_id, "candidate_id": candidate_id,
            "source_message_ids": list(candidate.get("source_message_ids", [])), "status": "published",
            "published_at": _now(), "reverted_at": "",
        }
        data["assets"] = [item for item in data["assets"] if item.get("asset_id") != asset_id] + [asset_record]
        candidate.update({"status": "published", "asset_id": asset_id, "last_error": "", "updated_at": _now(), "update_existing_asset": False})
        for message in data["messages"]:
            if message.get("message_id") in candidate.get("source_message_ids", []):
                message.update({
                    "content_status": "published", "review_status": "approved", "asset_id": asset_id,
                    "knowledge_file": str(asset_record.get("source_file") or ""), "reviewed_at": _now(),
                })
        self._append_audit(
            data,
            "asset_published",
            "asset",
            asset_id,
            f"聚合 {len(candidate.get('source_message_ids', []))} 条消息",
            actor=str(asset_record.get("published_by") or "system"),
        )
        self.save(data)
        return self.get_candidate(candidate_id)

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.load()["assets"] if item.get("asset_id") == asset_id), None)

    def revert_asset_authoritatively(
        self, asset_id: str, reason: str = "管理员撤销", actor: str = "system",
    ) -> Optional[Dict[str, Any]]:
        """Commit Feishu lifecycle state while retaining original evidence."""
        data = self.load()
        asset = next((item for item in data["assets"] if item.get("asset_id") == asset_id), None)
        if not asset:
            return None
        already_reverted = asset.get("status") == "reverted"
        if not already_reverted:
            reverted_at = _now()
            asset["status"] = "reverted"
            asset["reverted_at"] = reverted_at
            candidate_id = str(asset.get("candidate_id") or "")
            for candidate in data["candidates"]:
                if candidate.get("candidate_id") == candidate_id:
                    candidate["status"] = "reverted"
                    candidate["updated_at"] = reverted_at
            source_ids = set(asset.get("source_message_ids", []))
            for message in data["messages"]:
                if message.get("message_id") in source_ids:
                    message["content_status"] = "reverted"
                    message["review_status"] = "ignored"
                    message["reverted_at"] = reverted_at
            self._append_audit(data, "asset_reverted", "asset", asset_id, reason, actor=actor)
        asset["projection_status"] = "pending_refresh"
        asset["projection_errors"] = []
        asset["projection_updated_at"] = _now()
        result = self.repository.revert_feishu_asset(data, asset_id)
        result["already_reverted"] = already_reverted
        return result

    def update_asset_projection(
        self, asset_id: str, status: str, errors: Sequence[str] = (),
    ) -> Dict[str, Any]:
        with self._lock:
            return self.repository.update_feishu_asset_projection(asset_id, status, errors)["asset"]

    def revert_asset(
        self, asset_id: str, reason: str = "管理员撤销", actor: str = "system",
    ) -> Optional[Dict[str, Any]]:
        """Compatibility facade returning the reverted asset record."""
        result = self.revert_asset_authoritatively(asset_id, reason=reason, actor=actor)
        return dict(result.get("asset") or {}) if result else None

    def list_assets(self, status: str = "") -> List[Dict[str, Any]]:
        assets = [item for item in self.load()["assets"] if not status or item.get("status") == status]
        return sorted(
            assets,
            key=lambda item: str(item.get("reverted_at") or item.get("published_at") or ""),
            reverse=True,
        )

    def query_assets(
        self,
        statuses: Sequence[str] = (),
        keyword: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页查询知识资产；当前资产与撤销历史由调用方明确分开。"""
        allowed_statuses = {str(item).strip() for item in statuses if str(item).strip()}
        keyword = keyword.strip().lower()
        filtered = []
        for asset in self.list_assets():
            if allowed_statuses and str(asset.get("status") or "") not in allowed_statuses:
                continue
            haystack = " ".join(
                str(asset.get(key) or "")
                for key in ("source_file", "stored_file", "category", "published_by")
            ).lower()
            if keyword and keyword not in haystack:
                continue
            filtered.append(asset)

        safe_page_size = max(1, min(int(page_size or 20), 100))
        safe_page = max(1, int(page or 1))
        total = len(filtered)
        pages = max(1, math.ceil(total / safe_page_size))
        safe_page = min(safe_page, pages)
        offset = (safe_page - 1) * safe_page_size
        return {
            "items": filtered[offset:offset + safe_page_size],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
            "pages": pages,
        }

    def preview_policy(self, chat_id: str, policy: Dict[str, Any], limit: int = 100) -> Dict[str, Any]:
        group = {**(self.get_group(chat_id) or {"chat_id": chat_id}), **policy}
        mode = str(group.get("collection_mode") or "auto")
        if mode not in COLLECTION_MODES:
            raise ValueError("无效的采集模式")
        counts = {"auto_publish": 0, "review_required": 0, "archived": 0, "excluded": 0, "not_collected": 0}
        sample_reasons: Dict[str, int] = {}
        for message in self.list_items(chat_id=chat_id, limit=limit):
            result = evaluate_content(message, group)
            status = result["content_status"]
            if status == "candidate":
                counts["auto_publish"] += 1
            elif status in counts:
                counts[status] += 1
            sample_reasons[result["reason"]] = sample_reasons.get(result["reason"], 0) + 1
        return {
            "chat_id": chat_id, "sample_size": sum(counts.values()), "counts": counts,
            "top_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(sample_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
            ],
        }

    def _append_audit(self, data: Dict[str, Any], action: str, object_type: str, object_id: str, detail: str = "", actor: str = "system") -> Dict[str, Any]:
        raw = f"{_now()}:{action}:{object_type}:{object_id}:{len(data.get('audit', []))}"
        event = {
            "audit_id": f"audit_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}",
            "actor": actor, "action": action, "object_type": object_type, "object_id": object_id,
            "detail": str(detail or "")[:500], "created_at": _now(),
        }
        data.setdefault("audit", []).append(event)
        data["audit"] = data["audit"][-2000:]
        return event

    def record_audit(self, action: str, object_type: str, object_id: str, detail: str = "", actor: str = "system") -> Dict[str, Any]:
        data = self.load()
        event = self._append_audit(data, action, object_type, object_id, detail, actor)
        self.save(data)
        return event

    def list_audit(self, action: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        events = list(reversed(self.load()["audit"]))
        if action:
            events = [item for item in events if item.get("action") == action]
        return events[:max(1, min(limit, 500))]

    def record_event(self, event_type: str, action: str, message_id: str = "", error: str = "") -> Dict[str, Any]:
        data = self.load()
        diagnostics = data["diagnostics"]
        diagnostics.update({
            "last_event_at": _now(), "last_event_type": str(event_type or "unknown"),
            "last_action": str(action or "received"), "last_message_id": str(message_id or ""),
            "last_error": str(error or "")[:300],
        })
        self._append_audit(data, action or "event_received", "event", message_id or event_type, error)
        self.save(data)
        return diagnostics

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self.load()["diagnostics"])

    def summary(self) -> Dict[str, Any]:
        data = self.load()
        messages = data["messages"]
        candidates = data["candidates"]
        project_groups = [
            group for group in data["groups"]
            if group.get("chat_type") != "p2p" and group.get("collection_mode") != "off"
        ]
        today = datetime.now().date()

        def is_today(value: Any) -> bool:
            parsed = _message_time(value)
            return bool(parsed and parsed.date() == today)

        review_count = sum(1 for item in candidates if item.get("status") in {"review_required", "failed"})
        published_assets = [item for item in data["assets"] if item.get("status") == "published"]
        reverted_assets = [item for item in data["assets"] if item.get("status") == "reverted"]
        today_messages = [item for item in messages if is_today(item.get("create_time"))]
        today_candidates = [item for item in candidates if is_today(item.get("created_at"))]
        evaluated_statuses = {"candidate", "published", "archived", "excluded", "review_required", "reverted"}
        valuable_statuses = {"candidate", "published", "review_required"}
        automatically_handled_statuses = {"published", "archived", "excluded"}
        evaluated_count = sum(1 for item in messages if item.get("content_status") in evaluated_statuses)
        published_message_count = sum(1 for item in messages if item.get("content_status") == "published")
        automatically_handled_count = sum(
            1 for item in messages if item.get("content_status") in automatically_handled_statuses
        )
        today_evaluated_count = sum(
            1 for item in today_messages if item.get("content_status") in evaluated_statuses
        )
        today_valuable_count = sum(
            1 for item in today_messages if item.get("content_status") in valuable_statuses
        )
        today_published_keys = {
            str(item.get("candidate_id") or item.get("asset_id") or item.get("knowledge_file") or item.get("message_id"))
            for item in today_messages
            if item.get("content_status") == "published"
        }
        today_published_count = len(today_published_keys)
        today_completed_count = sum(1 for item in published_assets if is_today(item.get("published_at")))
        today_auto_handled_count = sum(
            1 for item in today_messages if item.get("content_status") in automatically_handled_statuses
        )
        message_count = len(messages)
        candidate_count = len(candidates)
        conversion_rate = round(len(published_assets) * 100 / message_count, 1) if message_count else 0.0
        candidate_conversion_rate = round(len(published_assets) * 100 / candidate_count, 1) if candidate_count else 0.0
        automation_rate = round(automatically_handled_count * 100 / message_count, 1) if message_count else 0.0
        today_conversion_rate = round(today_published_count * 100 / len(today_messages), 1) if today_messages else 0.0
        today_automation_rate = round(today_auto_handled_count * 100 / len(today_messages), 1) if today_messages else 0.0
        return {
            "group_count": len(project_groups), "conversation_count": len(data["groups"]),
            "message_count": message_count, "today_received": len(today_messages),
            "pending_count": review_count, "review_required_count": review_count,
            "approved_count": len(published_assets), "published_count": len(published_assets),
            "reverted_count": len(reverted_assets),
            "auto_published_count": sum(
                1 for item in data["audit"]
                if item.get("action") == "asset_published" and item.get("actor", "system") == "system"
            ),
            "archived_count": sum(1 for item in messages if item.get("content_status") == "archived"),
            "excluded_count": sum(1 for item in messages if item.get("content_status") == "excluded"),
            "reverted_message_count": sum(1 for item in messages if item.get("content_status") == "reverted"),
            "candidate_count": candidate_count, "failed_count": sum(1 for item in candidates if item.get("status") == "failed"),
            "evaluated_count": evaluated_count,
            "published_message_count": published_message_count,
            "automatically_handled_count": automatically_handled_count,
            "today_evaluated_count": today_evaluated_count,
            "today_valuable_count": today_valuable_count,
            "today_candidate_count": len(today_candidates),
            "today_published_count": today_published_count,
            "today_completed_count": today_completed_count,
            "today_auto_handled_count": today_auto_handled_count,
            "conversion_rate": conversion_rate,
            "candidate_conversion_rate": candidate_conversion_rate,
            "automation_rate": automation_rate,
            "today_conversion_rate": today_conversion_rate,
            "today_automation_rate": today_automation_rate,
        }


feishu_workspace = FeishuWorkspace()
