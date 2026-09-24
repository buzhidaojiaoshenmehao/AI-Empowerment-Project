"""飞书事件长连接管理。

本地运行时使用飞书官方 SDK 主动连接事件通道，无需将整个应用暴露到公网。
HTTP Webhook 仍由 ``backend.main`` 保留，供正式部署环境使用。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import gc
import importlib
from datetime import datetime
from typing import Any, Dict, Optional

from backend.config import settings


logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class FeishuLongConnection:
    """在独立线程中运行 lark-oapi WebSocket 客户端。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._sdk_loop: Optional[asyncio.AbstractEventLoop] = None
        self._app_loop: Optional[asyncio.AbstractEventLoop] = None
        self._state = "stopped"
        self._started_at = ""
        self._last_error = ""
        self._stop_requested = False

    def start(self, app_loop: asyncio.AbstractEventLoop) -> bool:
        """启动长连接；重复调用不会创建第二个客户端。"""
        if not (settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET):
            with self._lock:
                self._state = "unconfigured"
                self._last_error = "请先保存飞书 App ID 和 App Secret"
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return not self._stop_requested
            self._app_loop = app_loop
            self._state = "connecting"
            self._started_at = _now()
            self._last_error = ""
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="feishu-long-connection",
                daemon=True,
            )
            self._thread.start()
        return True

    def restart(self, app_loop: asyncio.AbstractEventLoop) -> bool:
        self.stop()
        return self.start(app_loop)

    def stop(self) -> None:
        """停止 SDK 连接；进程退出时也会由守护线程自动回收。"""
        with self._lock:
            self._stop_requested = True
            client = self._client
            sdk_loop = self._sdk_loop
            thread = self._thread

        if client is not None and sdk_loop is not None and sdk_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(client._disconnect(), sdk_loop)
                future.result(timeout=3)
            except Exception:
                pass
            try:
                sdk_loop.call_soon_threadsafe(self._cancel_sdk_tasks, sdk_loop)
            except RuntimeError:
                pass

        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=4)

        with self._lock:
            # A blocked SDK request may outlive join's timeout. Keep its owner
            # until it exits so a retry cannot rebind the SDK's global loop
            # while the previous client is still using it.
            if thread and thread.is_alive():
                self._state = "stopping"
            else:
                self._state = "stopped"
                self._client = None
                self._thread = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            thread_alive = bool(self._thread and self._thread.is_alive())
            connected = bool(thread_alive and self._client is not None and getattr(self._client, "_conn", None))
            state = "connected" if connected else self._state
            return {
                "mode": "long_connection",
                "state": state,
                "running": thread_alive,
                "connected": connected,
                "started_at": self._started_at,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        # lark-oapi caches its loop at module import time. Importing it again
        # does not replace the loop closed by a previous connection attempt.
        sdk_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(sdk_loop)
        with self._lock:
            self._sdk_loop = sdk_loop

        try:
            import lark_oapi as lark
            sdk_client_module = importlib.import_module("lark_oapi.ws.client")
            sdk_client_module.loop = sdk_loop

            def on_message(event: Any) -> None:
                try:
                    payload = json.loads(lark.JSON.marshal(event) or "{}")
                    app_loop = self._app_loop
                    if app_loop is None or app_loop.is_closed():
                        raise RuntimeError("应用事件循环不可用")
                    from backend.feishu_bot import handle_feishu_webhook

                    future = asyncio.run_coroutine_threadsafe(handle_feishu_webhook(payload), app_loop)
                    future.add_done_callback(self._log_dispatch_result)
                except Exception as exc:
                    self._set_error(f"消息分发失败: {exc}")

            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(on_message)
                .build()
            )
            client = lark.ws.Client(
                settings.FEISHU_APP_ID,
                settings.FEISHU_APP_SECRET,
                log_level=lark.LogLevel.WARNING,
                event_handler=event_handler,
                auto_reconnect=True,
            )
            with self._lock:
                if self._stop_requested:
                    return
                self._client = client
                self._state = "connecting"
            client.start()
        except RuntimeError as exc:
            # 主动停止事件循环时 SDK 会抛出此异常，不应展示为连接故障。
            if not self._stop_requested:
                self._set_error(str(exc))
        except Exception as exc:
            self._set_error(str(exc))
            logger.exception("飞书长连接启动失败")
        finally:
            with self._lock:
                if self._stop_requested:
                    self._state = "stopped"
                elif self._state != "error":
                    self._state = "disconnected"
                self._client = None
                self._sdk_loop = None
            try:
                pending = [task for task in asyncio.all_tasks(sdk_loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending and not sdk_loop.is_running():
                    sdk_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                client = None
                gc.collect()
                sdk_loop.close()
            except Exception:
                pass

    @staticmethod
    def _cancel_sdk_tasks(sdk_loop: asyncio.AbstractEventLoop) -> None:
        for task in asyncio.all_tasks(sdk_loop):
            task.cancel()
        sdk_loop.stop()

    def _log_dispatch_result(self, future: Any) -> None:
        try:
            future.result()
        except Exception as exc:
            self._set_error(f"消息处理失败: {exc}")
            logger.exception("飞书消息处理失败")

    def _set_error(self, message: str) -> None:
        # 仅保留简短错误，不记录凭证或完整连接 URL。
        safe_message = str(message or "未知错误")[:300]
        if settings.FEISHU_APP_SECRET:
            safe_message = safe_message.replace(settings.FEISHU_APP_SECRET, "***")
        with self._lock:
            self._state = "error"
            self._last_error = safe_message


feishu_long_connection = FeishuLongConnection()
