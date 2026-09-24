"""Exercise the installed SDK's real start/loop code without Feishu traffic."""
import asyncio
import importlib
import threading
import unittest
from unittest.mock import patch

from backend.config import settings
from backend.feishu_connection import FeishuLongConnection


class FeishuConnectionTest(unittest.TestCase):
    def setUp(self):
        self.app_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.app_loop)
        self.sdk = importlib.import_module("lark_oapi.ws.client")
        self.original_sdk_loop = self.sdk.loop
        stale_loop = asyncio.new_event_loop()
        stale_loop.close()
        self.sdk.loop = stale_loop
        self.connection = FeishuLongConnection()
        self.connected = threading.Event()
        self.used_loops = []
        self.fail_next = False

        async def connect(client):
            self.used_loops.append(asyncio.get_running_loop())
            if self.fail_next:
                self.fail_next = False
                raise self.sdk.ClientException(1, "simulated connection failure")
            client._conn = object()
            self.connected.set()

        async def disconnect(client):
            client._conn = None

        async def ping(client):
            await asyncio.sleep(3600)

        self.patches = [
            patch.object(settings, "FEISHU_APP_ID", "cli_test"),
            patch.object(settings, "FEISHU_APP_SECRET", "test_secret"),
            patch.object(self.sdk.Client, "_connect", connect),
            patch.object(self.sdk.Client, "_disconnect", disconnect),
            patch.object(self.sdk.Client, "_ping_loop", ping),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.connection.stop()
        for item in reversed(self.patches):
            item.stop()
        self.sdk.loop = self.original_sdk_loop
        self.app_loop.close()
        asyncio.set_event_loop(None)

    def assert_connects(self):
        self.connected.clear()
        self.assertTrue(self.connection.start(self.app_loop))
        self.assertTrue(self.connected.wait(3), self.connection.status())
        self.assertTrue(self.connection.status()["connected"])

    def test_first_start_replaces_closed_loop_from_cached_sdk(self):
        self.assert_connects()
        thread = self.connection._thread
        self.assertTrue(self.connection.start(self.app_loop))
        self.assertIs(self.connection._thread, thread)
        self.assertEqual(len(self.used_loops), 1)

    def test_stop_then_restart_uses_fresh_loop(self):
        self.assert_connects()
        old_thread = self.connection._thread
        self.connection.stop()
        self.assertFalse(old_thread.is_alive())
        self.assertTrue(self.used_loops[0].is_closed())
        self.assertEqual(self.connection.status()["state"], "stopped")
        self.assert_connects()
        self.assertIsNot(self.used_loops[0], self.used_loops[1])

    def test_failed_start_can_be_retried(self):
        self.fail_next = True
        self.connection.start(self.app_loop)
        self.connection._thread.join(3)
        self.assertFalse(self.connection._thread.is_alive())
        self.assertEqual(self.connection.status()["state"], "error")
        self.assertIn("simulated connection failure", self.connection.status()["last_error"])
        self.assertTrue(self.used_loops[0].is_closed())
        self.assert_connects()
        self.assertEqual(self.connection.status()["last_error"], "")

    def test_stop_timeout_does_not_allow_overlapping_clients(self):
        entered = threading.Event()
        release = threading.Event()

        def stalled_start(client):
            entered.set()
            release.wait(5)

        with patch.object(self.sdk.Client, "start", stalled_start):
            self.connection.start(self.app_loop)
            self.assertTrue(entered.wait(3))
            thread = self.connection._thread
            try:
                with patch.object(thread, "join"):
                    self.connection.stop()
                self.assertEqual(self.connection.status()["state"], "stopping")
                self.assertFalse(self.connection.start(self.app_loop))
                self.assertIs(self.connection._thread, thread)
            finally:
                release.set()
                thread.join(3)
        self.assert_connects()


if __name__ == "__main__":
    unittest.main()
