import threading
import unittest
from unittest.mock import patch

from backend.local_ocr import (
    LocalOCRInvalidResult,
    LocalOCRUnavailable,
    LocalRapidOCR,
)


class RapidOCROutputFake:
    def __init__(self, texts, scores):
        self.txts = tuple(texts)
        self.scores = tuple(scores)


class LocalRapidOCRTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_output_is_lazy_reused_and_runs_in_worker_thread(self):
        caller_thread = threading.get_ident()
        engine_calls = []
        factory_calls = []

        class Engine:
            def __call__(self, image_bytes):
                engine_calls.append((threading.get_ident(), image_bytes))
                return RapidOCROutputFake(["发布结论", "回滚方案待确认"], [0.98, 0.96])

        def factory():
            factory_calls.append(threading.get_ident())
            return Engine()

        service = LocalRapidOCR(engine_factory=factory)
        self.assertFalse(service.initialized)

        first = await service.recognize(b"first-image")
        second = await service.recognize(b"second-image")

        self.assertEqual(first, "发布结论\n回滚方案待确认")
        self.assertEqual(second, first)
        self.assertTrue(service.initialized)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(len(engine_calls), 2)
        self.assertTrue(all(thread_id != caller_thread for thread_id, _ in engine_calls))

    async def test_legacy_tuple_output_is_supported(self):
        class Engine:
            def __call__(self, image_bytes):
                return ([
                    [[[0, 0], [1, 0], [1, 1], [0, 1]], "风险确认", 0.93],
                    [[[0, 2], [1, 2], [1, 3], [0, 3]], "低置信噪声", 0.12],
                ], 0.1)

        service = LocalRapidOCR(engine_factory=Engine)
        self.assertEqual(await service.recognize(b"image"), "风险确认")

    async def test_missing_optional_dependency_is_nonfatal_to_caller(self):
        service = LocalRapidOCR()
        with patch("backend.local_ocr.importlib.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(LocalOCRUnavailable, "可选依赖未安装"):
                await service.recognize(b"image")

    async def test_obviously_invalid_result_is_rejected(self):
        class Engine:
            def __call__(self, image_bytes):
                return RapidOCROutputFake(["..."], [0.99])

        service = LocalRapidOCR(engine_factory=Engine)
        with self.assertRaisesRegex(LocalOCRInvalidResult, "有效文字"):
            await service.recognize(b"image")


if __name__ == "__main__":
    unittest.main()
