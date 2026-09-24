"""PyInstaller 运行时钩子 —— 在 chromadb 加载前禁用遥测"""
import os

# 禁用 chromadb 遥测（避免依赖 posthog 库）
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
# 兼容旧版本 chromadb
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
