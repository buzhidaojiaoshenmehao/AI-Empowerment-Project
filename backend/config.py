"""应用配置"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 服务配置
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    FRONTEND_URL: Optional[str] = None

    # OpenAI / LLM 配置（用户在系统设置页面手动填入，持久化到 config.json）
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    # 本地嵌入模型（当 OPENAI_API_KEY 为空时自动 fallback）
    LOCAL_EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB 持久化路径
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # SQLite 是业务数据的权威存储；向量和自动图谱仍是可重建派生文件。
    DATABASE_PATH: str = "./data/app.db"
    DATABASE_BACKUP_DIR: str = "./data/backups"
    STORAGE_AUTO_MIGRATE: bool = True

    # 仅限本地演示：成员自动激活，登录密码为企业邮箱 @ 前缀。
    # 正式部署必须保持关闭，继续使用邀请激活和强密码策略。
    DEMO_SIMPLE_AUTH: bool = False

    # 上传文件限制
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_BATCH_UPLOAD_FILES: int = 20
    MAX_BATCH_UPLOAD_SIZE_MB: int = 100
    MAX_DOCUMENT_OCR_PAGES: int = 100
    ALLOWED_EXTENSIONS: set = {"pdf", "txt", "md", "docx", "csv", "json"}

    # 检索参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVER_K: int = 4
    RETRIEVER_SCORE_THRESHOLD: float = 0.3

    # ── 飞书集成配置 ──
    FEISHU_APP_ID: Optional[str] = None
    FEISHU_APP_SECRET: Optional[str] = None
    FEISHU_WEBHOOK_SECRET: Optional[str] = None
    FEISHU_VERIFICATION_TOKEN: Optional[str] = None
    FEISHU_BOT_NAME: str = "AI 知识库助手"
    FEISHU_DEFAULT_CHAT_ID: Optional[str] = None

    # 扫描只负责持久化提醒任务，实际投递由处理任务执行器完成。
    TASK_REMINDER_SCAN_SECONDS: int = 60
    # 每日治理扫描只创建 SQLite 调度意图，备份和清理由处理任务执行。
    GOVERNANCE_SCAN_SECONDS: int = 300

    # ── 项目知识管理 ──
    PROJECT_NAME: str = "默认项目"
    ENABLE_KNOWLEDGE_GRAPH: bool = True
    AUTO_CLASSIFY_DOCUMENTS: bool = True

    # ── HTTP 代理配置 ──
    HTTP_PROXY_ENABLED: bool = False
    HTTP_PROXY_URL: Optional[str] = None
    HTTP_VERIFY_SSL: bool = True

    # ── config.json 持久化路径（在运行时由 main.py 设置） ──
    CONFIG_JSON_PATH: Optional[str] = None


settings = Settings()
