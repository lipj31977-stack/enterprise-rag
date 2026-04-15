"""
配置管理模块

使用 pydantic-settings 从 .env 文件读取所有配置项。
其他模块通过 get_settings() 获取配置单例，避免重复读取。
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，所有值从 .env 文件读取"""

    # ---------- 聊天大模型 配置 ----------
    openai_api_key: str = "sk-your-key"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-3.5-turbo"
    
    # ---------- 向量 Embedding 模型配置 ----------
    embedding_api_key: str = "sk-your-key"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model_name: str = "text-embedding-ada-002"

    # ---------- MySQL 配置 ----------
    mysql_host: str = "mysql"           # Docker默认，本地开发改为 localhost
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "kb_copilot_2026"
    mysql_database: str = "kb_copilot"

    # ---------- FastAPI 配置 ----------
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"

    # ---------- 文档处理配置 ----------
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 4

    # ---------- 文件存储路径 ----------
    upload_dir: str = "./data/uploads"
    vector_store_dir: str = "./data/vector_store"

    @property
    def database_url(self) -> str:
        """构建 SQLite 连接字符串，免安装数据库服务端"""
        import os
        # 把 sqlite 数据库文件存放在与 vector_store 同级的 data 目录下
        db_path = os.path.abspath(os.path.join(self.vector_store_dir, "..", "kb_copilot.db"))
        return f"sqlite:///{db_path}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,      # 环境变量名不区分大小写
        extra="ignore",            # 忽略 .env 中多余的变量
    )


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例。

    使用 lru_cache 保证整个应用生命周期内只读取一次 .env 文件。
    如果需要刷新配置，调用 get_settings.cache_clear()。
    """
    return Settings()
