"""
数据库连接模块

使用 SQLAlchemy 2.0 风格管理 MySQL 连接池。
提供 get_db() 依赖注入函数，供 FastAPI 路由使用。
包含 Docker 环境下的连接重试机制。
"""

import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

from backend.config import get_settings
from backend.utils.logger import logger


class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


def create_db_engine(max_retries: int = 5, retry_interval: int = 3):
    """
    创建数据库引擎，带自动重试（适配 Docker 中 MySQL 启动较慢的场景）。

    Args:
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）

    Returns:
        SQLAlchemy Engine 实例
    """
    settings = get_settings()

    engine = create_engine(
        settings.database_url,
        echo=False,            # 设为 True 可查看生成的 SQL（调试用）
        connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
    )

    # 尝试连接，失败则重试
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("数据库连接成功 (第 {} 次尝试)", attempt)
            return engine
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "数据库连接失败 (第 {}/{} 次尝试): {}，{}秒后重试...",
                    attempt, max_retries, str(e), retry_interval
                )
                time.sleep(retry_interval)
            else:
                logger.error("数据库连接失败，已达最大重试次数: {}", str(e))
                raise

    return engine


# 全局引擎和会话工厂 —— 延迟初始化，避免模块导入时就连接数据库
_engine = None
_SessionLocal = None


def get_engine():
    """获取全局数据库引擎（单例）"""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory():
    """获取全局会话工厂（单例）"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    数据库会话依赖注入。

    用法（在 FastAPI 路由中）:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    会话在请求结束后自动关闭，异常时自动回滚。
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    初始化数据库表。

    根据所有已注册的 ORM 模型自动创建表（如果表已存在则跳过）。
    注意：需要先 import models 模块，确保模型类已注册到 Base。
    """
    import backend.models  # noqa: F401 — 确保模型被加载
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表已同步（共 {} 张表）", len(Base.metadata.tables))


def check_db_connection() -> bool:
    """
    检查数据库连接是否正常。

    Returns:
        True 表示连接正常，False 表示连接异常
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
