"""
FastAPI 应用入口

职责：
- 创建 FastAPI 应用实例
- 注册所有路由
- 配置 CORS 中间件
- 启动时初始化数据库和日志
- 提供健康检查接口
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db, check_db_connection
from backend.utils.logger import logger, setup_logger
from backend.routers import documents, chat
from backend.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    startup: 初始化日志、创建目录、连接数据库、同步表结构
    shutdown: 清理资源（当前无需特殊处理）
    """
    # ---- 启动阶段 ----
    setup_logger()
    logger.info("=" * 60)
    logger.info("🚀 企业知识库 Copilot 后端启动中...")

    settings = get_settings()

    # 确保必要目录存在
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vector_store_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("📁 数据目录已就绪: uploads={}, vector_store={}", settings.upload_dir, settings.vector_store_dir)

    # 初始化数据库（含连接重试）
    try:
        init_db()
        logger.info("✅ 数据库初始化完成")
    except Exception as e:
        logger.error("❌ 数据库初始化失败: {}", str(e))
        raise

    logger.info("✅ 后端启动完成，监听 {}:{}", settings.backend_host, settings.backend_port)
    logger.info("📖 API 文档: http://localhost:{}/docs", settings.backend_port)
    logger.info("=" * 60)

    yield  # ---- 应用运行中 ----

    # ---- 关闭阶段 ----
    logger.info("后端服务正在关闭...")


# ============================================================
# 创建 FastAPI 应用
# ============================================================
app = FastAPI(
    title="企业知识库 Copilot API",
    description=(
        "基于 RAG 的企业知识库问答系统。\n\n"
        "支持文档上传、智能检索与问答，回答附带引用来源。\n\n"
        "技术栈：FastAPI + LangChain + FAISS + MySQL"
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI 地址
    redoc_url="/redoc",     # ReDoc 地址
    lifespan=lifespan,
)

# ============================================================
# CORS 中间件 — 允许 Streamlit 前端跨域访问
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 注册路由
# ============================================================
app.include_router(documents.router)
app.include_router(chat.router)


# ============================================================
# 健康检查接口
# ============================================================
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["系统"],
    summary="健康检查",
    description="检查后端服务、数据库连接、向量库状态。"
)
def health_check():
    settings = get_settings()

    # 检查数据库
    db_status = "connected" if check_db_connection() else "disconnected"

    # 检查向量库
    vector_store_path = os.path.join(settings.vector_store_dir, "index.faiss")
    vs_status = "loaded" if os.path.exists(vector_store_path) else "empty"

    return HealthResponse(
        status="healthy",
        database=db_status,
        vector_store=vs_status,
    )


# ============================================================
# 根路径
# ============================================================
@app.get("/", tags=["系统"], summary="根路径")
def root():
    return {
        "name": "企业知识库 Copilot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ============================================================
# 直接运行入口: python -m backend.main
# ============================================================
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
