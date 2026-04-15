"""
前端工具函数

封装对 FastAPI 后端的所有 HTTP 调用。
Streamlit 页面通过调用这些函数与后端通信，不直接使用 requests。

所有函数都包含异常处理，失败时返回统一格式的错误响应，
前端页面不需要再做 try/except。
"""

import os
import requests
from typing import Optional


# 后端 API 地址，优先从环境变量读取
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# 请求超时设置（秒）
TIMEOUT_SHORT = 10     # 普通查询
TIMEOUT_UPLOAD = 120   # 文件上传（含解析+向量化）
TIMEOUT_CHAT = 120     # 问答（含 LLM 生成）


def _api_url(path: str) -> str:
    """拼接完整的 API URL"""
    return f"{BACKEND_URL}{path}"


def _error_response(message: str) -> dict:
    """构造统一的错误响应"""
    return {"code": 500, "message": message}


# ============================================================
# 系统相关
# ============================================================

def health_check() -> dict:
    """
    健康检查。

    Returns:
        {"status": "healthy", "database": "connected", "vector_store": "loaded"}
    """
    try:
        resp = requests.get(_api_url("/api/v1/health"), timeout=5)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unhealthy", "database": "unknown", "vector_store": "unknown"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# 文档管理
# ============================================================

def upload_document(file) -> dict:
    """
    上传文档（触发解析→切分→向量化全流程）。

    Args:
        file: Streamlit 的 UploadedFile 对象
    """
    try:
        files = {
            "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
        }
        resp = requests.post(
            _api_url("/api/v1/documents/upload"),
            files=files,
            timeout=TIMEOUT_UPLOAD,
        )
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("上传超时，文件可能太大，请稍后重试")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务，请确认已启动")
    except Exception as e:
        return _error_response(f"上传失败: {str(e)}")


def get_documents(page: int = 1, page_size: int = 50) -> dict:
    """获取文档列表"""
    try:
        resp = requests.get(
            _api_url("/api/v1/documents"),
            params={"page": page, "page_size": page_size},
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"获取文档列表失败: {str(e)}")


def get_document_detail(document_id: int) -> dict:
    """获取单个文档详情"""
    try:
        resp = requests.get(
            _api_url(f"/api/v1/documents/{document_id}"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取文档详情失败: {str(e)}")


def delete_document(document_id: int) -> dict:
    """删除文档"""
    try:
        resp = requests.delete(
            _api_url(f"/api/v1/documents/{document_id}"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"删除失败: {str(e)}")


def get_document_stats() -> dict:
    """获取文档统计信息"""
    try:
        resp = requests.get(
            _api_url("/api/v1/documents/stats/overview"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取统计失败: {str(e)}")


# ============================================================
# 聊天问答
# ============================================================

def chat_with_kb(
    question: str,
    session_id: Optional[int] = None,
    top_k: int = 4,
) -> dict:
    """
    向知识库提问（RAG 问答）。

    Args:
        question: 用户问题
        session_id: 会话 ID（为空则自动创建新会话）
        top_k: 检索文档块数量
    """
    try:
        payload = {"question": question, "top_k": top_k}
        if session_id is not None:
            payload["session_id"] = session_id

        resp = requests.post(
            _api_url("/api/v1/chat"),
            json=payload,
            timeout=TIMEOUT_CHAT,
        )
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("回答生成超时，请稍后重试")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"问答失败: {str(e)}")


def get_sessions(page: int = 1, page_size: int = 50) -> dict:
    """获取聊天会话列表"""
    try:
        resp = requests.get(
            _api_url("/api/v1/chat/sessions"),
            params={"page": page, "page_size": page_size},
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"获取会话列表失败: {str(e)}")


def get_session_messages(session_id: int) -> dict:
    """获取指定会话的所有消息"""
    try:
        resp = requests.get(
            _api_url(f"/api/v1/chat/sessions/{session_id}/messages"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取消息失败: {str(e)}")


def delete_session(session_id: int) -> dict:
    """删除指定会话"""
    try:
        resp = requests.delete(
            _api_url(f"/api/v1/chat/sessions/{session_id}"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"删除会话失败: {str(e)}")


# ============================================================
# 辅助函数
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def is_backend_connected() -> bool:
    """快速检查后端是否可连接"""
    result = health_check()
    return result.get("status") == "healthy"
