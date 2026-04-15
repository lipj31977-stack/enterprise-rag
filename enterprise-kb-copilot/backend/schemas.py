"""
Pydantic 请求/响应模型

定义 API 的输入输出数据结构，用于：
- 请求参数自动校验
- 响应数据序列化
- Swagger 文档自动生成

命名约定：
- xxxRequest:  请求体
- xxxResponse: 完整响应（含 code + message + data）
- xxxInfo:     数据对象（嵌入到 Response.data 中）
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 通用响应模型
# ============================================================

class BaseResponse(BaseModel):
    """统一响应格式 — 所有 API 响应都遵循此结构"""
    code: int = Field(default=200, description="状态码，200 表示成功")
    message: str = Field(default="success", description="提示信息")


# ============================================================
# 文档相关模型
# ============================================================

class DocumentInfo(BaseModel):
    """单个文档的详细信息"""
    id: int = Field(description="文档 ID")
    filename: str = Field(description="原始文件名")
    file_type: str = Field(description="文件类型: pdf/txt/md")
    file_size: int = Field(description="文件大小(字节)")
    chunk_count: int = Field(description="切分的文本块数量")
    status: str = Field(description="处理状态: processing/completed/failed")
    error_message: Optional[str] = Field(default=None, description="失败原因")
    created_at: datetime = Field(description="上传时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseResponse):
    """文档上传响应"""
    data: Optional[DocumentInfo] = None


class DocumentDetailResponse(BaseResponse):
    """单个文档详情响应"""
    data: Optional[DocumentInfo] = None


class DocumentListData(BaseModel):
    """文档列表数据"""
    total: int = Field(description="文档总数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页条数")
    documents: list[DocumentInfo] = Field(description="文档列表")


class DocumentListResponse(BaseResponse):
    """文档列表响应"""
    data: Optional[DocumentListData] = None


# ============================================================
# 聊天相关模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="用户问题"
    )
    session_id: Optional[int] = Field(
        default=None,
        description="会话 ID，为空则自动创建新会话"
    )
    top_k: int = Field(
        default=4, ge=1, le=10,
        description="检索返回的文档块数量"
    )


class SourceChunk(BaseModel):
    """引用来源片段"""
    document_name: str = Field(description="来源文档名")
    chunk_content: str = Field(description="引用的文本内容")
    page: Optional[int] = Field(default=None, description="所在页码")
    relevance_score: Optional[float] = Field(default=None, description="相关度分数")


class ChatResponseData(BaseModel):
    """聊天响应数据"""
    answer: str = Field(description="AI 回答")
    sources: list[SourceChunk] = Field(default_factory=list, description="引用来源")
    session_id: int = Field(description="会话 ID")
    message_id: int = Field(description="消息 ID")


class ChatResponse(BaseResponse):
    """聊天响应"""
    data: Optional[ChatResponseData] = None


# ---------- 会话相关 ----------

class SessionInfo(BaseModel):
    """会话信息"""
    id: int = Field(description="会话 ID")
    title: str = Field(description="会话标题")
    message_count: int = Field(description="消息数量")
    is_active: bool = Field(description="是否激活")
    created_at: datetime = Field(description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="最后活跃时间")

    model_config = {"from_attributes": True}


class SessionListData(BaseModel):
    """会话列表数据"""
    total: int
    sessions: list[SessionInfo]


class SessionListResponse(BaseResponse):
    """会话列表响应"""
    data: Optional[SessionListData] = None


# ---------- 消息相关 ----------

class MessageInfo(BaseModel):
    """单条消息信息"""
    id: int = Field(description="消息 ID")
    session_id: int = Field(description="会话 ID")
    question: str = Field(description="用户问题")
    answer: str = Field(description="AI 回答")
    sources: Optional[list] = Field(default=None, description="引用来源")
    response_time: Optional[float] = Field(default=None, description="响应耗时(秒)")
    created_at: datetime = Field(description="消息时间")

    model_config = {"from_attributes": True}


class MessageListData(BaseModel):
    """消息列表数据"""
    total: int
    session_id: int
    messages: list[MessageInfo]


class MessageListResponse(BaseResponse):
    """消息列表响应"""
    data: Optional[MessageListData] = None


# ============================================================
# 健康检查模型
# ============================================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="healthy", description="服务状态")
    database: str = Field(default="unknown", description="数据库连接状态")
    vector_store: str = Field(default="unknown", description="向量库状态")
