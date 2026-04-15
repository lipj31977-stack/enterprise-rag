"""
聊天路由

提供 RAG 问答、会话管理、消息历史等接口。
问答接口调用 ChatService 实现完整 RAG：检索 → 构建上下文 → LLM 生成 → 保存记录。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ChatSession, ChatMessage
from backend.services.chat_service import ChatService
from backend.schemas import (
    BaseResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseData,
    SourceChunk,
    SessionInfo,
    SessionListData,
    SessionListResponse,
    MessageInfo,
    MessageListData,
    MessageListResponse,
)
from backend.utils.logger import logger

router = APIRouter(prefix="/api/v1/chat", tags=["智能问答"])


# ----------------------------------------------------------
# POST / — RAG 问答
# ----------------------------------------------------------
@router.post(
    "",
    response_model=ChatResponse,
    summary="RAG 问答",
    description="接收用户问题，从知识库检索相关文档，生成回答并附带引用来源。"
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    RAG 问答完整流程：
    1. 获取/创建聊天会话
    2. FAISS 向量检索最相关文档块
    3. 构建含上下文的 Prompt
    4. 调用 LLM 生成回答
    5. 保存消息记录到 MySQL
    6. 返回回答 + 引用来源
    """
    try:
        chat_service = ChatService(db)
        result = chat_service.ask(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
        )

        return ChatResponse(
            code=200,
            message="success",
            data=ChatResponseData(
                answer=result["answer"],
                sources=[
                    SourceChunk(
                        document_name=s["document_name"],
                        chunk_content=s["chunk_content"],
                        page=s.get("page"),
                        relevance_score=s.get("relevance_score"),
                    )
                    for s in result["sources"]
                ],
                session_id=result["session_id"],
                message_id=result["message_id"],
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("问答失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


# ----------------------------------------------------------
# GET /sessions — 获取会话列表
# ----------------------------------------------------------
@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="获取会话列表",
    description="获取所有激活状态的聊天会话，按最后活跃时间倒序。"
)
def list_sessions(
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    offset = (page - 1) * page_size

    total = db.query(ChatSession).filter(ChatSession.is_active == True).count()

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.is_active == True)
        .order_by(ChatSession.updated_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return SessionListResponse(
        code=200,
        data=SessionListData(
            total=total,
            sessions=[SessionInfo.model_validate(s) for s in sessions],
        ),
    )


# ----------------------------------------------------------
# GET /sessions/{session_id}/messages — 获取会话消息
# ----------------------------------------------------------
@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessageListResponse,
    summary="获取会话消息",
    description="获取指定会话的所有聊天消息，按时间正序排列。"
)
def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return MessageListResponse(
        code=200,
        data=MessageListData(
            total=len(messages),
            session_id=session_id,
            messages=[
                MessageInfo(
                    id=msg.id,
                    session_id=msg.session_id,
                    question=msg.question,
                    answer=msg.answer,
                    sources=msg.source_chunks,
                    response_time=msg.response_time,
                    created_at=msg.created_at,
                )
                for msg in messages
            ],
        ),
    )


# ----------------------------------------------------------
# DELETE /sessions/{session_id} — 删除会话
# ----------------------------------------------------------
@router.delete(
    "/sessions/{session_id}",
    response_model=BaseResponse,
    summary="删除会话",
    description="软删除指定会话（标记为不活跃，消息保留）。"
)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    session.is_active = False
    db.commit()

    logger.info("会话已软删除: id={}", session_id)
    return BaseResponse(code=200, message="会话已删除")
