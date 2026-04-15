"""
聊天服务 — RAG 检索问答

核心流程：
1. 接收用户问题
2. 从 FAISS 检索最相关的文档块
3. 将检索结果作为上下文，构建 Prompt
4. 调用 LLM 生成回答
5. 保存聊天记录到 MySQL
6. 返回回答 + 引用来源

使用 LangChain LCEL（LangChain Expression Language）构建 Chain，
代码风格清晰透明，初学者也能看懂每一步在做什么。
"""

import time
from typing import Optional
from sqlalchemy.orm import Session

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import get_settings
from backend.models import ChatSession, ChatMessage, Document
from backend.services.vector_service import VectorService
from backend.utils.logger import logger


# ============================================================
# RAG 系统提示词
# ============================================================
RAG_SYSTEM_PROMPT = """你是"企业知识库 Copilot"，一个专业的企业内部知识问答助手。

你的任务：根据下方【参考文档】中检索到的内容，准确回答用户的问题。

规则：
1. **基于文档回答**：仅根据【参考文档】中的内容回答，不要凭空编造信息。
2. **诚实告知**：如果参考文档中没有足够信息来回答问题，请诚实地说："根据现有知识库内容，暂时无法回答此问题。建议上传相关文档后重试。"
3. **引用原文**：回答时尽量引用文档中的关键内容，增强可信度。
4. **格式清晰**：使用清晰、专业、易懂的中文回答，适当使用列表和分点格式。
5. **简洁有力**：回答要有条理，避免冗余重复。

【参考文档】
{context}"""


class ChatService:
    """RAG 检索问答服务"""

    def __init__(self, db: Session):
        """
        初始化聊天服务。

        Args:
            db: 数据库会话（由 FastAPI 依赖注入提供）
        """
        self.db = db
        settings = get_settings()

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,    # 适度创造性
            max_tokens=2000,    # 限制回答长度
        )
        logger.info(
            "LLM 已初始化: model='{}', base_url='{}'",
            settings.llm_model_name, settings.openai_base_url,
        )

        # 获取向量服务单例
        self.vector_service = VectorService.get_instance()

    # ----------------------------------------------------------
    # 核心问答方法
    # ----------------------------------------------------------
    def ask(
        self,
        question: str,
        session_id: Optional[int] = None,
        top_k: int = 4,
    ) -> dict:
        """
        RAG 问答：检索 → 构建上下文 → LLM 生成回答。

        Args:
            question: 用户问题
            session_id: 会话 ID（为空则自动创建新会话）
            top_k: 检索返回的最相关文档块数量

        Returns:
            {
                "answer": str,        # AI 回答
                "sources": list,      # 引用来源列表
                "session_id": int,    # 会话 ID
                "message_id": int,    # 消息 ID
            }
        """
        start_time = time.time()
        logger.info("收到问题: '{}'", question[:80])

        # ---- 步骤 1: 获取或创建会话 ----
        session = self._get_or_create_session(session_id, question)

        # ---- 步骤 2: 从 FAISS 检索相关文档块 ----
        valid_doc_ids = self._get_valid_document_ids()
        search_results = self.vector_service.search(
            query=question,
            top_k=top_k,
            valid_document_ids=valid_doc_ids,
        )

        logger.info("检索到 {} 条相关文档块", len(search_results))

        # ---- 步骤 3: 构建上下文和来源信息 ----
        context, sources = self._build_context(search_results)

        # ---- 步骤 4: 调用 LLM 生成回答 ----
        answer = self._generate_answer(question, context)

        response_time = round(time.time() - start_time, 2)
        logger.info("回答生成完成，耗时 {}s", response_time)

        # ---- 步骤 5: 保存消息记录 ----
        message = self._save_message(
            session=session,
            question=question,
            answer=answer,
            sources=sources,
            response_time=response_time,
        )

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session.id,
            "message_id": message.id,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _get_or_create_session(
        self,
        session_id: Optional[int],
        question: str,
    ) -> ChatSession:
        """获取已有会话或创建新会话"""
        if session_id:
            session = self.db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.is_active == True,
            ).first()
            if not session:
                raise ValueError(f"会话 {session_id} 不存在或已关闭")
            return session

        # 创建新会话，标题取问题的前 50 个字符
        title = question[:50] + ("..." if len(question) > 50 else "")
        session = ChatSession(title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info("创建新会话: id={}, title='{}'", session.id, session.title)
        return session

    def _get_valid_document_ids(self) -> set[int]:
        """获取所有已完成处理的文档 ID，用于过滤陈旧向量"""
        docs = self.db.query(Document.id).filter(
            Document.status == "completed"
        ).all()
        return {doc.id for doc in docs}

    def _build_context(
        self,
        search_results: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        将检索结果构建为 LLM 上下文和引用来源列表。

        Args:
            search_results: vector_service.search() 的返回结果

        Returns:
            (context_str, sources_list)
        """
        if not search_results:
            return "未找到相关文档内容。", []

        context_parts = []
        sources = []

        for i, result in enumerate(search_results, 1):
            # 构建上下文片段（带编号，便于 LLM 引用）
            page_info = f"(第{result['page_number']}页)" if result.get("page_number") is not None else ""
            context_parts.append(
                f"[文档{i}] 来源: {result['document_name']}{page_info}\n"
                f"{result['content']}"
            )

            # 构建引用来源信息
            sources.append({
                "document_name": result["document_name"],
                "chunk_content": result["content"][:300],  # 截取前300字作为摘要
                "page": result.get("page_number"),
                "relevance_score": result["relevance_score"],
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _generate_answer(self, question: str, context: str) -> str:
        """
        调用 LLM 生成回答。

        使用 LCEL (LangChain Expression Language) 构建 Chain:
        Prompt → LLM → 字符串输出
        """
        # 构建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # 构建 Chain: prompt → llm → 解析为字符串
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": context,
                "question": question,
            })
            return answer
        except Exception as e:
            logger.error("LLM 调用失败: {}", str(e))
            return f"抱歉，AI 回答生成失败：{str(e)}。请检查 API Key 配置是否正确。"

    def _save_message(
        self,
        session: ChatSession,
        question: str,
        answer: str,
        sources: list[dict],
        response_time: float,
    ) -> ChatMessage:
        """保存聊天消息到数据库"""
        message = ChatMessage(
            session_id=session.id,
            question=question,
            answer=answer,
            source_chunks=sources,
            response_time=response_time,
        )
        self.db.add(message)

        # 更新会话消息计数
        session.message_count += 1

        self.db.commit()
        self.db.refresh(message)

        logger.info(
            "消息已保存: session_id={}, message_id={}, response_time={}s",
            session.id, message.id, response_time,
        )
        return message
