"""
文档处理服务

提供文档的完整生命周期管理：
- CRUD: 创建、查询、列表、删除文档元数据
- 文档处理流水线: 解析 → 切分 → 向量化 → 入库

路由层（routers/documents.py）调用此模块，不直接操作数据库或 AI 组件。
"""

import os
from typing import Optional
from sqlalchemy.orm import Session

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.models import Document, DocumentChunk
from backend.config import get_settings
from backend.utils.logger import logger
from backend.utils.file_parser import parse_file
from backend.services.vector_service import VectorService


class DocumentService:
    """文档 CRUD + 文档处理流水线"""

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    # ==============================================================
    # 文档处理流水线（核心链路）
    # ==============================================================

    def process_document(self, document_id: int) -> Document:
        """
        文档处理完整流水线：解析 → 切分 → 向量化 → 入库。

        流程图：
            上传文件 → parse_file() 解析为 Document 列表
                     → RecursiveCharacterTextSplitter 切分为小文本块
                     → VectorService.add_texts() 向量化并存入 FAISS
                     → 保存 DocumentChunk 到 MySQL
                     → 更新 Document 状态为 completed

        Args:
            document_id: 要处理的文档 ID（需已创建记录且文件已保存）

        Returns:
            处理完成的 Document 对象

        Raises:
            ValueError: 文档 ID 不存在
            Exception: 处理过程中的任何错误（文档状态会被标记为 failed）
        """
        doc = self.get_document(document_id)
        if not doc:
            raise ValueError(f"文档 id={document_id} 不存在")

        logger.info(
            "===== 开始处理文档: id={}, filename='{}' =====",
            doc.id, doc.filename,
        )

        try:
            # ---- 第 1 步: 解析文件 ----
            logger.info("[1/4] 解析文件...")
            raw_documents = parse_file(doc.file_path, doc.file_type)
            logger.info(
                "[1/4] 解析完成: {} 个文档片段",
                len(raw_documents),
            )

            # ---- 第 2 步: 文本切分 ----
            logger.info("[2/4] 切分文本...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            )
            chunks = text_splitter.split_documents(raw_documents)
            logger.info(
                "[2/4] 切分完成: {} 个文本块 (chunk_size={}, overlap={})",
                len(chunks), self.settings.chunk_size, self.settings.chunk_overlap,
            )

            if not chunks:
                raise ValueError("文档切分后无有效文本块，请检查文件内容")

            # ---- 第 3 步: 向量化 + 存入 FAISS ----
            logger.info("[3/4] 向量化并存入 FAISS...")
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {
                    "document_id": doc.id,
                    "document_name": doc.filename,
                    "chunk_index": i,
                    "page_number": chunk.metadata.get("page"),
                }
                for i, chunk in enumerate(chunks)
            ]

            vector_service = VectorService.get_instance()
            vector_ids = vector_service.add_texts(texts, metadatas)
            logger.info("[3/4] 向量化完成: {} 个向量已入库", len(vector_ids))

            # ---- 第 4 步: 保存文本块到 MySQL ----
            logger.info("[4/4] 保存文本块到数据库...")
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk.page_content,
                    page_number=chunk.metadata.get("page"),
                    vector_id=vector_ids[i] if i < len(vector_ids) else None,
                )
                self.db.add(db_chunk)

            # 更新文档状态
            doc.status = "completed"
            doc.chunk_count = len(chunks)
            doc.error_message = None
            self.db.commit()

            logger.info(
                "===== 文档处理完成: id={}, filename='{}', chunks={} =====",
                doc.id, doc.filename, len(chunks),
            )
            return doc

        except Exception as e:
            # 处理失败：标记状态，记录错误信息
            logger.error(
                "文档处理失败: id={}, error='{}'",
                document_id, str(e),
            )
            doc.status = "failed"
            doc.error_message = str(e)[:500]
            self.db.commit()
            raise

    # ==============================================================
    # 基础 CRUD 方法
    # ==============================================================

    def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        file_path: str,
    ) -> Document:
        """创建文档元数据记录（状态: processing）"""
        doc = Document(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            status="processing",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        logger.info(
            "文档记录已创建: id={}, filename='{}', size={}B",
            doc.id, doc.filename, doc.file_size,
        )
        return doc

    def get_document(self, document_id: int) -> Optional[Document]:
        """根据 ID 查询单个文档"""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_documents(self, page: int = 1, page_size: int = 10) -> dict:
        """分页查询文档列表（按创建时间倒序）"""
        offset = (page - 1) * page_size
        total = self.db.query(Document).count()
        documents = (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": documents,
        }

    def delete_document(self, document_id: int) -> bool:
        """
        删除文档（MySQL 记录 + 本地文件）。

        注意：FAISS 中的对应向量不会立即删除（V1 设计取舍），
        但检索时会通过 valid_document_ids 过滤掉陈旧结果。
        """
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False

        filename = doc.filename

        # 删除本地文件
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
                logger.info("本地文件已删除: {}", doc.file_path)
            except OSError as e:
                logger.error("删除本地文件失败: {}", str(e))

        # 删除数据库记录（级联删除 chunks）
        self.db.delete(doc)
        self.db.commit()
        logger.info("文档已删除: id={}, filename='{}'", document_id, filename)
        return True

    def save_uploaded_file(self, filename: str, content: bytes) -> str:
        """将上传文件保存到磁盘（同名文件自动加序号避免冲突）"""
        os.makedirs(self.settings.upload_dir, exist_ok=True)

        file_path = os.path.join(self.settings.upload_dir, filename)
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(file_path):
                file_path = os.path.join(
                    self.settings.upload_dir,
                    f"{name}_{counter}{ext}",
                )
                counter += 1

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("文件已保存: {} ({} bytes)", file_path, len(content))
        return file_path

    def get_document_chunks(self, document_id: int) -> list[DocumentChunk]:
        """获取某个文档的所有文本块（按序号排序）"""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def get_statistics(self) -> dict:
        """获取知识库统计信息"""
        from sqlalchemy import func

        total_docs = self.db.query(Document).count()
        total_chunks = self.db.query(DocumentChunk).count()

        status_counts = (
            self.db.query(Document.status, func.count(Document.id))
            .group_by(Document.status)
            .all()
        )
        status_map = dict(status_counts)

        # 获取向量库信息（安全读取，不触发初始化）
        try:
            if VectorService._instance is not None:
                total_vectors = VectorService._instance.total_vectors
            else:
                total_vectors = 0
        except Exception:
            total_vectors = 0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
            "completed": status_map.get("completed", 0),
            "processing": status_map.get("processing", 0),
            "failed": status_map.get("failed", 0),
        }
