"""
FAISS 向量服务

管理 FAISS 向量库的完整生命周期：
- 初始化 / 加载已有索引
- 添加文档向量
- 相似度检索
- 持久化到磁盘

使用单例模式，确保整个应用共享同一个 FAISS 实例。
通过线程锁保证并发安全。

设计说明：
    FAISS 不原生支持按 ID 删除向量（IndexFlatL2 无 remove_ids 方法），
    因此当文档被删除时，其向量仍保留在 FAISS 中。
    检索时通过比对 MySQL 中的有效文档 ID 来过滤陈旧结果。
    这是 V1 的设计取舍，对于小规模数据完全可行。
"""

import os
import uuid
import threading
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config import get_settings
from backend.utils.logger import logger


class VectorService:
    """FAISS 向量库管理（线程安全的单例）"""

    _instance: Optional["VectorService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VectorService":
        """
        获取 VectorService 单例。

        第一次调用时初始化 Embedding 模型和加载已有索引。
        后续调用直接返回已有实例。
        """
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定，防止并发时重复创建
                if cls._instance is None:
                    cls._instance = VectorService()
        return cls._instance

    def __init__(self):
        """初始化 Embedding 模型和 FAISS 索引"""
        settings = get_settings()

        self.embeddings = OpenAIEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model_name,
            chunk_size=25, # 兼容 DashScope (限制单批次最大 25)
            check_embedding_ctx_length=False, # 必须设为 False，否则 Langchain 底层在检查长度时会改变数据结构导致 DashScope 报错
        )
        logger.info(
            "Embedding 模型已初始化: model='{}', base_url='{}'",
            settings.embedding_model_name, settings.embedding_base_url,
        )

        # FAISS 相关配置
        self.store_dir = settings.vector_store_dir
        self.vector_store: Optional[FAISS] = None
        self._write_lock = threading.Lock()

        # 尝试加载已有索引
        self._load_index()

    # ----------------------------------------------------------
    # 加载已有索引
    # ----------------------------------------------------------
    def _load_index(self) -> None:
        """从磁盘加载已有 FAISS 索引（如果存在）"""
        index_path = os.path.join(self.store_dir, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.store_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # 信任本地数据
                )
                doc_count = self.vector_store.index.ntotal
                logger.info("FAISS 索引已从磁盘加载: {} 个向量", doc_count)
            except Exception as e:
                logger.error("FAISS 索引加载失败: {}，将创建新索引", str(e))
                self.vector_store = None
        else:
            logger.info("未找到已有 FAISS 索引，等待首次文档上传后创建")

    # ----------------------------------------------------------
    # 保存索引到磁盘
    # ----------------------------------------------------------
    def _save_index(self) -> None:
        """将 FAISS 索引持久化到磁盘"""
        if self.vector_store is not None:
            os.makedirs(self.store_dir, exist_ok=True)
            self.vector_store.save_local(self.store_dir)
            logger.info("FAISS 索引已保存到: {}", self.store_dir)

    # ----------------------------------------------------------
    # 添加文档向量
    # ----------------------------------------------------------
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict],
    ) -> list[str]:
        """
        将文本列表向量化并添加到 FAISS 索引。

        Args:
            texts: 文本内容列表
            metadatas: 每条文本对应的元数据列表，结构示例:
                {
                    "document_id": 1,
                    "document_name": "公司制度.pdf",
                    "chunk_index": 0,
                    "page_number": 1
                }

        Returns:
            向量 ID 列表（UUID 字符串），用于存入 DocumentChunk.vector_id
        """
        if not texts:
            logger.warning("add_texts 收到空文本列表，跳过")
            return []

        # 过滤掉空字符串，防止 DashScope 报 InvalidParameter 错误
        filtered_texts = []
        filtered_metadatas = []
        for t, m in zip(texts, metadatas):
            clean_t = t.strip()
            if clean_t:
                filtered_texts.append(clean_t)
                filtered_metadatas.append(m)
        texts = filtered_texts
        metadatas = filtered_metadatas

        if not texts:
            logger.warning("过滤修剪后文本列表为空，跳过")
            return []

        # 为每条文本生成唯一 ID
        ids = [str(uuid.uuid4()) for _ in texts]

        logger.info(
            "正在向量化 {} 条文本（来自文档: '{}'）...",
            len(texts),
            metadatas[0].get("document_name", "未知") if metadatas else "未知",
        )

        with self._write_lock:
            if self.vector_store is None:
                # 首次创建索引
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info("FAISS 索引已创建，首批 {} 个向量", len(texts))
            else:
                # 追加到已有索引
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(
                    "已追加 {} 个向量，索引总计 {} 个向量",
                    len(texts), self.vector_store.index.ntotal,
                )

            # 持久化
            self._save_index()

        return ids

    # ----------------------------------------------------------
    # 相似度检索
    # ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 4,
        valid_document_ids: Optional[set[int]] = None,
    ) -> list[dict]:
        """
        根据查询文本进行相似度检索。

        Args:
            query: 用户查询文本
            top_k: 返回最相关的 k 条结果
            valid_document_ids: 有效文档 ID 集合（用于过滤已删除文档的陈旧向量）

        Returns:
            检索结果列表，每项包含:
            {
                "content": str,          # 文本块内容
                "document_name": str,    # 来源文档名
                "document_id": int,      # 文档 ID
                "chunk_index": int,      # 块序号
                "page_number": int|None, # 页码
                "relevance_score": float # 相关度分数 (0~1, 越大越相关)
            }
        """
        if self.vector_store is None:
            logger.warning("FAISS 索引为空，无法检索")
            return []

        # 多取一些结果，便于过滤后仍有足够数量
        fetch_k = top_k * 3 if valid_document_ids is not None else top_k

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(fetch_k, self.vector_store.index.ntotal),  # 不超过总数
            )
        except Exception as e:
            logger.error("FAISS 检索失败: {}", str(e))
            return []

        # 格式化结果
        formatted = []
        for doc, distance in results:
            doc_id = doc.metadata.get("document_id")

            # 过滤已删除文档的陈旧向量
            if valid_document_ids is not None and doc_id not in valid_document_ids:
                continue

            formatted.append({
                "content": doc.page_content,
                "document_name": doc.metadata.get("document_name", "未知"),
                "document_id": doc_id,
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "page_number": doc.metadata.get("page_number"),
                # L2距离 → 相关度分数：1/(1+d)，越大越相关
                "relevance_score": round(1.0 / (1.0 + float(distance)), 4),
            })

            # 收集够 top_k 就停止
            if len(formatted) >= top_k:
                break

        logger.info(
            "检索完成: query='{}...', 返回 {} 条结果",
            query[:30], len(formatted),
        )
        return formatted

    # ----------------------------------------------------------
    # 索引状态
    # ----------------------------------------------------------
    @property
    def total_vectors(self) -> int:
        """当前索引中的向量总数"""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal

    @property
    def is_ready(self) -> bool:
        """索引是否可用"""
        return self.vector_store is not None and self.vector_store.index.ntotal > 0
