"""
文档管理路由

提供文档的上传、列表查询、详情查询、删除、统计等 API 接口。
上传接口会触发完整的文档处理流水线：解析 → 切分 → 向量化 → 入库。
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.document_service import DocumentService
from backend.schemas import (
    BaseResponse,
    DocumentUploadResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentListData,
    DocumentInfo,
)
from backend.utils.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["文档管理"])

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}


def _get_file_extension(filename: str) -> str:
    """提取文件扩展名（小写）"""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ----------------------------------------------------------
# POST /upload — 上传并处理文档
# ----------------------------------------------------------
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="上传文档",
    description="上传 PDF/TXT/MD 文件，自动执行: 解析 → 切分 → 向量化 → 入库。"
)
async def upload_document(
    file: UploadFile = File(..., description="支持 .pdf / .txt / .md 文件"),
    db: Session = Depends(get_db),
):
    # 1. 校验文件类型
    file_ext = _get_file_extension(file.filename)
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{file_ext}'，仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. 读取文件内容
    file_content = await file.read()
    file_size = len(file_content)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")

    logger.info("开始处理上传: filename='{}', size={}B", file.filename, file_size)

    service = DocumentService(db)

    try:
        # 3. 保存文件到磁盘
        file_path = service.save_uploaded_file(file.filename, file_content)

        # 4. 创建文档记录（status=processing）
        doc = service.create_document(
            filename=file.filename,
            file_type=file_ext,
            file_size=file_size,
            file_path=file_path,
        )

        # 5. 执行完整处理流水线: 解析 → 切分 → 向量化 → 入库
        doc = service.process_document(doc.id)

        # 重新加载以获取最新状态
        db.refresh(doc)

        return DocumentUploadResponse(
            code=200,
            message=f"文档上传并处理成功，共生成 {doc.chunk_count} 个文本块",
            data=DocumentInfo.model_validate(doc),
        )

    except Exception as e:
        logger.error("文档上传处理失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


# ----------------------------------------------------------
# GET / — 文档列表
# ----------------------------------------------------------
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="获取文档列表",
    description="分页查询已上传的文档列表，按上传时间倒序排列。"
)
def list_documents(
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    result = service.list_documents(page=page, page_size=page_size)

    return DocumentListResponse(
        code=200,
        data=DocumentListData(
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            documents=[
                DocumentInfo.model_validate(doc)
                for doc in result["documents"]
            ],
        ),
    )


# ----------------------------------------------------------
# GET /stats/overview — 文档统计（放在 {document_id} 路由前面）
# ----------------------------------------------------------
@router.get(
    "/stats/overview",
    summary="文档统计",
    description="获取知识库的文档统计信息。"
)
def get_document_stats(db: Session = Depends(get_db)):
    service = DocumentService(db)
    stats = service.get_statistics()
    return {"code": 200, "message": "success", "data": stats}


# ----------------------------------------------------------
# GET /{document_id} — 文档详情
# ----------------------------------------------------------
@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="获取文档详情",
)
def get_document(document_id: int, db: Session = Depends(get_db)):
    service = DocumentService(db)
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentDetailResponse(code=200, data=DocumentInfo.model_validate(doc))


# ----------------------------------------------------------
# DELETE /{document_id} — 删除文档
# ----------------------------------------------------------
@router.delete(
    "/{document_id}",
    response_model=BaseResponse,
    summary="删除文档",
    description="删除指定文档及其关联的文本块和本地文件。"
)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    service = DocumentService(db)
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    return BaseResponse(code=200, message="文档已删除")
