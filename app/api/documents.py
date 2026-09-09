import io
import uuid
import json
import hashlib
import logging
from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.session import get_async_session
from app.db.models import (
    Document, DocumentIndex, WorkspaceMember, DocumentStatus, EnrichmentStatus,
    AnalysisReference, ReferenceMapping, Verification, Analysis
)
from app.schemas.documents import DocumentResponse
from app.schemas.workspaces import WorkspaceRole
from app.schemas.document_index import DocumentSkeletonIndex
from app.schemas.tasks import TaskResponse
from app.auth.dependencies import verify_workspace_access, require_workspace_role
from app.storage.minio_client import minio_client
from app.storage.qdrant_client import qdrant_client
from app.indexing.structural_parser import StructuralDocumentParser
from app.indexing.enrichment import enrichment_engine
from app.indexing.embedder import node_embedder
from workers.tasks import create_task_record, run_document_indexing_task, get_task_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    workspace_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN, WorkspaceRole.EDITOR])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Upload a document (PDF/TXT), stream raw file to MinIO, record pending document
    in PostgreSQL, and immediately dispatch full indexing pipeline to background worker.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = uuid.uuid4()
    minio_key = f"raw/{workspace_id}/{document_id}/{file.filename}"
    checksum = hashlib.sha256(file_bytes).hexdigest()

    # 1. Stream Raw Upload to MinIO
    try:
        minio_client.init_buckets()
        minio_client.client.put_object(
            bucket_name=minio_client.RAW_DOCUMENTS_BUCKET,
            object_name=minio_key,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logger.error(f"MinIO upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document to MinIO: {e}")

    # 2. Record pending document entry in PostgreSQL
    doc_record = Document(
        id=document_id,
        workspace_id=workspace_id,
        original_filename=file.filename,
        checksum=checksum,
        minio_raw_path=minio_key,
        status=DocumentStatus.pending,
        created_at=datetime.utcnow(),
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    # 3. Register and dispatch background indexing task
    task_id = str(uuid.uuid4())
    create_task_record(task_id, "document_indexing", str(document_id), str(workspace_id))
    background_tasks.add_task(run_document_indexing_task, task_id, document_id, workspace_id)

    logger.info(f"Document {document_id} upload recorded. Dispatched background task {task_id}.")
    return doc_record


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_indexing_task_status(
    workspace_id: uuid.UUID,
    task_id: str,
    membership: WorkspaceMember = Depends(verify_workspace_access),
):
    """
    Query current status of a background document indexing task.
    """
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    workspace_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all documents uploaded to a workspace.
    """
    query = select(Document).where(Document.workspace_id == workspace_id)
    res = await db.execute(query)
    docs = res.scalars().all()
    return docs


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed information about a specific uploaded document.
    """
    query = select(Document).where(
        Document.workspace_id == workspace_id,
        Document.id == document_id,
    )
    res = await db.execute(query)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.get("/{document_id}/index", response_model=DocumentSkeletonIndex)
async def get_document_index(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Retrieve the complete DocumentSkeletonIndex JSON for a document in a workspace.
    """
    doc_query = select(Document).where(
        Document.workspace_id == workspace_id,
        Document.id == document_id,
    )
    doc_res = await db.execute(doc_query)
    if not doc_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found in workspace.")

    index_minio_key = f"indexes/{workspace_id}/{document_id}/skeleton.json"
    try:
        minio_res = minio_client.client.get_object(
            bucket_name=minio_client.INDEXES_BUCKET,
            object_name=index_minio_key,
        )
        index_bytes = minio_res.read()
        return json.loads(index_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Document index skeleton not found.")


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN, WorkspaceRole.EDITOR])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a document and completely purge all of its associated data across:
    1. Qdrant Vector DB (vector embeddings)
    2. MinIO Object Storage (raw uploads & index skeleton JSONs)
    3. PostgreSQL Database (document record, indexes, reference mappings, verifications, and analyses)
    """
    # 1. Retrieve Document Record
    query = select(Document).where(
        Document.workspace_id == workspace_id,
        Document.id == document_id,
    )
    res = await db.execute(query)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    raw_path = doc.minio_raw_path

    # 2. Delete Vector Embeddings from Qdrant Cloud
    try:
        await qdrant_client.delete_document_points(document_id=str(document_id))
    except Exception as e:
        logger.error(f"Error purging Qdrant points for document {document_id}: {e}")

    # 3. Delete Objects from MinIO Storage
    try:
        minio_client.remove_document_objects(
            workspace_id=str(workspace_id),
            document_id=str(document_id),
            minio_raw_path=raw_path,
        )
    except Exception as e:
        logger.error(f"Error purging MinIO objects for document {document_id}: {e}")

    # 4. Purge Dependent Records & Document from PostgreSQL
    try:
        # Delete AnalysisReferences
        await db.execute(delete(AnalysisReference).where(AnalysisReference.document_id == document_id))
        # Delete ReferenceMappings
        await db.execute(delete(ReferenceMapping).where(ReferenceMapping.document_id == document_id))
        # Delete Verifications
        await db.execute(delete(Verification).where(Verification.document_id == document_id))
        # Delete Analyses where base_document_id is this document
        await db.execute(delete(Analysis).where(Analysis.base_document_id == document_id))
        # Delete Document (cascades to DocumentIndex)
        await db.delete(doc)

        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete document {document_id} from PostgreSQL database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document from database: {e}")

    logger.info(f"Document {document_id} and all related data purged successfully from workspace {workspace_id}.")
    return {
        "detail": "Document and all associated data successfully deleted.",
        "document_id": str(document_id),
        "workspace_id": str(workspace_id),
    }
