import logging
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType,
    Filter, FieldCondition, MatchValue, PointStruct, ScoredPoint, FilterSelector
)

from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "evidentia_chunks"


class QdrantStorage:
    COLLECTION_NAME = COLLECTION_NAME

    def __init__(self):
        api_key = settings.get_qdrant_api_key()

        # Synchronous client for collection initialization with 60s timeout
        self.sync_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
            timeout=60.0,
        )
        # Asynchronous client for API execution with 60s timeout
        self.async_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
            timeout=60.0,
        )

    def init_collection(self, vector_size: int = 768, recreate: bool = False) -> None:
        """
        Initialize the shared evidentia_chunks collection and create payload indexes.
        """
        collections = [c.name for c in self.sync_client.get_collections().collections]
        if recreate and self.COLLECTION_NAME in collections:
            self.sync_client.delete_collection(collection_name=self.COLLECTION_NAME)
            collections.remove(self.COLLECTION_NAME)

        if self.COLLECTION_NAME not in collections:
            self.sync_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        # Create payload keyword indexes for fast filtering
        try:
            self.sync_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="workspace_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.sync_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.sync_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="is_reference",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Index already exists

    async def upsert_points(self, points: List[PointStruct], batch_size: int = 100) -> None:
        """
        Upsert vector points into the evidentia_chunks collection in batch chunks
        to avoid Qdrant Cloud HTTP payload timeouts.
        """
        if not points:
            return

        logger.info(f"Upserting {len(points)} vector points to Qdrant in batches of {batch_size}...")
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.async_client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch,
            )

    async def search_chunks(
        self,
        workspace_id: str,
        query_vector: List[float],
        document_id: Optional[str] = None,
        is_reference: Optional[bool] = None,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        """
        Query top matching chunks within a specific workspace.
        """
        must_filters = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=str(workspace_id)),
            )
        ]

        if document_id:
            must_filters.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            )

        if is_reference is not None:
            must_filters.append(
                FieldCondition(
                    key="is_reference",
                    match=MatchValue(value=is_reference),
                )
            )

        query_filter = Filter(must=must_filters)

        response = await self.async_client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

    async def delete_document_points(self, document_id: str) -> None:
        """
        Delete all vector points in Qdrant associated with a specific document_id.
        """
        try:
            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            )
            res = await self.async_client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=FilterSelector(filter=filter_cond),
            )
            logger.info(f"Deleted vector points for document {document_id} from Qdrant: {res}")
        except Exception as e:
            logger.error(f"Failed to delete vector points for document {document_id} from Qdrant: {e}")

    async def delete_workspace_points(self, workspace_id: str) -> None:
        """
        Delete all vector points in Qdrant associated with a specific workspace_id.
        """
        try:
            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="workspace_id",
                        match=MatchValue(value=str(workspace_id)),
                    )
                ]
            )
            res = await self.async_client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=FilterSelector(filter=filter_cond),
            )
            logger.info(f"Deleted vector points for workspace {workspace_id} from Qdrant: {res}")
        except Exception as e:
            logger.error(f"Failed to delete vector points for workspace {workspace_id} from Qdrant: {e}")


qdrant_client = QdrantStorage()
