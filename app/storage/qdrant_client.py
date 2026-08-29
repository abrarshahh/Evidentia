from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType,
    Filter, FieldCondition, MatchValue, PointStruct, ScoredPoint
)

from app.core.config import settings


class QdrantStorage:
    COLLECTION_NAME = "evidentia_chunks"

    def __init__(self):
        api_key = settings.get_qdrant_api_key()

        # Synchronous client for collection initialization
        self.sync_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
        )
        # Asynchronous client for API endpoint execution
        self.async_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
        )

    def init_collection(self, vector_size: int = 1536) -> None:
        """
        Initialize the shared evidentia_chunks collection and create payload indexes.
        """
        collections = [c.name for c in self.sync_client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.sync_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        # Create payload keyword indexes for fast filtering
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

    async def upsert_points(self, points: List[PointStruct]) -> None:
        """
        Upsert vector points into Qdrant collection.
        """
        await self.async_client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    async def search_document(
        self,
        workspace_id: str,
        document_id: str,
        query_vector: List[float],
        limit: int = 5,
        is_reference: Optional[bool] = None,
    ) -> List[ScoredPoint]:
        """
        Perform vector similarity search strictly scoped to workspace_id and document_id.
        """
        must_conditions = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=str(workspace_id)),
            ),
            FieldCondition(
                key="document_id",
                match=MatchValue(value=str(document_id)),
            ),
        ]

        if is_reference is not None:
            must_conditions.append(
                FieldCondition(
                    key="is_reference",
                    match=MatchValue(value=is_reference),
                )
            )

        query_filter = Filter(must=must_conditions)

        response = await self.async_client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points


qdrant_client = QdrantStorage()
