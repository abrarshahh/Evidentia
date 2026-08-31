import uuid
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from qdrant_client.models import PointStruct

from app.core.config import settings
from app.schemas.document_index import DocumentSkeletonIndex, DocumentNode
from app.storage.qdrant_client import qdrant_client, COLLECTION_NAME

logger = logging.getLogger(__name__)


class NodeEmbedder:
    """
    Chunk Vector Embedder & Qdrant Upserter.
    Generates 768-dimensional dense vector embeddings using Google Gemini's gemini-embedding-001
    and upserts point structs with payload metadata to Qdrant Cloud.
    """

    def __init__(self):
        self.gemini_key = settings.get_gemini_api_key()
        self.genai_client = None
        if self.gemini_key:
            self.genai_client = genai.Client(api_key=self.gemini_key)

    @staticmethod
    def generate_point_id(workspace_id: str, document_id: str, node_id: str) -> str:
        """
        Generate a deterministic UUIDv5 for a node to ensure idempotent upserts.
        """
        composite_key = f"evidentia:{workspace_id}:{document_id}:{node_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, composite_key))

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense 768-dim vector embeddings for a list of text strings using gemini-embedding-001.
        """
        if not self.genai_client:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        if not texts:
            return []

        embeddings: List[List[float]] = []
        batch_size = 20
        config = types.EmbedContentConfig(output_dimensionality=768)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=config,
                )
                if response and hasattr(response, "embeddings"):
                    for emb in response.embeddings:
                        embeddings.append(emb.values)
                else:
                    raise ValueError("No embeddings returned by gemini-embedding-001")
            except Exception as e:
                logger.warning(f"Embedding call with gemini-embedding-001 failed: {e}. Retrying with gemini-embedding-2...")
                try:
                    response = self.genai_client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=batch,
                        config=config,
                    )
                    if response and hasattr(response, "embeddings"):
                        for emb in response.embeddings:
                            embeddings.append(emb.values)
                except Exception as ex:
                    logger.error(f"Gemini embedding fallback failed: {ex}")
                    raise ex

        return embeddings

    async def upsert_document_index(
        self,
        workspace_id: str,
        index: DocumentSkeletonIndex,
    ) -> DocumentSkeletonIndex:
        """
        Embed all document nodes and upsert point vectors with payload metadata to Qdrant.
        Sets node.embedding_id on every processed node.
        """
        nodes_to_embed = [n for n in index.nodes if n.text and n.text.strip()]
        if not nodes_to_embed:
            logger.info("No text-bearing nodes found to embed.")
            return index

        logger.info(f"Generating embeddings for {len(nodes_to_embed)} nodes in document {index.document_id}...")
        texts = [n.text.strip() for n in nodes_to_embed]
        vectors = self.generate_embeddings(texts)

        points: List[PointStruct] = []
        for node, vector in zip(nodes_to_embed, vectors):
            point_id = self.generate_point_id(
                workspace_id=workspace_id,
                document_id=index.document_id,
                node_id=node.node_id,
            )

            payload = {
                "workspace_id": str(workspace_id),
                "document_id": str(index.document_id),
                "node_id": node.node_id,
                "node_type": node.node_type,
                "parent_section_id": node.parent_section_id,
                "page_range": node.page_range,
                "char_offsets": node.char_offsets,
                "text": node.text,
                "is_reference": False,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
            # Assign embedding reference ID on node
            node.embedding_id = f"qdrant:{point_id}"

        logger.info(f"Upserting {len(points)} vector points to Qdrant collection '{COLLECTION_NAME}'...")
        try:
            await qdrant_client.upsert_points(points)
        except Exception as e:
            if "dimension error" in str(e).lower():
                logger.warning("Re-initializing Qdrant collection to 768 dimensions...")
                qdrant_client.init_collection(vector_size=768, recreate=True)
                await qdrant_client.upsert_points(points)
            else:
                raise e

        logger.info(f"Qdrant vector upsert complete for document {index.document_id}.")
        return index


node_embedder = NodeEmbedder()
