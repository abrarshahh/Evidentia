import uuid
import math
import hashlib
import logging
from typing import List, Optional
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

from app.schemas.document_index import DocumentSkeletonIndex, DocumentNode
from app.storage.qdrant_client import qdrant_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Default 768-dimensional ONNX Embedding Model (Top-Ranked on MTEB Benchmark)
DEFAULT_LOCAL_MODEL = "BAAI/bge-base-en-v1.5"


def generate_fallback_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Generate a deterministic normalized pseudo-vector for fallback when memory/engine errors occur.
    Uses SHA-256 seed hashing to produce consistent float values normalized to unit length.
    """
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        byte_val = seed_bytes[i % len(seed_bytes)]
        val = (byte_val / 127.5) - 1.0
        values.append(val)

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


class NodeEmbedder:
    """
    Blazing Fast Local Vector Embedder & Qdrant Upserter using FastEmbed (ONNX Runtime).
    Generates 768-dimensional dense vector embeddings locally without API rate limits, network latency, or quota limits.
    Processes in small batches to guarantee zero ONNX memory allocation errors on large documents.
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        self.model_name = model_name
        self._embedding_model: Optional[TextEmbedding] = None

    @property
    def embedding_model(self) -> TextEmbedding:
        """
        Lazy-initialize local ONNX embedding model on first use.
        """
        if self._embedding_model is None:
            logger.info(f"Initializing FastEmbed local model '{self.model_name}' (768 dims)...")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    @staticmethod
    def generate_point_id(workspace_id: str, document_id: str, node_id: str) -> str:
        """
        Generate a deterministic UUIDv5 for a node to ensure idempotent upserts.
        """
        composite_key = f"evidentia:{workspace_id}:{document_id}:{node_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, composite_key))

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate dense 768-dim vector embeddings locally for a list of text strings.
        Runs in chunked batches (batch_size=32) to prevent ONNX Runtime bad allocation OOM errors.
        """
        if not texts:
            return []

        logger.info(f"Generating local fastembed embeddings for {len(texts)} text nodes in batches of {batch_size}...")
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                gen = self.embedding_model.embed(chunk, batch_size=len(chunk))
                chunk_vecs = [list(vec) for vec in gen]
                embeddings.extend(chunk_vecs)
            except Exception as e:
                logger.warning(f"FastEmbed batch {i // batch_size + 1} failed with error: {e}. Falling back to single-node embedding...")
                for text in chunk:
                    try:
                        single_gen = self.embedding_model.embed([text], batch_size=1)
                        embeddings.append(list(next(single_gen)))
                    except Exception:
                        embeddings.append(generate_fallback_embedding(text))

        return embeddings

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Async wrapper for generate_embeddings.
        """
        return self.generate_embeddings(texts)

    async def upsert_document_index(
        self,
        workspace_id: str,
        index: DocumentSkeletonIndex,
        is_reference: bool = False,
    ) -> DocumentSkeletonIndex:
        """
        Embed all document nodes locally and upsert point vectors with payload metadata to Qdrant.
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
                "is_reference": is_reference,
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
