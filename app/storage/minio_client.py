import io
import json
from datetime import timedelta
from typing import Any, Dict, Optional
from minio import Minio
from minio.error import S3Error

from app.core.config import settings


class MinIOStorage:
    RAW_DOCUMENTS_BUCKET = "raw-documents"
    INDEXES_BUCKET = "indexes"
    LOGS_BUCKET = "logs"
    EXPORTS_BUCKET = "exports"

    def __init__(self):
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.get_minio_secret_key(),
            secure=settings.MINIO_SECURE,
        )

    def init_buckets(self) -> None:
        """
        Ensure all required Evidentia buckets exist on MinIO.
        """
        buckets = [
            self.RAW_DOCUMENTS_BUCKET,
            self.INDEXES_BUCKET,
            self.LOGS_BUCKET,
            self.EXPORTS_BUCKET,
        ]
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
            except S3Error as e:
                print(f"Error checking/creating bucket '{bucket}': {e}")

    def put_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload raw byte data to a MinIO bucket.
        """
        data_stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )
        return f"{bucket_name}/{object_name}"

    def put_json(self, bucket_name: str, object_name: str, data: Dict[str, Any]) -> str:
        """
        Serialize and upload a Python dictionary as a JSON object.
        """
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self.put_bytes(
            bucket_name=bucket_name,
            object_name=object_name,
            data=json_bytes,
            content_type="application/json",
        )

    def get_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """
        Download object content as raw bytes.
        """
        response = self.client.get_object(bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_json(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """
        Download and parse a JSON object.
        """
        content = self.get_bytes(bucket_name, object_name)
        return json.loads(content.decode("utf-8"))

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        """
        Remove an object from MinIO.
        """
        self.client.remove_object(bucket_name, object_name)

    def get_presigned_url(
        self, bucket_name: str, object_name: str, expires_minutes: int = 60
    ) -> str:
        """
        Generate a presigned GET URL for secure temporary client downloads.
        """
        return self.client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=object_name,
            expires=timedelta(minutes=expires_minutes),
        )

    def remove_document_objects(
        self,
        workspace_id: str,
        document_id: str,
        minio_raw_path: Optional[str] = None,
    ) -> None:
        """
        Purge all raw files and skeleton index JSON artifacts associated with document_id from MinIO.
        """
        # Delete specific raw path if provided
        if minio_raw_path:
            try:
                self.client.remove_object(self.RAW_DOCUMENTS_BUCKET, minio_raw_path)
            except Exception:
                pass

        # Purge all objects under raw/{workspace_id}/{document_id}/
        try:
            raw_objects = self.client.list_objects(
                self.RAW_DOCUMENTS_BUCKET,
                prefix=f"raw/{workspace_id}/{document_id}/",
                recursive=True,
            )
            for obj in raw_objects:
                self.client.remove_object(self.RAW_DOCUMENTS_BUCKET, obj.object_name)
        except Exception:
            pass

        # Delete skeleton index JSON object
        index_key = f"indexes/{workspace_id}/{document_id}/skeleton.json"
        try:
            self.client.remove_object(self.INDEXES_BUCKET, index_key)
        except Exception:
            pass

        # Purge all objects under indexes/{workspace_id}/{document_id}/
        try:
            idx_objects = self.client.list_objects(
                self.INDEXES_BUCKET,
                prefix=f"indexes/{workspace_id}/{document_id}/",
                recursive=True,
            )
            for obj in idx_objects:
                self.client.remove_object(self.INDEXES_BUCKET, obj.object_name)
        except Exception:
            pass

    def remove_workspace_objects(self, workspace_id: str) -> None:
        """
        Purge all raw documents, indexes, exports, and log files associated with workspace_id from MinIO.
        """
        buckets_prefixes = [
            (self.RAW_DOCUMENTS_BUCKET, f"raw/{workspace_id}/"),
            (self.INDEXES_BUCKET, f"indexes/{workspace_id}/"),
            (self.EXPORTS_BUCKET, f"exports/{workspace_id}/"),
            (self.LOGS_BUCKET, f"logs/{workspace_id}/"),
        ]

        for bucket, prefix in buckets_prefixes:
            try:
                objs = self.client.list_objects(bucket, prefix=prefix, recursive=True)
                for obj in objs:
                    self.client.remove_object(bucket, obj.object_name)
            except Exception:
                pass


minio_client = MinIOStorage()
