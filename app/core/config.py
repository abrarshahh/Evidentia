from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str
    ENV: str
    DEBUG: bool
    PORT: int

    # Security & Auth (loaded from .env)
    SECRET_KEY: SecretStr
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int

    # Database Settings
    POSTGRES_SERVER: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_DB: str
    DATABASE_URL: str

    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_URL: str

    # MinIO Object Storage
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: SecretStr
    MINIO_SECURE: bool

    # Qdrant Vector Database
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[SecretStr] = None

    # LLM Providers
    GEMINI_API_KEY: Optional[SecretStr] = None

    # Secret getter helpers for SDKs
    def get_secret_key(self) -> str:
        return self.SECRET_KEY.get_secret_value()

    def get_postgres_password(self) -> str:
        return self.POSTGRES_PASSWORD.get_secret_value()

    def get_minio_secret_key(self) -> str:
        return self.MINIO_SECRET_KEY.get_secret_value()

    def get_qdrant_api_key(self) -> Optional[str]:
        val = getattr(self, "QDRANT_API_KEY", None)
        return val.get_secret_value() if val else None

    def get_gemini_api_key(self) -> Optional[str]:
        val = getattr(self, "GEMINI_API_KEY", None)
        return val.get_secret_value() if val else None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
