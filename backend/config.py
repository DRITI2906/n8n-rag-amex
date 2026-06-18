from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # Embedding
    EMBEDDING_MODEL: str = "huggingface"          # "huggingface" | "openai"
    HF_MODEL_NAME: str = "all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LLM
    LLM_PROVIDER: str = "ollama"                  # "ollama" | "anthropic" | "openai"
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

    # Chunking
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 100

    # Retrieval
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.45


@lru_cache
def get_settings() -> Settings:
    return Settings()
