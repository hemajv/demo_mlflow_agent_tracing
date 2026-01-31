"""Configuration settings for the expense agent."""

from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow", env_ignore_empty=True)

    # OpenAI
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for authenticating with the server")
    OPENAI_MODEL_NAME: Optional[str] = Field(None, description="Name of the model to use (e.g. `qwen3:8b`)")
    OPENAI_BASE_URL: Optional[str] = Field(None, description="Base URL of the server")

    # Embedding Server
    EMBEDDING_API_KEY: Optional[SecretStr] = Field(None, description="API key for authenticating with the server")
    EMBEDDING_MODEL_NAME: Optional[str] = Field(None, description="Name of the model to use (e.g. `nomic-embed-text`)")
    EMBEDDING_BASE_URL: Optional[str] = Field(None, description="Base URL of the server")
    #EMBEDDING_DOCUMENT_PREFIX: Optional[str] = Field("", description="Prefix for embeddings for documents")
    EMBEDDING_SEARCH_PREFIX: Optional[str] = Field("", description="Prefix for embeddings for search queries")

    # Chainlit
    CHAINLIT_AUTH_SECRET: Optional[SecretStr] = Field(
        None,
        description="Authorization secret used for signing tokens. Can be generated using `chainlit create-secret`",
    )

    # MLFlow
    MLFLOW_TRACKING_URI: Optional[str] = Field(None, description="MLFlow Tracking URI")
    MLFLOW_EXPERIMENT_NAME: Optional[str] = Field(None, description="MLFlow Experiment Name")
    MLFLOW_SYSTEM_PROMPT_URI: Optional[str] = Field(None, description="MLFlow Prompt URI (e.g. prompts:/my-prompt@latest)")
    EVAL_DATASET_NAME: Optional[str] = Field(None, description="MLflow evaluation dataset name (used by scripts/evals.py)")

    @property
    def openai_enabled(self) -> bool:
        """Check if required OpenAI environment variables are set."""
        return self.OPENAI_API_KEY is not None and self.OPENAI_MODEL_NAME is not None

    @property
    def auth_enabled(self) -> bool:
        """Check if required Keycloak environment variables are set."""
        return self.CHAINLIT_AUTH_SECRET is not None

    @property
    def embedding_server_enabled(self) -> bool:
        """Check if optional embedding server environment variables are set."""
        return self.EMBEDDING_API_KEY is not None and self.EMBEDDING_MODEL_NAME is not None

    @model_validator(mode="after")
    def llm(self) -> Self:
        """Validate that required environment variables are set for the LLM."""
        if not self.openai_enabled:
            raise Exception(
                "Missing required environment variables for LLM. Please ensure OPENAI_API_KEY and OPENAI_MODEL_NAME are in your environment or .env file."
            )
        return self

    @model_validator(mode="after")
    def auth(self) -> Self:
        """Validate that required environment variables are set for authorization."""
        if not self.auth_enabled:
            raise Exception(
                "Missing required environment variables for authentication. Please ensure CHAINLIT_AUTH_SECRET is in your environment or .env file."
            )
        return self
