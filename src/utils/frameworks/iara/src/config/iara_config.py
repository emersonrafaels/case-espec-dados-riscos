"""Configuration builder for IARA GenAI settings."""

import os
from functools import lru_cache

from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_dynaconf import get_settings


@lru_cache()
def get_iara_config(
    client_id: str = None,
    client_secret: str = None,
    access_token: str = None,
    environment: str = None,
    provider: str = None,
    model: str = None,
    # NEW (refactoring): optional OpenAI API key — ignored for iaragenai providers.
    # Resolved in priority order: argument > OPENAI_API_KEY env var > settings.toml.
    api_key: str = None,
) -> dict:
    """
    Build and return IARA GenAI configuration dictionary.

    Merges settings.toml defaults with any runtime overrides.
    All parameters are optional — omitted values fall back to settings.toml.

    Args:
        client_id (str, optional): IARA OAuth client ID.
        client_secret (str, optional): IARA OAuth client secret.
        access_token (str, optional): Pre-existing project access token.
        environment (str, optional): 'dev' | 'homol' | 'prod'.
        provider (str, optional): 'azure_openai' | 'bedrock' | 'vertex' | 'openai'.
        model (str, optional): Model name override (e.g. 'gpt-4.1-mini').
        api_key (str, optional): OpenAI API key — used when provider='openai'.
            NEW (refactoring): falls back to OPENAI_API_KEY env var or settings.toml.

    Returns:
        dict: Resolved configuration dictionary.
    """

    settings = get_settings()

    # NEW: resolve OpenAI API key — argument > env var > settings.toml
    resolved_api_key = (
        api_key or os.environ.get("OPENAI_API_KEY") or settings.get("openai.api_key", None) or None
    )

    resolved_provider = provider or settings.get("iara.provider", "azure_openai")

    # NEW: when provider is 'openai', fall back to openai-specific model defaults
    if resolved_provider == "openai":
        default_chat_model = settings.get("openai.default_chat", "gpt-4.1-mini")
        default_embedding_model = settings.get("openai.default_embedding", "text-embedding-ada-002")
    else:
        default_chat_model = settings.get("iara.models.default_chat", "gpt-4.1-mini")
        default_embedding_model = settings.get(
            "iara.models.default_embedding", "text-embedding-ada-002"
        )

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "access_token": access_token,
        "environment": environment or settings.get("iara.environment", "dev"),
        "provider": resolved_provider,
        # NEW: resolved OpenAI API key (None for iaragenai providers)
        "api_key": resolved_api_key,
        "model": model or default_chat_model,
        "temperature": settings.get("iara.models.default_temperature", 0.7),
        "top_p": settings.get("iara.models.default_top_p", 1.0),
        "frequency_penalty": settings.get(
            "iara.models.default_frequency_penalty",
            0.0,
        ),
        "presence_penalty": settings.get(
            "iara.models.default_presence_penalty",
            0.0,
        ),
        "enable_polling": settings.get(
            "iara.models.enable_polling",
            True,
        ),
        "stream": settings.get(
            "iara.models.stream",
            False,
        ),
        "default_embedding_model": default_embedding_model,
        "response_format": settings.get(
            "iara.chat.response_format",
            "text",
        ),
    }
