"""IARA GenAI API client — multi-backend facade.

Public interface is UNCHANGED from the original implementation.

Internally the client delegates every call to a private backend adapter
chosen at construction time based on the ``provider`` argument:

  * ``provider in ('azure_openai', 'bedrock', 'vertex')``
    → :class:`_IaraGenAIBackend`  (original bank-internal path, iaragenai SDK)

  * ``provider == 'openai'``
    → :class:`_OpenAIBackend`  (new external path, openai SDK)

NEW (refactoring): Added multi-backend support via private adapters in
``_backends.py``.  Every existing public method and its signature are
preserved verbatim; no callers need to change any import or code.
"""

from typing import Any, Dict, Iterator, List, Optional

from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_logger import logger

# NEW: import private backend adapters — not exposed to public callers
from construct_cost_ai.infra.ai.frameworks.iara.src.utils._backends import (
    _IaraGenAIBackend,
    _OpenAIBackend,
)

# Providers that route through the iaragenai SDK (bank environment)
_IARAGENAI_PROVIDERS = frozenset({"azure_openai", "bedrock", "vertex"})


class IaraAPIClient:
    """Facade around multiple LLM backend providers.

    The public interface is identical to the original single-backend
    implementation.  Provider selection happens transparently inside
    :meth:`__init__` — callers need only supply an extra ``api_key``
    when ``provider='openai'``.

    Args:
        client_id (str): IARA OAuth client ID — required for iaragenai providers.
        client_secret (str): IARA OAuth client secret — required for iaragenai providers.
        access_token (str, optional): Pre-existing project access token.
        environment (str): Target environment ('dev' | 'homol' | 'prod').
        provider (str): LLM provider:
            ``'azure_openai'`` | ``'bedrock'`` | ``'vertex'`` → iaragenai SDK.
            ``'openai'`` → official openai SDK (external environments).
        api_key (str, optional): OpenAI API key — used only when
            ``provider='openai'``.  Falls back to the ``OPENAI_API_KEY``
            environment variable when omitted.  Ignored for other providers.

    NEW (refactoring): Added ``api_key`` parameter and internal backend
    selection.  All previously existing parameters are unchanged.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        environment: str = "dev",
        provider: str = "azure_openai",
        # NEW: OpenAI API key — ignored when using iaragenai providers
        api_key: Optional[str] = None,
    ) -> None:
        if provider in _IARAGENAI_PROVIDERS:
            # Original code path — delegates to iaragenai SDK (bank environment)
            self._backend = _IaraGenAIBackend(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                environment=environment,
                provider=provider,
            )
        elif provider == "openai":
            # NEW code path — delegates to official openai SDK (external environments)
            self._backend = _OpenAIBackend(api_key=api_key)
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported values: {sorted(_IARAGENAI_PROVIDERS | {'openai'})}"
            )

        logger.debug(f"IaraAPIClient ready — env={environment}, provider={provider}")

    # ------------------------------------------------------------------
    # All public methods below are UNCHANGED in signature and behaviour.
    # They delegate to self._backend which is selected in __init__.
    # ------------------------------------------------------------------

    # --------------------------------------------------------------
    # Models
    # --------------------------------------------------------------

    def list_models(self) -> List[Any]:
        """Return all models available in the current environment."""
        try:
            models = self._backend.list_models()
            logger.debug(f"Available models: {models}")
            return models
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            raise

    # --------------------------------------------------------------
    # Chat Completions
    # --------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4.1-mini",
        enable_polling: bool = True,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        **extra_kwargs: Any,
    ) -> Any:
        """Call the chat completions endpoint.

        Args:
            messages: Conversation messages in OpenAI format.
            model: Model identifier.
            enable_polling: Use long-polling for the response (iaragenai only).
            stream: When True returns a streaming iterator.
            response_format: e.g. ``{"type": "json_object"}``.
            **extra_kwargs: Extra kwargs forwarded to the SDK.
        """
        try:
            logger.debug(f"Chat completion — model={model}, stream={stream}")
            return self._backend.chat_completion(
                messages=messages,
                model=model,
                enable_polling=enable_polling,
                stream=stream,
                response_format=response_format,
                **extra_kwargs,
            )
        except Exception as exc:
            logger.error(f"Chat completion failed: {exc}")
            raise

    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4.1-mini",
        **extra_kwargs: Any,
    ) -> Iterator[str]:
        """Yield chat completion chunks as they arrive."""
        try:
            response = self.chat_completion(
                messages=messages,
                model=model,
                stream=True,
                **extra_kwargs,
            )

            for chunk in response:
                yield chunk

        except Exception as exc:
            logger.error(f"Streaming failed: {exc}")
            raise

    # --------------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------------

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
    ) -> Any:
        """Generate an embedding vector for the given text."""
        try:
            logger.debug(f"Creating embedding — model={model}")
            return self._backend.create_embedding(text=text, model=model)
        except Exception as exc:
            logger.error(f"Embedding creation failed: {exc}")
            raise
