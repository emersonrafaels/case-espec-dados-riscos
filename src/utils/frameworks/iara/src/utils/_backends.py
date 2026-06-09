"""Private backend adapters for IaraAPIClient.

Each backend class exposes the same internal interface so that
:class:`IaraAPIClient` can swap implementations transparently without
touching its public surface.

These classes are implementation details — they are NOT part of the
public API and should not be imported directly by client code.

NEW (refactoring): This module was added to enable multi-backend support
(iaragenai + openai) while keeping IaraAPIClient's public interface intact.
"""

import os
from typing import Any, Dict, Iterator, List, Optional

from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_logger import logger


# ---------------------------------------------------------------------------
# Internal contract (informal)
# ---------------------------------------------------------------------------
# Both backends must implement:
#   list_models() -> List[Any]
#   chat_completion(messages, model, enable_polling, stream,
#                   response_format, **extra_kwargs) -> Any
#   create_embedding(text, model) -> Any
# ---------------------------------------------------------------------------


class _IaraGenAIBackend:
    """Backend adapter that delegates to the ``iaragenai`` SDK.

    This is the original execution path used inside the bank environment.
    Behaviour is identical to the pre-refactoring IaraAPIClient — only
    moved here to allow sibling backends to co-exist.

    Args:
        client_id (str): IARA OAuth client ID.
        client_secret (str): IARA OAuth client secret.
        access_token (str, optional): Pre-existing project access token.
        environment (str): 'dev' | 'homol' | 'prod'.
        provider (str): 'azure_openai' | 'bedrock' | 'vertex'.

    Raises:
        ImportError: When the ``iaragenai`` package is not installed.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: Optional[str] = None,
        environment: str = "dev",
        provider: str = "azure_openai",
    ) -> None:
        try:
            from iaragenai import IaraGenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"The 'iaragenai' package is required for provider='{provider}' "
                "but is not installed. Run: pip install iaragenai"
            ) from exc

        self._client = IaraGenAI(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            environment=environment,
            provider=provider,
        )

        logger.debug(f"_IaraGenAIBackend ready — env={environment}, provider={provider}")

    # ------------------------------------------------------------------
    # Internal interface implementation
    # ------------------------------------------------------------------

    def list_models(self) -> List[Any]:
        return self._client.models.list()

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4.1-mini",
        enable_polling: bool = True,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        **extra_kwargs: Any,
    ) -> Any:
        params: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            **extra_kwargs,
        }

        if stream:
            params["stream"] = True
        else:
            params["enable_polling"] = enable_polling

        if response_format is not None:
            params["response_format"] = response_format

        return self._client.chat.completions.create(**params)

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
    ) -> Any:
        return self._client.embeddings.create(model=model, input=text)


class _OpenAIBackend:
    """Backend adapter that delegates to the official ``openai`` SDK.

    Used when ``provider='openai'``, enabling the framework to operate
    outside the bank environment without any changes to public interfaces
    or client code.

    The API key is resolved in this priority order:
      1. ``api_key`` constructor argument.
      2. ``OPENAI_API_KEY`` environment variable.

    Args:
        api_key (str, optional): OpenAI API key.
        base_url (str, optional): Custom base URL (for proxies / Azure OpenAI direct).

    Raises:
        ImportError: When the ``openai`` package is not installed.
        ValueError: When no API key can be resolved.

    NEW: Added as part of multi-backend refactoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for provider='openai' "
                "but is not installed. Run: pip install openai"
            ) from exc

        # Resolve key: explicit arg > env var
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required. Provide it via the 'api_key' "
                "parameter or set the OPENAI_API_KEY environment variable."
            )

        init_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            init_kwargs["base_url"] = base_url

        self._client = _openai.OpenAI(**init_kwargs)

        logger.debug("_OpenAIBackend ready")

    # ------------------------------------------------------------------
    # Internal interface implementation
    # ------------------------------------------------------------------

    def list_models(self) -> List[Any]:
        return list(self._client.models.list())

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4.1-mini",
        enable_polling: bool = True,  # NOTE: not applicable for OpenAI direct — silently ignored
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        **extra_kwargs: Any,
    ) -> Any:
        # Strip iaragenai-specific kwargs that the OpenAI SDK does not accept
        extra_kwargs.pop("enable_polling", None)

        # OpenAI chat.completions for newer models rejects ``max_tokens`` and
        # expects ``max_completion_tokens``. Normalize here to keep callers
        # backend-agnostic.
        if "max_tokens" in extra_kwargs and "max_completion_tokens" not in extra_kwargs:
            extra_kwargs["max_completion_tokens"] = extra_kwargs.pop("max_tokens")

        params: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": stream,
            **extra_kwargs,
        }

        if response_format is not None:
            params["response_format"] = response_format

        return self._client.chat.completions.create(**params)

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
    ) -> Any:
        return self._client.embeddings.create(model=model, input=text)
