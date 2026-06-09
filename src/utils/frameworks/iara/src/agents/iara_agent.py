"""Core IARA GenAI agent — chat completion with full parameter control."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from construct_cost_ai.infra.ai.frameworks.iara.src.agents.base_agent import BaseAgent
from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_logger import logger
from construct_cost_ai.infra.ai.frameworks.iara.src.config.iara_config import get_iara_config
from construct_cost_ai.infra.ai.frameworks.iara.src.models.llm import IaraLLMConfig
from construct_cost_ai.infra.ai.frameworks.iara.src.utils.api_client import IaraAPIClient


class IaraAgent(BaseAgent):
    """IARA GenAI agent with full control over model and chat parameters.

    All constructor arguments are optional — omitted values are resolved
    from `settings.toml` via :func:`get_iara_config`.

    Args:
        client_id (str, optional): IARA OAuth client ID.
        client_secret (str, optional): IARA OAuth client secret.
        access_token (str, optional): Pre-existing project access token.
        environment (str, optional): 'dev' | 'homol' | 'prod'.
        provider (str, optional): 'azure_openai' | 'bedrock' | 'vertex' | 'openai'.
        llm_config (IaraLLMConfig, optional): Full model configuration override.
        system_prompt (str, optional): System instruction prepended to every conversation.
        api_key (str, optional): OpenAI API key — used when provider='openai'.
            NEW (refactoring): falls back to OPENAI_API_KEY env var when omitted.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        environment: Optional[str] = None,
        provider: Optional[str] = None,
        llm_config: Optional[IaraLLMConfig] = None,
        system_prompt: Optional[str] = None,
        # NEW (refactoring): OpenAI API key — forwarded to IaraAPIClient
        api_key: Optional[str] = None,
    ) -> None:

        config = get_iara_config(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            environment=environment,
            provider=provider,
            # NEW: forward api_key so config can resolve it alongside env vars
            api_key=api_key,
        )

        # Fix: was `self.llm_config = IaraLLMConfig = ...` which shadowed the
        # imported class name and caused UnboundLocalError when llm_config is None.
        self.llm_config = llm_config or IaraLLMConfig(
            model=config["model"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            enable_polling=config["enable_polling"],
            stream=config["stream"],
        )
        self.system_prompt: Optional[str] = system_prompt

        # When provider is "openai", client_id and client_secret are not required —
        # the OpenAI backend only needs an api_key. Pass None for both so that
        # _OpenAIBackend (which ignores them) doesn't receive stale placeholder values.
        resolved_provider = config["provider"]
        self.api_client = IaraAPIClient(
            client_id=config["client_id"] if resolved_provider != "openai" else None,
            client_secret=config["client_secret"] if resolved_provider != "openai" else None,
            access_token=config["access_token"] if resolved_provider != "openai" else None,
            environment=config["environment"],
            provider=resolved_provider,
            api_key=config["api_key"],
        )

        logger.info(
            f"IARA Agent initialized with model '{self.llm_config.model}' in environment '{config['environment']}' using provider '{config['provider']}'."
        )

    """
        Internal Helpers for Message Building and File Handling
    """

    # Supported MIME types that IARA Vision accepts
    _SUPPORTED_MIME_TYPES = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp",
    }

    def _build_messages(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble the full message list for the API call."""
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": prompt})

        return messages

    def _encode_file_as_data_url(self, file_path: Union[str, Path]) -> str:
        """Encode a local file as a base64 Data URL.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            str: Data URL string in the form `data:<mime>;base64,<data>`.

        Raises:
            ValueError: When the MIME type is not supported by IARA.
            FileNotFoundError: When the file does not exist.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime_type, _ = mimetypes.guess_type(str(path))

        # fallback manual
        _ext_fallback = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        if mime_type is None:
            mime_type = _ext_fallback.get(path.suffix.lower())

        if mime_type not in self._SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type '{mime_type}' for file '{path.name}'. "
                f"Supported types: {sorted(self._SUPPORTED_MIME_TYPES)}"
            )

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _build_multimodal_content(
        self,
        prompt: str,
        attachments: List[Union[str, Path]],
    ) -> List[Dict[str, Any]]:
        """Build an OpenAI-style multipart content block.

        Args:
            prompt: Text question / instruction for the model.
            attachments: List of local file paths (images or PDFs).

        Returns:
            List of content parts ready to embed in a ``user`` message.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for file_path in attachments:
            path = Path(file_path)
            data_url = self._encode_file_as_data_url(file_path)
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                mime_type = {
                    ".pdf": "application/pdf",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(path.suffix.lower(), "")
            if mime_type == "application/pdf":
                # OpenAI chat completions uses {"type": "file"} for PDFs,
                # not {"type": "image_url"}
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": path.name,
                            "file_data": data_url,
                        },
                    }
                )
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
            logger.debug(f"Attached file: {path.name} ({mime_type})")

        return content

    def _build_messages_multimodal(
        self,
        prompt: str,
        attachments: List[Union[str, Path]],
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble the message list for a multimodal (vision) API call."""
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if context:
            messages.extend(context)

        multimodal_content = self._build_multimodal_content(prompt, attachments)

        messages.append(
            {
                "role": "user",
                "content": multimodal_content,
            }
        )

        return messages

    """
    
        BaseAgent Interface Implementation
    
    """

    def execute(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Send a prompt and return ``{"message": str, "raw": response}``."""
        try:
            logger.info(f"Executing prompt: {prompt[:80]}...")

            messages = self._build_messages(prompt, context)
            cfg = self.llm_config

            response = self.api_client.chat_completion(
                messages=messages,
                model=cfg.model,
                enable_polling=cfg.enable_polling,
                stream=False,
                response_format=cfg.response_format,
            )

            text = response.choices[0].message.content

            logger.success("Prompt executed successfully")

            return {"message": text, "raw": response}

        except Exception as exc:
            logger.error(f"Agent execution failed: {exc}")
            raise

    def execute_multimodal(
        self,
        prompt: str,
        attachments: List[Union[str, Path]],
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send a prompt together with one or more file attachments (vision / document)."""
        try:
            logger.info(
                f"Executing multimodal prompt with {len(attachments)} attachment(s): "
                f"{prompt[:80]}..."
            )

            messages = self._build_messages_multimodal(prompt, attachments, context)
            cfg = self.llm_config

            response = self.api_client.chat_completion(
                messages=messages,
                model=cfg.model,
                enable_polling=cfg.enable_polling,
                stream=False,
                response_format=cfg.response_format,
            )

            text = response.choices[0].message.content
            logger.success("Multimodal prompt executed successfully")

            return {"message": text, "raw": response}

        except Exception as exc:
            logger.error(f"Agent execution failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Extended capabilities
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Yield completion chunks for the given prompt (streaming mode)."""
        try:
            logger.info(f"Streaming prompt: {prompt[:80]}...")

            messages = self._build_messages(prompt, context)

            yield from self.api_client.stream_chat_completion(
                messages=messages,
                model=self.llm_config.model,
            )

        except Exception as exc:
            logger.error(f"Streaming failed: {exc}")
            raise

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> Any:
        """Generate an embedding vector for the given text."""
        try:
            embedding_model = model or get_iara_config()["default_embedding_model"]

            logger.info(f"Creating embedding — model={embedding_model}")

            return self.api_client.create_embedding(
                text=text,
                model=embedding_model,
            )

        except Exception as exc:
            logger.error(f"Embedding failed: {exc}")
            raise

    def list_models(self) -> List[Any]:
        """Return available models for the current environment."""
        try:
            return self.api_client.list_models()

        except Exception as exc:
            logger.error(f"Model listing failed: {exc}")
            raise
