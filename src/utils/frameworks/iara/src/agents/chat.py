"""Simple chat interface for IARA agents."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from construct_cost_ai.infra.ai.frameworks.iara.src.agents.iara_agent import IaraAgent
from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_logger import logger
from construct_cost_ai.infra.ai.frameworks.iara.src.models.llm import IaraLLMConfig


class IaraAgentChat(IaraAgent):
    """High-level chat interface built on top of :class:`IaraAgent`.

    Provides a single :meth:`ask` method that hides low-level message
    building and returns a clean dict or streaming iterator.

    Inherits all construction parameters from :class:`IaraAgent`.
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
        # NEW (refactoring): OpenAI API key — passed through to IaraAgent
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            environment=environment,
            provider=provider,
            llm_config=llm_config,
            system_prompt=system_prompt,
            # NEW: forward api_key to parent
            api_key=api_key,
        )

    def ask(
        self,
        question: str,
        context: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[str]]:
        """Send a question to the IARA agent.

        Args:
            question (str): The user's question or instruction.
            context (list, optional): Prior conversation turns.
            stream (bool): When True, returns a streaming iterator of text chunks.

        Returns:
            Union[Dict[str, Any], Iterator[str]]:
                - Non-streaming: ``{"message": str, "raw": response}``
                - Streaming: iterator yielding text chunks
        """
        try:
            if stream:
                return self.stream(prompt=question, context=context)
            return self.execute(prompt=question, context=context)

        except Exception as exc:
            logger.error(f"IaraAgentChat.ask failed: {exc}")
            return {"message": "", "raw": None}

    def ask_with_files(
        self,
        question: str,
        attachments: List[Union[str, Path]],
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send a question alongside one or more file attachments.

        Wraps :meth:`IaraAgent.execute_multimodal` with the same
        clean interface as :meth:`ask`.

        Args:
            question (str): The user's question or instruction.
            attachments (list): Local file paths to attach (images or PDFs).
            context (list, optional): Prior conversation turns.

        Returns:
            Dict[str, Any]: ``{"message": str, "raw": response}``
        """
        try:
            return self.execute_multimodal(
                prompt=question,
                attachments=attachments,
                context=context,
            )

        except Exception as exc:
            logger.error(f"IaraAgentChat.ask_with_files failed: {exc}")
            return {"message": "", "raw": None}
