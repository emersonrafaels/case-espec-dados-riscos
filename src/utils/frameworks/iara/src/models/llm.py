"""LLM configuration model for IARA GenAI."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class IaraLLMConfig:
    """Configuration for the IARA Language Model.

    Attributes:
        model (str): Model identifier (e.g. 'gpt-4.1-mini').
        temperature (float): Controls randomness (0.0-1.0).
        top_p (float): Nucleus sampling threshold.
        frequency_penalty (float): Penalises repeated tokens (0.0-1.0).
        presence_penalty (float): Penalises repeated topics (0.0-1.0).
        enable_polling (bool): Use polling for long-running completions.
        stream (bool): Enable token streaming.
        response_format (dict, optional): e.g. ``{"type": "json_object"}``.
    """

    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_polling: bool = True
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dictionary, omitting None values."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "enable_polling": self.enable_polling,
            "stream": self.stream,
        }

        if self.response_format is not None:
            payload["response_format"] = self.response_format

        return payload
