"""Abstract base class for all IARA GenAI agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Base interface that every IARA agent must implement."""

    @abstractmethod
    def execute(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute a prompt and return the agent's response.

        Args:
            prompt (str): User prompt to send to the model.
            context (List[Dict[str, str]], optional): Prior conversation turns.

        Returns:
            Dict[str, Any]: Raw API response payload.
        """
        pass
