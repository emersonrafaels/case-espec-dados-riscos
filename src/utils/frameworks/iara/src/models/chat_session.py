"""Chat session model for IARA GenAI framework."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class IaraMessage:
    """A single chat message exchanged with an IARA agent."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IaraChatSession:
    """Maintains conversation history for a single IARA session."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[IaraMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Append a new message to the conversation history."""
        self.messages.append(IaraMessage(role=role, content=content))

    def get_context(self) -> List[Dict[str, str]]:
        """Return the conversation history in IARA API format."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self) -> None:
        """Clear all messages while preserving the session ID."""
        self.messages.clear()
