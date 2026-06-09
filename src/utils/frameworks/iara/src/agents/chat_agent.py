"""Compatibility re-export for IaraAgentChat.

This module exists so that both of the following imports work identically:

    from .chat import IaraAgentChat          # original module name
    from .chat_agent import IaraAgentChat    # canonical public module name

NEW (refactoring): Added as a thin re-export to stabilise the public import
path without renaming or duplicating the existing chat.py implementation.
"""

from construct_cost_ai.infra.ai.frameworks.iara.src.agents.chat import IaraAgentChat  # noqa: F401

__all__ = ["IaraAgentChat"]
