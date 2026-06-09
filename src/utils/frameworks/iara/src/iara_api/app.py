"""IARA GenAI application entry-point."""

from typing import Any, Dict, List, Optional

from construct_cost_ai.infra.ai.frameworks.iara.src.agents.chat_agent import IaraAgentChat
from construct_cost_ai.infra.ai.frameworks.iara.src.config.config_logger import logger
from construct_cost_ai.infra.ai.frameworks.iara.src.models.chat_session import IaraChatSession
from construct_cost_ai.infra.ai.frameworks.iara.src.models.llm import IaraLLMConfig


def chat(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    access_token: Optional[str] = None,
    environment: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    question: str = "",
    system_prompt: Optional[str] = None,
    stream: bool = False,
    llm_config: Optional[IaraLLMConfig] = None,
    # NEW (refactoring): OpenAI API key — passed through to IaraAgentChat
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single chat turn against an IARA GenAI agent.

    All credential/environment parameters are optional — omitted values
    are resolved from ``settings.toml``.

    Returns:
        Dict with keys ``session``, ``response``, and ``message``.
    """
    session: Optional[IaraChatSession] = None
    answer: Optional[Dict[str, Any]] = None
    message: str = ""

    try:
        logger.info("IARA GenAI Chat — carregando configurações...")

        resolved_llm_config = llm_config
        if model and not llm_config:
            resolved_llm_config = IaraLLMConfig(model=model)

        session = IaraChatSession()

        agent = IaraAgentChat(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            environment=environment,
            provider=provider,
            llm_config=resolved_llm_config,
            system_prompt=system_prompt,
            # NEW: forward api_key for openai provider
            api_key=api_key,
        )

        session.add_message(role="user", content=question)

        if stream:
            chunks: List[str] = []

            for chunk in agent.ask(
                question=question,
                context=session.get_context(),
                stream=True,
            ):
                print(chunk, end="", flush=True)
                chunks.append(chunk)

            print()

            message = "".join(chunks)
            answer = {"message": message, "raw": None}

        else:
            answer = agent.ask(
                question=question,
                context=session.get_context(),
                stream=False,
            )
            message = answer.get("message", "")

        session.add_message(role="assistant", content=message)

        return {
            "session": session,
            "response": answer,
            "message": message,
        }

    except Exception as exc:
        logger.error(f"Erro no chat IARA: {exc}")
        return {
            "session": session,
            "response": answer,
            "message": message,
        }
