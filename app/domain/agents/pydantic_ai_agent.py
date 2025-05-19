from __future__ import annotations

from typing import Optional, List

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool as PydanticTool

from app.domain.tools.base import Tool


class PydanticAIAgent:
    """Simple wrapper around `pydantic_ai.Agent` to mirror the existing interface."""

    def __init__(
        self,
        tools: List[Tool],
        *,
        system_message: str,
        model_name: str = "openai:gpt-3.5-turbo-0125",
        context: Optional[str] = None,
        user_context: Optional[str] = None,
        examples: Optional[List[dict]] = None,
    ) -> None:
        self.tools = tools
        self.system_message = system_message
        self.model_name = model_name
        self.context = context
        self.user_context = user_context
        self.examples = examples or []
        self.agent = self._create_agent()

    def _create_agent(self) -> PydanticAgent:
        pydantic_tools = [
            PydanticTool(t.function, name=t.name, description=t.description)
            for t in self.tools
        ]
        instructions = self.system_message.format(context=self.context or "")
        return PydanticAgent(
            self.model_name,
            tools=pydantic_tools,
            instructions=instructions,
        )

    def run(self, user_input: str, context: Optional[str] = None) -> str:
        if self.user_context:
            context = context if context else self.user_context
        if context:
            user_input = f"{context}\n---\n\nUser Message: {user_input}"
        result = self.agent.run_sync(user_input)
        return result.output
