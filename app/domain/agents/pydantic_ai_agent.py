from __future__ import annotations

from typing import Optional, List

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool as PydanticTool

from app.domain.tools.base import Tool


class PydanticAIAgent:
    """
    A wrapper around `pydantic_ai.Agent` that provides compatibility with the application's agent interface.
    
    This agent encapsulates the functionality of pydantic-ai, handling tool conversion,
    context management, and execution of user requests through the underlying LLM.
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        system_message: str,
        model_name: str = "openai:gpt-3.5-turbo-0125",
        context: str | None = None,
        user_context: str | None = None,
        examples: list[dict] | None = None,
    ) -> None:
        """
        Initialize the PydanticAIAgent.
        
        Args:
            tools: List of tools available to the agent
            system_message: Instructions for the agent
            model_name: The LLM model to use
            context: Additional context information 
            user_context: Context related to the user
            examples: Example interactions for few-shot learning
        """
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

    def run(self, user_input: str, context: str | None = None) -> str:
        """
        Run the agent synchronously with the provided user input and optional context.
        
        Args:
            user_input: The user's request or message
            context: Optional additional context for this specific request
            
        Returns:
            str: The agent's response
            
        Raises:
            Exception: If agent execution fails
        """
        try:
            if self.user_context:
                context = context if context else self.user_context
            if context:
                user_input = f"{context}\n---\n\nUser Message: {user_input}"
            logger.debug(f"Running agent with input: {user_input[:100]}...")
            result = self.agent.run_sync(user_input)
            return result.output
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise
    
    async def run_async(self, user_input: str, context: str | None = None) -> str:
        """
        Run the agent asynchronously with the provided user input and optional context.
        
        Args:
            user_input: The user's request or message
            context: Optional additional context for this specific request
            
        Returns:
            str: The agent's response
            
        Raises:
            Exception: If agent execution fails
        """
        try:
            if self.user_context:
                context = context if context else self.user_context
            if context:
                user_input = f"{context}\n---\n\nUser Message: {user_input}"
            logger.debug(f"Running agent asynchronously with input: {user_input[:100]}...")
            result = await self.agent.run(user_input)
            return result.output
        except Exception as e:
            logger.error(f"Error running agent asynchronously: {e}")
            raise
