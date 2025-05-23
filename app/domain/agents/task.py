from typing import Type, Callable, Optional

from app.domain.agents.pydantic_ai_agent import PydanticAIAgent
from app.domain.tools.base import Tool
from app.domain.tools.report_tool import report_tool
from pydantic import BaseModel, ConfigDict, Field
import asyncio

from pydantic_ai.tools import Tool as PATool, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.result import Usage


SYSTEM_MESSAGE = """You are tasked with completing specific objectives and must report the outcomes. At your disposal, you have a variety of tools, each specialized in performing a distinct type of task.

For successful task completion:
Thought: Consider the task at hand and determine which tool is best suited based on its capabilities and the nature of the work. 
If you can complete the task or answer a question, soley by the information provided you can use the report_tool directly.

Use the report_tool with an instruction detailing the results of your work or to answer a user question.
If you encounter an issue and cannot complete the task:

Use the report_tool to communicate the challenge or reason for the task's incompletion.
You will receive feedback based on the outcomes of each tool's task execution or explanations for any tasks that couldn't be completed. This feedback loop is crucial for addressing and resolving any issues by strategically deploying the available tools.

On error: If information are missing consider if you can deduce or calculate the missing information and repeat the tool call with more arguments.

Use the information provided by the user to deduct the correct tool arguments.
Before using a tool think about the arguments and explain each input argument used in the tool. 
Return only one tool call at a time! Explain your thoughts!
{context}
"""


class EmptyArgModel(BaseModel):
    pass


class TaskAgent(BaseModel):
    name: str
    description: str
    arg_model: Type[BaseModel] = EmptyArgModel
    access_roles: list[str] = ["all"]

    create_context: Callable = None
    create_user_context: Callable = None
    tool_loader: Callable = None

    system_message: str = SYSTEM_MESSAGE
    tools: list[Tool]
    examples: list[dict] = None
    routing_example: list[dict] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_agent(self, **kwargs) -> PydanticAIAgent:

        input_kwargs = self.arg_model(**kwargs)
        kwargs = input_kwargs.dict()

        context = self.create_context(**kwargs) if self.create_context else None
        user_context = self.create_user_context(**kwargs) if self.create_user_context else None

        if self.tool_loader:
            self.tools.extend(self.tool_loader(**kwargs))

        if report_tool not in self.tools:
            self.tools.append(report_tool)

        return PydanticAIAgent(
            tools=self.tools,
            system_message=self.system_message,
            context=context,
            user_context=user_context,
            examples=self.examples,
        )

    @property
    def openai_tool_schema(self):
        def _dummy(arg: self.arg_model) -> str:  # type: ignore
            return ""

        pa_tool = PATool(_dummy)
        provider = OpenAIProvider(api_key="dummy")
        ctx = RunContext(deps=None, model=OpenAIModel("gpt-3.5-turbo", provider=provider), usage=Usage(), prompt=None)
        tool_def = asyncio.run(pa_tool.prepare_tool_def(ctx))
raw_func_params_schema = tool_def.parameters_json_schema
actual_model_schema = raw_func_params_schema.get("properties", {}).get("arg")

if not actual_model_schema:
    # Handle error or return empty schema if "arg" not found as expected
    final_parameters = {"type": "object", "properties": {}}
else:
    final_parameters = actual_model_schema.copy() # Use the schema of self.arg_model
    if final_parameters.get("required"):
        final_parameters.pop("required")

# The rest of the return structure remains the same, using final_parameters
return {
    "type": "function",
    "function": {
        "name": self.name,
        "description": self.description,
        "parameters": final_parameters,
    },
}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
