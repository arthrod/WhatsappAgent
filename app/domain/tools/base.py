from typing import Type, Callable, Union

from pydantic import BaseModel, ConfigDict
import asyncio
from pydantic_ai.tools import Tool as PATool, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.result import Usage
from sqlmodel import SQLModel


class ToolResult(BaseModel):
    content: str
    success: bool


class Tool(BaseModel):
    name: str
    description: str = ""
    model: Union[Type[BaseModel], Type[SQLModel], None]
    function: Callable
    validate_missing: bool = True
    parse_model: bool = False
    exclude_keys: list[str] = ["id"]

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def run(self, **kwargs) -> ToolResult:
        missing_values = self.validate_input(**kwargs)
        if missing_values:
            content = f"Missing values: {', '.join(missing_values)}"
            return ToolResult(content=content, success=False)

        if self.parse_model:
            if hasattr(self.model, "model_validate"):
                input_ = self.model.model_validate(kwargs)
            else:
                input_ = self.model(**kwargs)
            result = self.function(input_)

        else:
            result = self.function(**kwargs)
        return ToolResult(content=str(result), success=True)

    def validate_input(self, **kwargs):
        if not self.validate_missing or not self.model:
            return []
        model_keys = set(self.model.__annotations__.keys()) - set(self.exclude_keys)
        input_keys = set(kwargs.keys())
        missing_values = model_keys - input_keys
        return list(missing_values)

    @property
    def openai_tool_schema(self):
        """Return the OpenAI compatible tool schema using pydantic-ai."""
        pa_tool = PATool(self.function)
        provider = OpenAIProvider(api_key="dummy")
        ctx = RunContext(deps=None, model=OpenAIModel("gpt-3.5-turbo", provider=provider), usage=Usage(), prompt=None)
        tool_def = asyncio.run(pa_tool.prepare_tool_def(ctx))
        parameters = tool_def.parameters_json_schema
        if parameters.get("required"):
            parameters.pop("required")
        parameters["properties"] = {
            key: value for key, value in parameters.get("properties", {}).items()
            if key not in self.exclude_keys
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ReportSchema(BaseModel):
    report: str


def report_function(report: ReportSchema) -> str:
    return report.report


report_tool = Tool(
    name="report_tool",
    model=ReportSchema,
    function=report_function,
    validate_missing=False,
    parse_model=True
)
