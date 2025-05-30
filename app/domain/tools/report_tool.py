from pydantic import BaseModel

from app.domain.tools.base import Tool


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
