from typing import Type

from sqlmodel import Session

from app.persistance.db import engine
from app.persistance.models import *
from app.domain.tools.base import Tool


def add_row_to_table(model_instance: SQLModel):
    with Session(engine) as session:
        session.add(model_instance)
        session.commit()
        session.refresh(model_instance)
    return f"Successfully added {model_instance} to the table"


def add_entry_to_table(sql_model: Type[SQLModel]):
    # return a Callable that takes a SQLModel instance and adds it to the table
    return lambda **data: add_row_to_table(model_instance=sql_model.model_validate(data))


def create_add_data_sql_tool(
        model: Type[SQLModel],
        name: str = None,
        description: str = None,
        exclude_keys: list[str] = ["id"]
) -> Tool:
    return Tool(
        model=model,
        function=add_entry_to_table(model),
        name=name,
        description=description,
        exclude_keys=exclude_keys
    )

#
# add_expense_tool = create_add_data_sql_tool(Expense)
# add_revenue_tool = create_add_data_sql_tool(Revenue)
# add_time_tracking_tool = create_add_data_sql_tool(TimeTracking)
# add_employee_tool = create_add_data_sql_tool(Employee)
# add_customer_tool = create_add_data_sql_tool(Customer)
# add_invoice_tool = create_add_data_sql_tool(Invoice)

