from typing import Optional

from pydantic import BeforeValidator, model_validator
from sqlmodel import SQLModel, Field
from datetime import time, datetime
from typing_extensions import Annotated


# === validator ===
def validate_date(v):
    if isinstance(v, datetime):
        return v

    for f in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
        try:
            return datetime.strptime(v, f)
        except ValueError:
            pass

    raise ValueError("Invalid date format")


def validate_time(v):
    if isinstance(v, time):
        return v

    if isinstance(v, str):
        try:
            return time.fromisoformat(v)
        except ValueError:
            raise ValueError("Invalid time format")


def numeric_validator(v):
    if isinstance(v, int):
        return float(v)
    elif isinstance(v, float):
        return v
    raise ValueError("Value must be a number")


DateFormat = Annotated[datetime, BeforeValidator(validate_date)]
TimeFormat = Annotated[time, BeforeValidator(validate_time)]
Numeric = Annotated[float, BeforeValidator(numeric_validator)]


# ==== Models ===


class Revenue(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    description: str
    net_amount: Numeric
    gross_amount: Numeric
    tax_rate: Numeric
    date: DateFormat

    @model_validator(mode="before")
    @classmethod
    def check_net_gross(cls, data: any):
        if isinstance(data, dict):
            if "net_amount" in data and "tax_rate" in data:
                data["gross_amount"] = round(data["net_amount"] * (1 + data["tax_rate"]), 2)
            elif "gross_amount" in data and "tax_rate" in data:
                data["net_amount"] = round(data["gross_amount"] / (1 + data["tax_rate"]), 2)
            elif "net_amount" in data and "gross_amount" in data:
                data["tax_rate"] = round((data["gross_amount"] - data["net_amount"]) / data["net_amount"], 2)

        return data


class Expense(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    description: str
    net_amount: Numeric = Field(description="The net amount of the expense (before tax)")
    gross_amount: Numeric
    tax_rate: Numeric
    date: DateFormat

    @model_validator(mode="before")
    @classmethod
    def check_net_gross(cls, data: any):
        if isinstance(data, dict):
            if "net_amount" in data and "tax_rate" in data:
                data["gross_amount"] = round(data["net_amount"] * (1 + data["tax_rate"]), 2)
            elif "gross_amount" in data and "tax_rate" in data:
                data["net_amount"] = round(data["gross_amount"] / (1 + data["tax_rate"]), 2)
            elif "net_amount" in data and "gross_amount" in data:
                data["tax_rate"] = round((data["gross_amount"] - data["net_amount"]) / data["net_amount"], 2)

        return data


class Customer(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    company_name: Optional[str] = None
    first_name: str
    last_name: str
    phone: str
    address: str
    city: str
    zip: str
    country: str


class Invoice(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    customer_id: Optional[int] = Field(default=None, foreign_key="customer.id")
    invoice_number: str
    description: str
    amount: Numeric
    tax_rate: Numeric
    date: DateFormat
