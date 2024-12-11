from datetime import datetime
from sqlmodel import Session
from app.persistance.db import engine
from app.persistance.models import Revenue, Expense, Customer, Invoice

# Mock data for Revenue
revenues = [
    {"id": 1, "description": "Website development", "net_amount": 1000.0, "gross_amount": 1190.0, "tax_rate": 0.19, "date": datetime(2024, 1, 15)},
    {"id": 2, "description": "Consulting services", "net_amount": 2000.0, "gross_amount": 2380.0, "tax_rate": 0.19, "date": datetime(2024, 2, 10)},
    {"id": 3, "description": "Annual subscription", "net_amount": 500.0, "gross_amount": 595.0, "tax_rate": 0.19, "date": datetime(2024, 3, 5)},
]

# Mock data for Expense
expenses = [
    {"id": 1, "description": "Office supplies", "net_amount": 300.0, "gross_amount": 357.0, "tax_rate": 0.19, "date": datetime(2024, 1, 20)},
    {"id": 2, "description": "Cloud hosting", "net_amount": 150.0, "gross_amount": 178.5, "tax_rate": 0.19, "date": datetime(2024, 2, 5)},
    {"id": 3, "description": "Marketing campaign", "net_amount": 1200.0, "gross_amount": 1428.0, "tax_rate": 0.19, "date": datetime(2024, 2, 28)},
]

# Mock data for Customer
customers = [
    {"id": 1, "company_name": "Tech Solutions Inc.", "first_name": "John", "last_name": "Doe", "phone": "123456789", "address": "123 Elm Street", "city": "Tech City", "zip": "45678", "country": "USA"},
    {"id": 2, "company_name": None, "first_name": "Jane", "last_name": "Smith", "phone": "987654321", "address": "456 Oak Street", "city": "Innovate Town", "zip": "78901", "country": "Canada"},
    {"id": 3, "company_name": "Future Ventures", "first_name": "Albert", "last_name": "Einstein", "phone": "555666777", "address": "789 Pine Avenue", "city": "Science City", "zip": "12345", "country": "Germany"},
]

# Mock data for Invoice
invoices = [
    {"id": 1, "customer_id": 1, "invoice_number": "INV-1001", "description": "Monthly retainer", "amount": 1190.0, "tax_rate": 0.19, "date": datetime(2024, 1, 31)},
    {"id": 2, "customer_id": 2, "invoice_number": "INV-1002", "description": "Project completion", "amount": 2380.0, "tax_rate": 0.19, "date": datetime(2024, 2, 15)},
    {"id": 3, "customer_id": 3, "invoice_number": "INV-1003", "description": "Software license", "amount": 595.0, "tax_rate": 0.19, "date": datetime(2024, 3, 10)},
]


with Session(engine) as session:
    # Insert Revenues
    for revenue in revenues:
        session.add(Revenue(**revenue))

    # Insert Expenses
    for expense in expenses:
        session.add(Expense(**expense))

    # Insert Customers
    for customer in customers:
        session.add(Customer(**customer))

    # Insert Invoices
    for invoice in invoices:
        session.add(Invoice(**invoice))

    session.commit()
