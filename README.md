
## Open AI Agent with Routing Agent for Business Databases

This repository contains a demonstration of an Open AI Agent with a Routing Agent designed to manage business-related databases. The project is written in Python and uses the Pydantic library for data validation and settings management using Python type annotations.

### Project Structure

The project is structured as follows:

```plaintext
project-root/

└── app/
    ├── domain/
    │   ├── agents/
    │   │   ├── __init__.py
    │   │   ├── base.py            # Base class for agents
    │   │   ├── routing.py         # Routing Agent for selecting task agents
    │   │   ├── tasks.py           # Task Agent for handling tasks
    │   │   ├── demo_agent.py      # Demo agent with tools
    │   │   └── utils.py           # Agent utilities
    │   ├── tools/
    │   │   ├── __init__.py
    │   │   ├── base.py            # Base class for tools
    │   │   ├── add.py             # Tool for adding data to the database
    │   │   ├── query.py           # Tool for querying data from the database
    │   │   ├── report_tool.py     # Tool for generating reports
    │   │   └── utils.py           # Tool utilities
    │   │
    │   ├── __init__.py
    │   ├── message_service.py     # Main logic for handling messages
    │   └── exceptions.py          # Custom exceptions
    │
    ├── infrastructure/
    │   ├── __init__.py
    │   └── llm.py                  # openai client
    │
    ├── persistance/
    │   ├── __init__.py
    │   ├── db.py
    │   ├── mock_data.py            # Script for setting up the database with mock data
    │   └── models.py
    │
    ├── __init__.py
    ├── main.py
    └── schema.py
```

### Setup

Install requirements using poetry
    
```bash
poetry install
```


To use this code, you need to add a `.env` file to the root directory of the project. This file should contain your OpenAI API key, like so:

```env
OPENAI_API_KEY=your-api-key-here
```

Replace `your-api-key-here` with your actual OpenAI API key.

## Setup Database

To setup the database and fill it with some data, run the following script:


```bash
python app/persistance/mock_data.py
```


## Test The AI Agent without WhatsApp Webhook

To test the AI agent without the WhatsApp webhook, you can simply run the message_service.py script. 
```bash
python app/domain/message_service.py
```

You should see the agent in the terminal executing the following request:

`START: Starting Routing Agent with Input:
'''What are my expenses'''`

When usiong the mockdata it should return the something like the following:
`Message: Here are your expenses:
1. Expense: Office supplies
   - Net Amount: $300.00
   - Gross Amount: $357.00
   - Tax Rate: 19%
   - Date: 2024-01-20

2. Expense: Cloud hosting
   - Net Amount: $150.00
   - Gross Amount: $178.50
   - Tax Rate: 19%
   - Date: 2024-02-05

3. Expense: Marketing campaign
   - Net Amount: $1200.00
   - Gross Amount: $1428.00
   - Tax Rate: 19%
   - Date: 2024-02-28

If you need more information or have any other requests, feel free to let me know!`


### Usage

To run the code, open the `app/main.py` and run the script. This will start the server and you can interact with the AI agent using the command line.


Please ensure these are installed before running the code.

### Contributing

Contributions are welcome. Please open an issue to discuss your ideas before making a pull request.

### License

This project is licensed under the terms of the MIT license.