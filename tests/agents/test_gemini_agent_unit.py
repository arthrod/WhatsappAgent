import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from google_labs_html_editing.agents.gemini_agent import GeminiAgent
from google_labs_html_editing.agents.tool_utils import Tool
# Assuming ToolResult is here, adjust if not.
from google_labs_html_editing.domain.tools.base import ToolResult 
from google.generativeai import client as genai
from google.generativeai import types as genai_types
from google.generativeai import generative_models

# It seems like the genai.Client is not directly available, let's try to use generative_models.GenerativeModel
# and patch that instead. Or maybe, generative_models.ChatSession if that's what's being used.
# For now, let's assume genai.Client is the correct path and adjust if necessary.

@pytest.fixture
def mock_genai_client():
    with patch('google.generativeai.client.Client') as mock_client_constructor:
        mock_client_instance = MagicMock()
        mock_client_constructor.return_value = mock_client_instance
        yield mock_client_instance

def test_gemini_agent_init_without_tools(mock_genai_client):
    """Tests GeminiAgent initialization without tools."""
    agent = GeminiAgent(model_name="gemini-pro")
    assert agent.model_name == "gemini-pro"
    assert agent.client == mock_genai_client
    assert agent.tools == []
    assert agent.session is None

def test_gemini_agent_init_with_tools(mock_genai_client):
    """Tests GeminiAgent initialization with tools."""
    mock_tool1 = Tool(name="tool1", description="description1", function=lambda x: x)
    mock_tool2 = Tool(name="tool2", description="description2", function=lambda y: y*2)
    tools = [mock_tool1, mock_tool2]
    
    agent = GeminiAgent(model_name="gemini-ultra", tools=tools)
    assert agent.model_name == "gemini-ultra"
    assert agent.client == mock_genai_client
    assert agent.tools == tools
    assert agent.session is None

@pytest.mark.asyncio
async def test_gemini_agent_connect_to_live_server_success(mock_genai_client):
    """Tests successful connection to live server."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Mock the live connect method
    mock_connect = AsyncMock()
    mock_genai_client.aio.live.connect = mock_connect
    
    await agent.connect_to_live_server()
    
    mock_connect.assert_called_once_with(
        config={"response_modalities": ["TEXT"]},
        model_name="gemini-pro",
        tools=[] # Default when no tools are provided to agent
    )
    assert agent.session == mock_connect.return_value

@pytest.mark.asyncio
async def test_gemini_agent_connect_to_live_server_custom_config(mock_genai_client):
    """Tests connection with custom initial configuration."""
    agent = GeminiAgent(model_name="gemini-pro")
    custom_config = {"response_modalities": ["TEXT", "CODE"], "temperature": 0.8}
    
    # Mock the live connect method
    mock_connect = AsyncMock()
    mock_genai_client.aio.live.connect = mock_connect
    
    await agent.connect_to_live_server(initial_config=custom_config)
    
    mock_connect.assert_called_once_with(
        config=custom_config,
        model_name="gemini-pro",
        tools=[] # Default when no tools are provided to agent
    )
    assert agent.session == mock_connect.return_value

@pytest.mark.asyncio
async def test_gemini_agent_connect_to_live_server_with_tools(mock_genai_client):
    """Tests connection with tools provided to the agent."""
    mock_tool1 = Tool(name="tool1", description="description1", function=lambda x: x)
    mock_tool1.openai_tool_schema = {
        "name": "tool1",
        "description": "description1",
        "parameters": {"type": "object", "properties": {"param1": {"type": "string"}}},
    }
    mock_tool2 = Tool(name="tool2", description="description2", function=lambda y: y*2)
    mock_tool2.openai_tool_schema = {
        "name": "tool2",
        "description": "description2",
        "parameters": {"type": "object", "properties": {"param2": {"type": "integer"}}},
    }
    tools = [mock_tool1, mock_tool2]
    agent = GeminiAgent(model_name="gemini-ultra", tools=tools)

    # Mock the live connect method
    mock_connect = AsyncMock()
    mock_genai_client.aio.live.connect = mock_connect

    await agent.connect_to_live_server()

    expected_tool_schemas = [
        genai_types.FunctionDeclaration(
            name="tool1",
            description="description1",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={"param1": genai_types.Schema(type=genai_types.Type.STRING)},
            ),
        ),
        genai_types.FunctionDeclaration(
            name="tool2",
            description="description2",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={"param2": genai_types.Schema(type=genai_types.Type.INTEGER)},
            ),
        ),
    ]
    
    mock_connect.assert_called_once_with(
        config={"response_modalities": ["TEXT"]},
        model_name="gemini-ultra",
        tools=expected_tool_schemas
    )
    assert agent.session == mock_connect.return_value

@pytest.mark.asyncio
async def test_gemini_agent_connect_to_live_server_with_invalid_tools(mock_genai_client, caplog):
    """Tests connection with tools that have missing/invalid schema."""
    mock_tool1 = Tool(name="tool1", description="description1", function=lambda x: x) # Missing schema
    mock_tool2 = Tool(name="tool2", description="description2", function=lambda y: y*2)
    mock_tool2.openai_tool_schema = { # Invalid schema (e.g. missing 'name')
        "description": "description2",
        "parameters": {"type": "object", "properties": {"param2": {"type": "integer"}}},
    }
    mock_tool3 = Tool(name="tool3", description="description3", function=lambda z: z*3)
    mock_tool3.openai_tool_schema = {
        "name": "tool3",
        "description": "description3",
        "parameters": {"type": "object", "properties": {"param3": {"type": "string"}}},
    }

    tools = [mock_tool1, mock_tool2, mock_tool3]
    agent = GeminiAgent(model_name="gemini-pro", tools=tools)

    # Mock the live connect method
    mock_connect = AsyncMock()
    mock_genai_client.aio.live.connect = mock_connect

    await agent.connect_to_live_server()

    # Only tool3 has a valid schema
    expected_tool_schemas = [
        genai_types.FunctionDeclaration(
            name="tool3",
            description="description3",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={"param3": genai_types.Schema(type=genai_types.Type.STRING)},
            ),
        )
    ]
    
    mock_connect.assert_called_once_with(
        config={"response_modalities": ["TEXT"]},
        model_name="gemini-pro",
        tools=expected_tool_schemas
    )
    assert agent.session == mock_connect.return_value
    
    # Check for warnings (caplog is a pytest fixture for capturing logs)
    assert "Tool 'tool1' is missing openai_tool_schema and will be skipped." in caplog.text
    assert "Tool 'tool2' has an invalid openai_tool_schema (missing 'name') and will be skipped." in caplog.text


@pytest.mark.asyncio
async def test_gemini_agent_connect_to_live_server_existing_active_session(mock_genai_client):
    """Tests connection when a session already exists and is active."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Create a mock for the existing session
    mock_old_session = AsyncMock()
    mock_old_session.is_active = True
    agent.session = mock_old_session
    
    # Mock the live connect method to return a new session
    mock_new_session = AsyncMock()
    mock_connect = AsyncMock(return_value=mock_new_session)
    mock_genai_client.aio.live.connect = mock_connect
    
    await agent.connect_to_live_server()
    
    # Assert old session was closed
    mock_old_session.close.assert_called_once()
    
    # Assert connect was called to create a new session
    mock_connect.assert_called_once_with(
        config={"response_modalities": ["TEXT"]},
        model_name="gemini-pro",
        tools=[] 
    )
    # Assert agent's session is the new session
    assert agent.session == mock_new_session

@pytest.mark.asyncio
async def test_gemini_agent_send_user_content_success(mock_genai_client):
    """Tests successful sending of user content."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Mock the session and its send_client_content method
    mock_session = AsyncMock()
    mock_session.send_client_content = AsyncMock() # This needs to be an async function
    agent.session = mock_session
    
    await agent.send_user_content("Hello, world!")
    
    mock_session.send_client_content.assert_called_once()
    call_args = mock_session.send_client_content.call_args
    
    # Check the structure of the 'turns' argument
    # Based on usage, send_client_content is called with turns=...
    # However, the actual method signature in the library might be just the content itself.
    # Let's assume it's called with the content directly as an argument,
    # and that content is what we're constructing.
    
    # The GeminiAgent.send_user_content creates a genai_types.Content object.
    # We need to assert that this object was passed.
    
    assert "turns" in call_args.kwargs
    sent_content = call_args.kwargs["turns"]
    
    assert isinstance(sent_content, genai_types.Content)
    assert sent_content.role == "user"
    assert len(sent_content.parts) == 1
    assert sent_content.parts[0].text == "Hello, world!"

@pytest.mark.asyncio
async def test_gemini_agent_send_user_content_no_session(mock_genai_client):
    """Tests sending user content when the session is not established."""
    agent = GeminiAgent(model_name="gemini-pro")
    # agent.session is None by default after initialization
    
    with pytest.raises(RuntimeError, match="Agent is not connected. Call connect_to_live_server first."):
        await agent.send_user_content("test")

@pytest.mark.asyncio
async def test_gemini_agent_send_tool_function_response_single(mock_genai_client):
    """Tests successful sending of a single tool function response."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Mock the session and its send_tool_response method
    mock_session = AsyncMock()
    mock_session.send_tool_response = AsyncMock() 
    agent.session = mock_session
    
    mock_function_response = MagicMock(spec=genai_types.FunctionResponse)
    
    await agent.send_tool_function_response(mock_function_response)
    
    mock_session.send_tool_response.assert_called_once_with(
        function_responses=[mock_function_response]
    )

@pytest.mark.asyncio
async def test_gemini_agent_send_tool_function_response_list(mock_genai_client):
    """Tests successful sending of a list of tool function responses."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Mock the session and its send_tool_response method
    mock_session = AsyncMock()
    mock_session.send_tool_response = AsyncMock()
    agent.session = mock_session
    
    mock_function_response1 = MagicMock(spec=genai_types.FunctionResponse)
    mock_function_response2 = MagicMock(spec=genai_types.FunctionResponse)
    list_of_mock_function_responses = [mock_function_response1, mock_function_response2]
    
    await agent.send_tool_function_response(list_of_mock_function_responses)
    
    mock_session.send_tool_response.assert_called_once_with(
        function_responses=list_of_mock_function_responses
    )

@pytest.mark.asyncio
async def test_gemini_agent_send_tool_function_response_no_session(mock_genai_client):
    """Tests sending tool function response when the session is not established."""
    agent = GeminiAgent(model_name="gemini-pro")
    # agent.session is None by default after initialization
    
    mock_function_response = MagicMock(spec=genai_types.FunctionResponse)
    
    with pytest.raises(RuntimeError, match="Agent is not connected. Call connect_to_live_server first."):
        await agent.send_tool_function_response(mock_function_response)

@pytest.mark.asyncio
async def test_gemini_agent_receive_model_messages_success_various_types(mock_genai_client, caplog):
    """Tests successful receiving and yielding of various message types."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Mock messages to be yielded by the session's receive method
    mock_text_message = MagicMock(spec=genai_types.LiveServerMessage)
    mock_text_message.text = "Hello from model"
    mock_text_message.tool_call = None # Ensure only one attribute is set for clarity

    mock_tool_call_message = MagicMock(spec=genai_types.LiveServerMessage)
    mock_function_call = MagicMock(spec=genai_types.FunctionCall)
    mock_function_call.name = "test_tool"
    mock_function_call.args = {"arg1": "value1"}
    mock_tool_call_message.tool_call = MagicMock(function_calls=[mock_function_call])
    mock_tool_call_message.text = None # Ensure only one attribute is set

    mock_other_message = MagicMock(spec=genai_types.LiveServerMessage) # e.g. turn_complete or other signal
    mock_other_message.text = None
    mock_other_message.tool_call = None
    
    mock_messages_sequence = [mock_text_message, mock_tool_call_message, mock_other_message]

    async def mock_receive_generator():
        for msg in mock_messages_sequence:
            yield msg
            
    mock_session = AsyncMock()
    mock_session.receive = mock_receive_generator # Assign the generator function itself
    agent.session = mock_session
    
    received_messages = []
    async for message in agent.receive_model_messages():
        received_messages.append(message)
        
    assert received_messages == mock_messages_sequence
    
    # Check caplog for debug messages
    assert "Received raw message object:" in caplog.text
    assert f"Received raw message object: {mock_text_message!r}" in caplog.text
    assert f"Received text: {mock_text_message.text}" in caplog.text
    assert f"Received raw message object: {mock_tool_call_message!r}" in caplog.text
    assert f"Received tool call: {mock_tool_call_message.tool_call!r}" in caplog.text
    assert f"Received raw message object: {mock_other_message!r}" in caplog.text
    # Ensure no "Received text" or "Received tool call" for the other message if they are None
    assert caplog.text.count(f"Received text: ") == 1 
    assert caplog.text.count(f"Received tool call: ") == 1

@pytest.mark.asyncio
async def test_gemini_agent_receive_model_messages_no_session(mock_genai_client):
    """Tests receiving model messages when the session is not established."""
    agent = GeminiAgent(model_name="gemini-pro")
    # agent.session is None by default

    async def consume_generator():
        # Attempt to consume the generator, which should raise the error
        # if the session check at the beginning of receive_model_messages fails.
        return [msg async for msg in agent.receive_model_messages()]

    with pytest.raises(RuntimeError, match="Agent is not connected. Call connect_to_live_server first."):
        await consume_generator()

@pytest.mark.asyncio
async def test_gemini_agent_execute_tool_success(mock_genai_client, caplog):
    """Tests successful execution of a tool."""
    # 1. Create a mock ToolResult
    mock_tool_result_content = "Successfully executed the tool"
    mock_tool_result = MagicMock(spec=ToolResult)
    mock_tool_result.content = mock_tool_result_content

    # 2. Create a mock Tool
    mock_tool = MagicMock(spec=Tool)
    mock_tool.name = "my_test_tool"
    mock_tool.run = MagicMock(return_value=mock_tool_result) # Synchronous run method

    # 3. Initialize GeminiAgent with the mock tool
    agent = GeminiAgent(model_name="gemini-pro", tools=[mock_tool])

    # 4. Create a mock func_call object
    func_call_args = {"param1": "value1", "param2": 123}
    mock_func_call = MagicMock(spec=genai_types.FunctionCall) # Simulate FunctionCall structure
    mock_func_call.name = "my_test_tool"
    mock_func_call.args = func_call_args # Directly use a dict for args
    mock_func_call.id = "test_call_id_success"

    # 5. Call agent._execute_tool
    # Although _execute_tool is async, the tool's run method is synchronous
    function_response = await agent._execute_tool(mock_func_call)

    # 6. Assert tool's run method was called
    mock_tool.run.assert_called_once_with(**func_call_args)

    # 7. Assert the result is a genai_types.FunctionResponse
    assert isinstance(function_response, genai_types.FunctionResponse)

    # 8. Check the FunctionResponse fields
    assert function_response.name == "my_test_tool"
    assert function_response.id == "test_call_id_success"
    assert function_response.response == {"result": mock_tool_result_content}

    # 9. Check caplog for log messages
    assert f"Executing tool: {mock_tool.name}" in caplog.text
    assert f"Tool {mock_tool.name} executed successfully. Result: {mock_tool_result_content}" in caplog.text


@pytest.mark.asyncio
async def test_gemini_agent_execute_tool_not_found(mock_genai_client, caplog):
    """Tests behavior when the requested tool is not found."""
    # 1. Initialize GeminiAgent with an empty tools list
    agent = GeminiAgent(model_name="gemini-pro", tools=[])

    # 2. Create a mock func_call object for an unknown tool
    mock_func_call = MagicMock(spec=genai_types.FunctionCall)
    mock_func_call.name = "unknown_tool"
    mock_func_call.args = {}
    mock_func_call.id = "test_call_id_not_found"

    # 3. Call agent._execute_tool
    function_response = await agent._execute_tool(mock_func_call)

    # 4. Assert the result is a genai_types.FunctionResponse
    assert isinstance(function_response, genai_types.FunctionResponse)

    # 5. Check the FunctionResponse fields for error
    assert function_response.name == "unknown_tool"
    assert function_response.id == "test_call_id_not_found"
    assert function_response.response == {"error": "Tool unknown_tool not found."}

    # 6. Check caplog for log message
    assert "Tool unknown_tool not found." in caplog.text


@pytest.mark.asyncio
async def test_gemini_agent_execute_tool_exception(mock_genai_client, caplog):
    """Tests behavior when tool execution raises an exception."""
    # 1. Create a mock Tool whose run method raises an exception
    mock_error_tool = MagicMock(spec=Tool)
    mock_error_tool.name = "error_tool"
    exception_message = "Tool failed dramatically!"
    mock_error_tool.run = MagicMock(side_effect=Exception(exception_message))

    # 2. Initialize GeminiAgent with this tool
    agent = GeminiAgent(model_name="gemini-pro", tools=[mock_error_tool])

    # 3. Create a mock func_call object
    mock_func_call = MagicMock(spec=genai_types.FunctionCall)
    mock_func_call.name = "error_tool"
    mock_func_call.args = {"param": "useless"}
    mock_func_call.id = "test_call_id_error"

    # 4. Call agent._execute_tool
    function_response = await agent._execute_tool(mock_func_call)

    # 5. Assert the result is a genai_types.FunctionResponse
    assert isinstance(function_response, genai_types.FunctionResponse)

    # 6. Check the FunctionResponse fields for error
    assert function_response.name == "error_tool"
    assert function_response.id == "test_call_id_error"
    assert function_response.response == {
        "error": f"Error executing tool error_tool: {exception_message}"
    }

    # 7. Check caplog for log message
    assert f"Executing tool: {mock_error_tool.name}" in caplog.text # Attempt to execute
    assert f"Error executing tool error_tool: {exception_message}" in caplog.text

@pytest.mark.asyncio
@patch.object(GeminiAgent, 'send_user_content', new_callable=AsyncMock)
@patch.object(GeminiAgent, 'connect_to_live_server', new_callable=AsyncMock)
async def test_chat_loop_basic_text_response(
    mock_connect_to_live_server, 
    mock_send_user_content, 
    mock_genai_client, 
    caplog
):
    """Tests chat_loop basic flow with only a text response."""
    agent = GeminiAgent(model_name="gemini-pro")
    # Ensure session is None initially so connect_to_live_server is called
    agent.session = None

    # Mock receive_model_messages
    mock_text_response = "Hello from the model!"
    mock_server_message = MagicMock(spec=genai_types.LiveServerMessage)
    mock_server_message.text = mock_text_response
    mock_server_message.tool_call = None
    # Simulate server_content indicating the turn is complete
    mock_server_message.server_content = MagicMock(model_turn=True, turn_complete=True)


    async def mock_receive_gen():
        yield mock_server_message
        # The loop in chat_loop continues as long as server_content.turn_complete is not True
        # or if there is a tool_call.
        # To stop after one text message, the message itself must indicate turn_complete
        # The agent's chat_loop checks message.server_content.turn_complete.
        # And also message.tool_call.

    agent.receive_model_messages = mock_receive_gen 

    user_input = "Hi there!"
    responses = await agent.chat_loop(user_input)

    mock_connect_to_live_server.assert_called_once()
    mock_send_user_content.assert_called_once_with(user_input)
    
    assert responses == [mock_text_response]
    assert f"CHAT_LOOP: Received text response: {mock_text_response}" in caplog.text
    assert "CHAT_LOOP: Model turn complete." in caplog.text


@pytest.mark.asyncio
@patch.object(GeminiAgent, 'send_tool_function_response', new_callable=AsyncMock)
@patch.object(GeminiAgent, '_execute_tool', new_callable=AsyncMock)
@patch.object(GeminiAgent, 'send_user_content', new_callable=AsyncMock)
@patch.object(GeminiAgent, 'connect_to_live_server', new_callable=AsyncMock)
async def test_chat_loop_with_tool_call_and_response(
    mock_connect_to_live_server,
    mock_send_user_content,
    mock_execute_tool,
    mock_send_tool_function_response,
    mock_genai_client,
    caplog
):
    """Tests chat_loop flow with a tool call, execution, and subsequent text response."""
    agent = GeminiAgent(model_name="gemini-pro")
    agent.session = None # Ensure connect is called

    # 1. Mock receive_model_messages to yield a tool call, then a text message
    mock_tool_name = "test_tool"
    mock_tool_args = {"arg": "value"}
    mock_tool_call_id = "tool_call_123"

    mock_func_call = MagicMock(spec=genai_types.FunctionCall)
    mock_func_call.name = mock_tool_name
    mock_func_call.args = mock_tool_args
    mock_func_call.id = mock_tool_call_id
    
    tool_call_message = MagicMock(spec=genai_types.LiveServerMessage)
    tool_call_message.text = None
    tool_call_message.tool_call = MagicMock(function_calls=[mock_func_call])
    # Simulate turn not complete because a tool call is expected to be followed by a response
    tool_call_message.server_content = MagicMock(model_turn=True, turn_complete=False) 

    final_text_response = "Tool executed, here's the final answer."
    text_message_after_tool = MagicMock(spec=genai_types.LiveServerMessage)
    text_message_after_tool.text = final_text_response
    text_message_after_tool.tool_call = None
    text_message_after_tool.server_content = MagicMock(model_turn=True, turn_complete=True)

    async def mock_receive_gen():
        yield tool_call_message
        yield text_message_after_tool

    agent.receive_model_messages = mock_receive_gen

    # 2. Mock _execute_tool to return a FunctionResponse
    mock_tool_execution_result = {"status": "Tool success"}
    mock_function_response = MagicMock(spec=genai_types.FunctionResponse)
    mock_function_response.name = mock_tool_name
    mock_function_response.id = mock_tool_call_id
    mock_function_response.response = mock_tool_execution_result
    mock_execute_tool.return_value = mock_function_response

    user_input = "Use the tool please."
    responses = await agent.chat_loop(user_input)

    mock_connect_to_live_server.assert_called_once()
    mock_send_user_content.assert_called_once_with(user_input)
    
    # Assert tool execution flow
    mock_execute_tool.assert_called_once_with(mock_func_call)
    mock_send_tool_function_response.assert_called_once_with(mock_function_response)
    
    assert responses == [final_text_response] # Only final text responses are collected
    
    # Check logs
    assert f"CHAT_LOOP: Received tool call: {tool_call_message.tool_call!r}" in caplog.text
    assert "CHAT_LOOP: Executing 1 tool calls." in caplog.text
    assert f"CHAT_LOOP: Sending tool response for {mock_tool_name}" in caplog.text
    assert f"CHAT_LOOP: Received text response: {final_text_response}" in caplog.text
    assert "CHAT_LOOP: Model turn complete." in caplog.text

@pytest.mark.asyncio
@patch.object(GeminiAgent, 'connect_to_live_server', new_callable=AsyncMock)
async def test_chat_loop_connect_fails(
    mock_connect_to_live_server,
    mock_genai_client
):
    """Tests chat_loop when connect_to_live_server fails."""
    agent = GeminiAgent(model_name="gemini-pro")
    
    # Configure connect_to_live_server to raise an error AND ensure session remains None
    async def connect_side_effect():
        # Simulate the behavior where connect might raise an error
        # and agent.session is not set or is explicitly set to None.
        agent.session = None 
        raise RuntimeError("Simulated connection failure")

    mock_connect_to_live_server.side_effect = connect_side_effect
    
    user_input = "Any input"
    with pytest.raises(RuntimeError, match="Failed to connect to the live server."):
        await agent.chat_loop(user_input)
        
    mock_connect_to_live_server.assert_called_once()

# TODO: Add more tests for other methods in GeminiAgent.
