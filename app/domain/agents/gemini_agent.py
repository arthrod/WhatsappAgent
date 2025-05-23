import google.genai as genai
import google.genai.types as genai_types
-from typing import Optional, Dict, List, Any, AsyncIterator
+from typing import Any, AsyncIterator

from app.domain.tools.base import Tool, ToolResult

class GeminiAgent:
  """A class to represent the Gemini Agent."""
  model_name: str
  client: genai.Client
  session: Optional[genai_types.LiveSession]
  tools: List[Tool]

  def __init__(self, model_name: str, api_key: Optional[str] = None, tools: List[Tool] = None):
    self.model_name = model_name
    self.client = genai.Client(api_key=api_key)
    self.session = None
    self.tools = tools if tools is not None else []

  async def connect_to_live_server(self, initial_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Connects to the live server using the specified model and configuration.
    Tool configurations are automatically prepared if tools are provided.

    Args:
      initial_config: An optional dictionary for base session configuration.
                      Defaults to {"response_modalities": ["TEXT"]}.
                      Tool configurations will be merged with this.
    """
    if initial_config is None:
      session_config = {"response_modalities": ["TEXT"]}
    else:
      session_config = initial_config.copy() # Avoid modifying the original dict

    if self.tools:
      function_declarations = []
      for tool in self.tools:
        if hasattr(tool, 'openai_tool_schema') and 'function' in tool.openai_tool_schema:
          # Assuming Gemini's FunctionDeclaration is compatible with OpenAI's function schema part
          function_declarations.append(tool.openai_tool_schema['function'])
        else:
          # Log or handle tools that don't have the expected schema
          print(f"Warning: Tool {tool.name} does not have a compatible 'openai_tool_schema.function'.")
      
      if function_declarations:
        # Ensure "tools" key exists, then add "function_declarations"
        if "tools" not in session_config:
            session_config["tools"] = {}
        session_config["tools"]["function_declarations"] = function_declarations
      elif "tools" in session_config:
        # If no valid function declarations were found but "tools" key exists, remove it or ensure it's empty
        # For simplicity, let's ensure it's not there if no functions are declared.
        # Or, Gemini might require an empty list for tools if the key is present.
        # Let's assume if function_declarations is empty, we don't add it.
        pass


    # Simple reassignment for now, will handle existing sessions later if needed.
    # If a session already exists, it should ideally be closed first.
    # For now, we'll just let the new connect overwrite it.
    print(f"Connecting to live server with config: {session_config}")
    # Ensure any previously active session is closed before creating a new one.
    # This is a simplified example; consider logging and more robust error handling here.
    if self.session and hasattr(self.session, 'is_active') and self.session.is_active and hasattr(self.session, 'close'):
        await self.session.close()

    self.session = await self.client.aio.live.connect(model=self.model_name, config=session_config)
      # TODO: Implement message sending/receiving logic here (if any immediate action after connect).
      pass

  async def send_user_content(self, user_input: str) -> None:
    """
    Sends user input to the live session.

    Args:
      user_input: The text input from the user.

    Raises:
      RuntimeError: If the agent is not connected to a live session.
    """
    if not self.session:
      raise RuntimeError("Agent is not connected. Call connect_to_live_server() first.")

    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])
    await self.session.send_client_content(turns=content)

  async def send_tool_function_response(
      self,
      function_responses: genai_types.FunctionResponse | List[genai_types.FunctionResponse]
  ) -> None:
    """
    Sends tool function responses back to the live session.

    Args:
      function_responses: A single FunctionResponse or a list of FunctionResponse objects.

    Raises:
      RuntimeError: If the agent is not connected to a live session.
    """
    if not self.session:
      raise RuntimeError("Agent is not connected. Call connect_to_live_server() first.")

    await self.session.send_tool_response(function_responses=function_responses)

  async def receive_model_messages(self) -> genai_types.AsyncIterator[Any]: # TODO: Replace Any with specific genai_types.LiveServerMessage if available
    """
    Receives and yields messages from the model via the live session.

    Yields:
      Messages from the model. Type is Any for now, ideally genai_types.LiveServerMessage.

    Raises:
      RuntimeError: If the agent is not connected to a live session.
    """
    if not self.session:
      raise RuntimeError("Agent is not connected. Call connect_to_live_server() first.")

    async for message in self.session.receive():
      print(f"Received raw message object: {message}") # For debugging
      if message.text:
        print(f"Received text: {message.text}")
      if message.tool_call:
        print(f"Received tool call: {message.tool_call}")
      yield message


  async def _execute_tool(self, func_call: Any) -> genai_types.FunctionResponse:
    """
    Executes a tool call and returns a FunctionResponse.

    Args:
      func_call: A function call object from the model (e.g., message.tool_call.function_calls[0]).
                 Expected to have .name, .args, and .id attributes.

    Returns:
      A genai_types.FunctionResponse indicating the outcome of the tool execution.
    """
    tool_name = func_call.name
    # func_call.args is a google.protobuf.struct_pb2.Struct, convert to dict for **kwargs
    py_tool_args = dict(func_call.args) 
    tool_id = func_call.id

    print(f"Attempting to execute tool: {tool_name} with args: {py_tool_args} and id: {tool_id}")

    found_tool: Optional[Tool] = None
    for tool_instance in self.tools:
      if tool_instance.name == tool_name:
        found_tool = tool_instance
        break

    if found_tool:
      try:
        print(f"Executing tool '{tool_name}' with arguments: {py_tool_args}")
        # tool.run is expected to be synchronous as per the subtask description
        tool_result: ToolResult = found_tool.run(**py_tool_args)
        print(f"Tool '{tool_name}' executed successfully. Result: {tool_result.content}")
        return genai_types.FunctionResponse(
            name=tool_name,
            id=tool_id,
            response={'result': tool_result.content}
        )
      except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        return genai_types.FunctionResponse(
            name=tool_name,
            id=tool_id,
            response={'error': f'Error executing tool {tool_name}: {str(e)}'}
        )
    else:
      print(f"Tool {tool_name} not found.")
      return genai_types.FunctionResponse(
          name=tool_name,
          id=tool_id,
          response={'error': f'Tool {tool_name} not found.'}
      )

  async def chat_loop(self, initial_user_input: str) -> List[str]:
    """
    Manages a chat session: connects, sends initial input, and processes messages.
    Handles text responses and tool calls.

    Args:
      initial_user_input: The first message from the user.

    Returns:
      A list of text responses received from the model.
    """
    collected_text_responses: List[str] = []

    if not self.session or not self.session.is_active: # Check if session is active
      print("No active session, connecting to live server...")
      await self.connect_to_live_server()
      if not self.session: # connect_to_live_server might fail or not set session if context manager exits
          raise RuntimeError("Failed to connect to the live server.")


    print(f"Sending initial user input: {initial_user_input}")
    await self.send_user_content(initial_user_input)

    print("Starting to receive model messages...")
    async for message in self.receive_model_messages():
      has_actionable_content = False # Flag to check if the message had text or tool_call

      if message.text:
        # print(f"Chat loop received text: {message.text}") # Already printed in receive_model_messages
        collected_text_responses.append(message.text)
        has_actionable_content = True
      
      if message.tool_call and message.tool_call.function_calls:
        # print(f"Chat loop received tool_call: {message.tool_call}") # Already printed
        has_actionable_content = True
        # Process all function calls in the message
        function_responses_to_send = []
        for func_call in message.tool_call.function_calls:
          # func_call is expected to be of a type like genai_types.FunctionCallPart or similar
          # which should have .name, .args, and .id
          # _execute_tool now always returns a FunctionResponse
          tool_response = await self._execute_tool(func_call)
          function_responses_to_send.append(tool_response)


        if function_responses_to_send:
          print(f"Sending {len(function_responses_to_send)} tool function response(s).")
          await self.send_tool_function_response(function_responses=function_responses_to_send)
      
      # Check for other server content if no text or tool_call was primary
      if not message.text and not message.tool_call and message.server_content:
        print(f"Received other server content: {message.server_content}")
        # Depending on what server_content means, it might be actionable
        # For now, just logging it.
        # has_actionable_content = True # Uncomment if server_content should be considered actionable

      if message.turn_complete:
        print(f"Turn complete signal received. Message had tool_call: {bool(message.tool_call)}")
        if not message.tool_call: # If turn is complete AND there's no tool call in THIS message
          print("Turn complete and no pending tool call in this message. Breaking receive loop for this turn.")
          break
        else:
          print("Turn complete, but a tool call was present. Continuing to process potential tool responses.")
      
      if not has_actionable_content and not message.turn_complete:
          print(f"Received a message with no text or tool_call, and turn not yet complete: {message}")

    print("Chat loop finished processing messages for this turn.")
    return collected_text_responses
