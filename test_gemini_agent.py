import asyncio
from app.domain.agents.gemini_agent import GeminiAgent
from app.domain.tools.base import report_tool # Assuming report_tool is a Tool instance

async def main():
    """
    Basic integration test for GeminiAgent.
    """
    # IMPORTANT: Replace with your actual API key
    API_KEY = "YOUR_API_KEY_HERE" 
    MODEL_NAME = 'gemini-2.0-flash-live-001' # As specified in the task from issue context

    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please replace 'YOUR_API_KEY_HERE' with your actual Gemini API key in test_gemini_agent.py")
        return

    print(f"Initializing GeminiAgent with model: {MODEL_NAME}")
    agent = GeminiAgent(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        tools=[report_tool] # report_tool should be an instance of a Tool class
    )

    initial_input = "Hello, this is a test. Please summarize this message using the report tool: Test message for summarization."
    print(f"Sending initial input to chat_loop: '{initial_input}'")

    try:
        # The connect_to_live_server is called within chat_loop if session is not active
        responses = await agent.chat_loop(initial_user_input=initial_input)
        print("\nAgent responses:")
        if responses:
            for response in responses:
                print(response)
        else:
            print("No text responses received from the agent.")

    except Exception as e:
        print(f"An error occurred during the chat loop: {e}")
    finally:
        if agent.session and agent.session.is_active:
            print("Closing agent session...")
            await agent.session.close() # Ensure session is closed if opened

if __name__ == "__main__":
    asyncio.run(main())
