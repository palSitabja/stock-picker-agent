import asyncio
import json
import sys
import uuid

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from McpClient import McpClient

# Global variables to maintain state
client = None
agent = None
config = None
checkpointer = InMemorySaver()


async def initialize_agent():
    """Initialize the agent and tools"""
    global client, agent, config

    print("🔄 Initializing agent...")

    client = McpClient()
    tools = await client.get_tools()

    agent = create_react_agent(
        name='stock-picker-agent',
        model=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
        ),
        tools=tools,
        checkpointer=checkpointer
    )

    # Create a persistent thread for the conversation
    config = {"configurable": {"thread_id": uuid.uuid4()}}

    print("✅ Agent initialized successfully!")
    print("💬 You can now start chatting. Type 'quit', 'exit', or 'bye' to end the conversation.\n")


async def handle_event(event: dict) -> None:
    """Handle different types of events from the agent stream"""
    kind = event["event"]

    if kind == "on_chat_model_stream":
        # Handle streaming tokens from the LLM
        chunk = event["data"]["chunk"]
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)

    elif kind == "on_tool_start":
        tool_name = event['name']
        tool_input = event['data'].get('input', {})

        print(f"\n\n🔧 Using tool: {tool_name}")
        if tool_input:
            try:
                print(f"📥 Input: {json.dumps(tool_input, indent=2, ensure_ascii=False)}")
            except (TypeError, ValueError):
                print(f"📥 Input: {str(tool_input)}")
        print("-" * 50)

    elif kind == "on_tool_end":
        tool_name = event['name']
        tool_output = event['data'].get('output', None)

        print(f"\n✅ Tool completed: {tool_name}")
        if tool_output is not None:
            # Just print the raw output without JSON serialization
            output_str = str(tool_output)
            output_preview = output_str[:500] + "..." if len(output_str) > 500 else output_str
            print(f"📤 Output: {output_preview}")
        print("-" * 50)
        print("🤖 ", end="", flush=True)  # Continue assistant response

    elif kind == "on_chain_end" and event["name"] == "stock-picker-agent":
        pass  # Agent finished processing - handled in main loop


async def process_message(user_message: str) -> None:
    """Process a user message and stream the response"""
    print(f"\n🤖 Assistant: ", end="", flush=True)

    try:
        async for event in agent.astream_events(
                {"messages": [("user", user_message)]},
                config=config,
                version="v1"
        ):
            await handle_event(event)

    except Exception as e:
        print(f"\n❌ Error processing message: {str(e)}")

    print("\n")  # Add newline after response


def get_user_input() -> str:
    """Get user input with a nice prompt"""
    try:
        return input("👤 You: ").strip()
    except (KeyboardInterrupt, EOFError):
        return "quit"


def is_exit_command(message: str) -> bool:
    """Check if the user wants to exit"""
    return message.lower() in ['quit', 'exit', 'bye', 'q']


def display_welcome():
    """Display welcome message"""
    print("\n" + "=" * 60)
    print("🚀 Welcome to the Stock Picker Agent Chat!")
    print("=" * 60)
    print("Ask me anything about stocks, market data, or financial information.")
    print("I can help you with stock prices, company information, and more!")
    print("-" * 60)


def display_goodbye():
    """Display goodbye message"""
    print("\n" + "=" * 40)
    print("👋 Thanks for chatting! Goodbye!")
    print("=" * 40)


async def run_chat():
    """Run the main chat loop"""
    await initialize_agent()
    display_welcome()

    while True:
        try:
            user_message = get_user_input()

            if not user_message:
                continue

            if is_exit_command(user_message):
                break

            await process_message(user_message)

        except KeyboardInterrupt:
            print("\n\n🛑 Chat interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

    display_goodbye()


async def main():
    """Main entry point"""
    load_dotenv()
    await run_chat()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)