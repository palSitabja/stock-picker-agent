import uuid

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from McpClient import McpClient
import asyncio

async def main():
    client = McpClient()
    checkpointer = InMemorySaver()
    tools = await client.get_tools()
    agent = create_react_agent(
        name='stock-picker-agent',
        model=ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
            ),
        tools=tools,
        checkpointer=checkpointer
    )
    print("Agent created successfully. Add agent invocation logic.")
    # Example: Invoke the agent with a query
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    response = await agent.ainvoke(
        {"messages": [("user", "What is the stock price of Apple?")]},
        config=config
    )
    print(response)
    print("Agent created successfully. Add agent invocation logic.")


if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())