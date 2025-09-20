from praisonaiagents import Agent, MCP


agent = Agent(
    instructions="You are a file system manager with MCP filesystem tools. Create and write files directly using your tools.",
    llm="ollama/llama3.2",
    tools=MCP("npx -y @modelcontextprotocol/server-filesystem C:/Users/slayer/Desktop/mcp")
)


agent.start("Create a file named 'text.txt' and write the text 'hello MCP' to it")