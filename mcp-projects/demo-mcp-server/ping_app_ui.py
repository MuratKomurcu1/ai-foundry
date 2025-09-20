from praisonaiagents import Agent, MCP
import gradio as gr


def ask_mcp_server (query):
    agent = Agent(
        instructions="You are a pins assistant. Use the tool to test a website's reachability.",
        llm="ollama/llama3.2",
        tools=MCP("node index.js")
    )

    result = agent.start(query)
    return f" ###Response \n\n{result}"


demo = gr.Interface(
    fn = ask_mcp_server,
    inputs=gr.Textbox(placeholder="exapmple: enter a website link"),
    outputs=gr.Markdown(),
    title="MCP Ping Ui ",
    description="test a website is reachability"
)

if __name__ == "__main__":
    demo.launch()
    