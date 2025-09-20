from praisonaiagents import Agent,MCP
import gradio as gr

def file_system_custom(query):
    agent = Agent(
        instructions="You manage the file system using MCP.",
        llm="ollama/llama3.2",
        tools=MCP("npx -y @modelcontextprotocol/server-filesystem C:/Users/slayer/Desktop")
)

    result = agent.start(query)
    return f"## File System results \n\n {result}"


demo = gr.Interface(
    fn = file_system_custom,
    inputs = gr.Textbox(placeholder="Write 'hello mcp' to the file at C:/Users/slayer/Desktop/test.txt..."),
    outputs = gr.Markdown(),
    title= "File System MCP Asistant",
    description = "Enter your requirements below",

) 

if __name__ == "__main__":
    demo.launch()

