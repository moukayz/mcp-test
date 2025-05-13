import asyncio
import json
import os
import gradio as gr

from contextlib import AsyncExitStack
from typing import AsyncGenerator, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, CallToolRequest

from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass, asdict


load_dotenv()  # load environment variables from .env

mcpServersConfig = {
    "fileSystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/mouka/mcp-test",
        ],
    },
    "weather": {
        "command": "python",
        "args": [
            "./server/weather.py",
        ],
    },
    "code_executor": {
        "command": "python",
        "args": [
            "./server/code_executor.py",
        ],
    },
    "brave-search": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-brave-search"
        ],
        "env": {
            "BRAVE_API_KEY": "BSAPLs6U9IdkmBc-OGLjaRlh4_yis0I"
        }
    },
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"]
    }
}


def get_tools_format(tools, type="qwen"):
    if type == "qwen":
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    elif type == "anthropic":
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]
    return available_tools


class MCPClient:
    def __init__(self):
        self.mcpSessions = {}
        self.available_tools = []
        self.mcpToolsSessionMap = {}
        self.exit_stack = AsyncExitStack()
        self.qwenClient = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.is_initialized = False

    async def connect_to_server(self, configs: dict):
        for server_name, config in configs.items():
            server_params = StdioServerParameters(**config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await session.initialize()

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server {server_name} with tools:", [tool.name for tool in tools])

            self.available_tools.extend(get_tools_format(tools, type="qwen"))
            self.mcpSessions[server_name] = session
            for tool in tools:
                self.mcpToolsSessionMap[tool.name] = session
        
        self.is_initialized = True

    def get_chat_completion(self, messages):
        response = self.qwenClient.chat.completions.create(
            model="qwen-turbo-2024-11-01",
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
        )
        return response

    def get_tool_result_message(
        self, result: CallToolResult | str, tool_call_id: str, type="tool"
    ):
        if type == "tool":
            return {
                "content": (
                    result.content if isinstance(result, CallToolResult) else result
                ),
                "role": "tool",
                "tool_call_id": tool_call_id,
            }
        if type == "user":
            return {"content": result.content, "role": "user", "name": "tool caller"}

    async def call_tool(self, tool_name, tool_args) -> str | CallToolRequest:
        print(f"[Calling tool] {tool_name} with args {tool_args}")

        session: ClientSession = self.mcpToolsSessionMap.get(tool_name)
        if session is None:
            return f"Cannot find servers for tool ${tool_name}"

        result = await session.call_tool(tool_name, tool_args)
        tool_result = f"[Tool {tool_name}]: {result.content[0].text}"
        print(f"[Calling tool] {tool_name} with args {tool_args}, \n[Tool result]: {result.content[0].text}")
        return result

    async def process_message(self, messages: list[dict]) -> AsyncGenerator[gr.ChatMessage, None]:
        assistant_messages = []
        tool_outputs = []

        # print(messages[0]["content"])

        round = 1
        while True:
            print(f"{'=' * 20}Processing round {round} {'=' * 20}")
            round += 1

            response = self.get_chat_completion(messages)
            assistant_output = response.choices[0].message
            if assistant_output.content:
                assistant_messages.append(assistant_output.content)
                print(f"[assistant message]: {assistant_output.content}")

                yield gr.ChatMessage(
                    role="assistant",
                    content=assistant_output.content,
                )
            else:
                assistant_output.content = ""

            messages.append(assistant_output)

            if assistant_output.tool_calls is None:
                # this is the stop point of tool calling loop
                print('*' * 20 + "Exit tool calling loop" + '*' * 20)
                break

            tools_call_record = {
                "assistant_message": assistant_output.content,
                "tool_calls": []
            }

            for idx, tool_call in enumerate(assistant_output.tool_calls):
                called_function = tool_call.function
                tool_name = called_function.name
                tool_args = json.loads(called_function.arguments)

                tool_call_item = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": None
                }

                yield gr.ChatMessage(
                    role="assistant",
                    content=f"🛠[️ Calling tool]: {tool_name} with args {tool_args}",
                    metadata={"title": f"🛠[️ Calling tool]: {tool_name}", "status": "pending"},
                )

                result = await self.call_tool(tool_name, tool_args)
                result_content = result.content[0].text if hasattr(result, 'content') else result

                yield gr.ChatMessage(
                    role="assistant",
                    content=result_content,
                    metadata={"title": f"🛠[️ Calling tool]: {tool_name}", "status": "done"},
                )

                tool_call_item["tool_result"] = result_content
                tools_call_record["tool_calls"].append(tool_call_item)
                
                messages.append(
                    self.get_tool_result_message(result, tool_call.id, type="tool")
                )

            tool_outputs.append(tools_call_record)


        # return assistant_messages[-1], tool_outputs

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


# Initialize the MCP client
client = MCPClient()

# Create an event for initialization completion
init_complete = asyncio.Event()

async def initialize_client():
    print("Initializing client...")
    await client.connect_to_server(mcpServersConfig)
    init_complete.set()

async def cleanup_client():
    await client.cleanup()


# System message for the chat
system_message = {
    "role": "system",
    "content": """you should generate the response step by step.
    you can use file system tools to achieve file system operations.

    you can use bash executor tools to execute shell commands or scripts.

    you can use python executor tools to execute python code block.

    the current directory is a python project with venv, if you need install any python tools, you can install them using "uv add <package_name>" rather than "pip install <package_name>".

    if you need to get information from the Internet, you should use the **search tool** to get useful links,
      and use **fetch** tool to get the content of top 5 among these links, then generate the response using these resources!

    """,
}

def format_message(role, content):
    if role == "user":
        return f"👤 **User**: {content}"
    else:
        return f"🤖 **Assistant**: {content}"

def updateSystemPrompt(system_prompt, current_conversation):
    if len(current_conversation) == 0:
        current_conversation.append({"role": "system", "content": system_prompt})
    elif current_conversation[0]["role"] != "system":
        current_conversation.insert(0, {"role": "system", "content": system_prompt})
    else:
        current_conversation[0]["content"] = system_prompt


# Create the Gradio interface
with gr.Blocks(title="MCP Chat Assistant") as demo:
    demo.load(initialize_client)
    demo.unload(client.cleanup)
    
    gr.Markdown("# MCP Chat Assistant")
    gr.Markdown("Chat with an AI assistant that can perform various operations using tools.")
    
    
    with gr.Row():

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, type="messages")

        with gr.Column(scale=1):

            gr.Markdown("## System Message")
            system_prompt = gr.Textbox(
                value=system_message["content"],
                # label="System Message (instructions for the AI)",
                show_label=False,
                container=False,
                lines=20
            )

    msg = gr.Textbox(
        placeholder="Ask a question or request an action...",
        show_label=False,
        container=False
    )
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
        reset_system = gr.Button("Reset System Message")
    
    def onUserSubmit(user_input, history, system_prompt):
        # Add user message to history
        return "", history + [{"role": "user", "content": user_input}], system_prompt
    
    async def getCompletion(history: list[gr.ChatMessage], system_prompt):
        if not client.is_initialized:
            print("Waiting for initialization to complete...")
            await init_complete.wait()
            print("Initialization completed")

        updateSystemPrompt(system_prompt, history)

        # Process the message and update history with bot response
        print(history[-1])
        async for response in client.process_message(history.copy()):
            response = asdict(response)
            new_message_title = response.get('metadata').get('title') if response.get('metadata') else None
            last_message_title = history[-1].get('metadata').get('title') if history[-1].get('metadata') else None
            if new_message_title == last_message_title and new_message_title != None:
                history[-1] = response
            else:
                history.append(response)
            yield history

    
    def reset_system_message():
        return system_message["content"]
    
    msg.submit(onUserSubmit, [msg, chatbot, system_prompt], [msg, chatbot, system_prompt], queue=False).then(
        getCompletion, [chatbot, system_prompt], chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)
    reset_system.click(reset_system_message, None, system_prompt, queue=False)

# Start the Gradio app
if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        pass
