import asyncio
import json
import os

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, CallToolRequest

from dotenv import load_dotenv
from openai import OpenAI

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
            print("\nConnected to server with tools:", [tool.name for tool in tools])

            self.available_tools.extend(get_tools_format(tools, type="qwen"))
            self.mcpSessions[server_name] = session
            for tool in tools:
                self.mcpToolsSessionMap[tool.name] = session

    def get_chat_completion(self, messages):
        response = self.qwenClient.chat.completions.create(
            model="qwen-turbo-2024-11-01",
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
            # parallel_tool_calls=True
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

    async def call_tool(self, tool_name, tool_args) -> str | CallToolResult:
        session: ClientSession = self.mcpToolsSessionMap[tool_name]
        if session is None:
            return f"Cannot find servers for tool ${tool_name}"

        result = await session.call_tool(tool_name, tool_args)
        print(
            f"[Calling tool {tool_name} with args {tool_args}], \n  result: {
                result.content[0].text}"
        )
        return result

    async def get_response_2(self, messages):
        final_text = []

        round = 1
        while True:
            print(
                f"{'=' * 20}Processing round {round} {'=' * 20}",
            )
            round += 1

            response = self.get_chat_completion(messages)
            assistant_output = response.choices[0].message
            if assistant_output.content:
                final_text.append(assistant_output.content)
                print(f"assistant message: {assistant_output.content}")
            else:
                assistant_output.content = ""

            messages.append(assistant_output)

            if assistant_output.tool_calls is None:
                # this is the stop point of tool calling loop
                break

            for idx, tool_call in enumerate(assistant_output.tool_calls):
                called_function = tool_call.function
                tool_name = called_function.name
                tool_args = json.loads(called_function.arguments)

                result = await self.call_tool(tool_name, tool_args)

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                messages.append(
                    self.get_tool_result_message(result, tool_call.id, type="tool")
                )

        return final_text

    async def chat_loop(self):
        """Run an interactive chat loop"""
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system",
                "content": """you should generate the response step by step.
                you can use file system tools to achieve file system operations.

                you can use bash executor tools to execute shell commands or scripts.
                but everytime you need to execute a bash command or script, you should first get my approval then execute it, **especially when the command or script has side effects**.

                you can use python executor tools to execute python code block.

                the current directory is a python project with venv, if you need install any python tools, you can install them using "uv add <package_name>" rather than "pip install <package_name>".
                """,
            },
        ]

        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                messages.append({"role": "user", "content": query})

                response = await self.get_response_2(messages)
                # print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")
                print(messages)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(mcpServersConfig)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())
