from rich import print as rprint
import asyncio
from dataclasses import dataclass, field
import json
import os

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, CallToolRequest

from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

load_dotenv()  # load environment variables from .env

mcpServersConfig = {
    "fileSystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/bytedance/mcp-test",
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
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "BSAPLs6U9IdkmBc-OGLjaRlh4_yis0I"},
    },
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
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


def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


@dataclass
class ChatResponseContext:
    tool_info_map: dict[int, ChoiceDeltaToolCall] = field(default_factory=dict)
    tool_info: list[dict] = field(default_factory=list)
    tool_call_tasks: list[asyncio.Task] = field(default_factory=list)
    is_answering: bool = False
    is_reasoning: bool = False
    is_tool_calling: bool = False
    reasoning_content: str = ""
    answer_content: str = ""


@dataclass 
class AssistantResponseChunk:
    type: str
    content: str | dict | ChoiceDeltaToolCall

class MCPClient:
    def __init__(self):
        self.mcpSessions = {}
        self.available_tools = []
        self.mcpToolsSessionMap = {}
        self.exit_stack = AsyncExitStack()
        self.qwenClient = AsyncOpenAI(
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

    async def get_chat_completion(self, messages):
        # rprint(messages)
        response = await self.qwenClient.chat.completions.create(
            # model="qwen-turbo-2024-11-01",
            model="qwen-plus-2025-04-28",
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
            # extra_body={"enable_thinking": True, "thinking_budget": 200},
            extra_body={"enable_thinking": True},
            stream=True,
            parallel_tool_calls=True,
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

    async def call_tool(self, tool_name, tool_args) -> CallToolResult:
        print(f"call_tool: {tool_name} with args {str(tool_args)[:100]}...")
        session: ClientSession = self.mcpToolsSessionMap[tool_name]
        if session is None:
            return CallToolResult(
                content=[{"text": f"Cannot find servers for tool {tool_name}"}],
                isError=True,
            )

        result = await session.call_tool(tool_name, tool_args)
        print(
            f"[Calling tool {tool_name} with args {tool_args}], \n  result: {
                result.content[0].text}"
        )
        return result

    async def process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk]
    ):
        context = ChatResponseContext()

        def process_thinking_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            if delta.tool_calls is not None:
                print("*" * 20 + "UNEXPECTED TOOL CALLS" + "*" * 20)
                print(delta.tool_calls)
                print("*" * 50)

            return AssistantResponseChunk(type="thinking", content=delta.reasoning_content)


        def process_answer_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            return AssistantResponseChunk(type="answer", content=delta.content)

        def process_tool_call_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            for tool_call in delta.tool_calls:
                if (
                    tool_call.function
                    and not tool_call.function.name
                    and not tool_call.function.arguments
                ):
                    break

                index = tool_call.index
                if index not in context.tool_info_map:
                    context.tool_info_map[index] = tool_call
                else:
                    context.tool_info_map[index].function.arguments += tool_call.function.arguments

                current_tool_call = context.tool_info_map[index]
                if is_valid_json(current_tool_call.function.arguments):
                    return AssistantResponseChunk(type="tool_call", content=current_tool_call)
            
            return None

        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                yield process_thinking_chunk(delta, context)

            else:
                if delta.content is not None and delta.content != "":
                    yield process_answer_chunk(delta, context)

                if delta.tool_calls is not None:
                    chunk = process_tool_call_chunk(delta, context)
                    if chunk is not None:
                        yield chunk

    async def get_assistant_response(self, messages):

        round = 1
        while True:
            print(
                f"{'=' * 20} Assistant round {round} {'=' * 20}",
            )
            round += 1

            response = await self.get_chat_completion(messages)

            result = self.process_streamed_response(response)

            answer_content = ""
            reasoning_content = ""
            tool_call_tasks = []
            tool_call_info = []

            async for chunk in result:
                if chunk.type == "answer":
                    answer_content += chunk.content
                    yield AssistantResponseChunk(type="answer", content=chunk.content)
                elif chunk.type == "thinking":
                    reasoning_content += chunk.content
                    yield AssistantResponseChunk(type="thinking", content=chunk.content)
                elif chunk.type == "tool_call":
                    task = asyncio.create_task(self.call_tool(chunk.content.function.name, json.loads(chunk.content.function.arguments)))
                    tool_call_tasks.append(task)
                    tool_call_info.append(chunk.content)
                    yield AssistantResponseChunk(type="tool_call", content=f"Calling tool {chunk.content.function.name} with args {str(chunk.content.function.arguments)[:20]}...")

            tool_call_results = await asyncio.gather(*tool_call_tasks, return_exceptions=True)

            assistant_msg_record = {
                "role": "assistant",
                "content": answer_content,
            }
            if tool_call_info:
                assistant_msg_record["tool_calls"] = tool_call_info

            messages.append(assistant_msg_record)

            if tool_call_results:
                messages.extend(
                    {
                        "content": str(tool_call_result),
                        "role": "tool",
                        "tool_call_id": tool_call_info[idx].id,
                    }
                    for idx, tool_call_result in enumerate(tool_call_results)
                )
            else:
                break


    async def chat_loop(self):
        """Run an interactive chat loop"""
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system",
                "content": load_system_prompt(),
            },
        ]

        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        round = 1
        while True:
            try:
                print(f"\n\n{'*' * 20} Chat round {round} {'*' * 20}")
                round += 1

                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                if not query:
                    continue

                messages.append({"role": "user", "content": query})

                current_type = None
                async for chunk in self.get_assistant_response(messages):
                    if chunk.type == "answer":
                        if current_type != "answer":
                            current_type = "answer"
                            rprint("[bold yellow]" + "\n" + "=" * 20 + "完整回复" + "=" * 20 + "[/bold yellow]\n")
                        rprint(chunk.content, end="", flush=True)
                    elif chunk.type == "thinking":
                        if current_type != "thinking":
                            current_type = "thinking"
                            rprint("[bold cyan]" + "\n" + "=" * 20 + "思考过程" + "=" * 20 + "[/bold cyan]\n")
                        rprint(chunk.content, end="", flush=True)
                    elif chunk.type == "tool_call":
                        if current_type != "tool_call":
                            current_type = "tool_call"
                            rprint("[bold green]" + "\n" + "=" * 20 + "工具调用" + "=" * 20 + "[/bold green]\n")
                        rprint(chunk.content, end="", flush=True)

            except KeyboardInterrupt:
               break 

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


def load_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(mcpServersConfig)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())
