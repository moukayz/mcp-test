from typing import Any
from rich import print as rprint, logging as rich_logging
import asyncio
from dataclasses import dataclass, field
import json
import os
import logging

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, CallToolRequest

from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

load_dotenv()  # load environment variables from .env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(rich_logging.RichHandler())

SERVER_CONFIG_FILE = ".server_config.json"

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
class ToolCallInfo:
    id: str
    name: str
    args: dict
    result: CallToolResult

@dataclass 
class AssistantResponseChunk:
    type: str
    # content: str | dict | ChoiceDeltaToolCall | ToolCallInfo
    content: str | dict | ToolCallInfo

class MCPClient:
    def __init__(self):
        self.mcpSessions = {}
        self.tools : list[Tool] = []
        self.mcpToolsSessionMap = {}
        self.exit_stack = AsyncExitStack()
        self.mcpServersConfig = {}

    @staticmethod
    def get_tool_prompt(tool: Tool, note: str = ""):
        args_desc = []
        if "properties" in tool.input_schema:
            for param_name, param_info in tool.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in tool.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
        Tool: {tool.name}
        Description: {tool.description}
        Arguments:
        {chr(10).join(args_desc)}
        Additional Notes: {note}
        """

    async def initialize(self):
        with open(SERVER_CONFIG_FILE, "r") as f:
            self.mcpServersConfig = json.load(f)
        await self.connect_to_server(self.mcpServersConfig)

    async def call_tool(self,id, tool_name, tool_args) -> ToolCallInfo:
        logger.info(f"call_tool: {tool_name} with args {str(tool_args)[:100]}...")
        session: ClientSession = self.mcpToolsSessionMap[tool_name]
        if session is None:
            return ToolCallInfo(id=id, name=tool_name, args=tool_args, result=CallToolResult(
                content=[{"text": f"Cannot find servers for tool {tool_name}"}],
                isError=True,
            ))

        result = await session.call_tool(tool_name, tool_args)
        logger.info(
            f"[Calling tool {tool_name} with args {tool_args}], \n  result: {
                result.content[0].text}"
        )
        return ToolCallInfo(id=id, name=tool_name, args=tool_args, result=result)


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

            self.mcpSessions[server_name] = session
            for tool in tools:
                self.mcpToolsSessionMap[tool.name] = session

            self.tools.extend(tools)

    def list_tools(self):
        return self.tools

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

class LLMClient:
    def __init__(self):
        self.available_tools = []
        self.tools : list[Tool] = []
        self.qwenClient = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.mcpClient = MCPClient()

    async def __aenter__(self):
        await self.mcpClient.initialize()
        self.tools = self.mcpClient.list_tools()
        self.available_tools = get_tools_format(self.tools, type="qwen")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.mcpClient.cleanup()

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
        self, result: CallToolResult | Any, tool_call_id: str, type="tool"
    ):
        if type == "tool":
            return {
                "content": (
                    result.content if isinstance(result, CallToolResult) else str(result)
                ),
                "role": "tool",
                "tool_call_id": tool_call_id,
            }
        if type == "user":
            return {"content": result.content, "role": "user", "name": "tool caller"}

    async def process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk]
    ):
        tool_info_map: dict[int, ChoiceDeltaToolCall] = {}

        def process_thinking_chunk(delta: ChoiceDelta):
            if delta.tool_calls is not None:
                assert False, f"UNEXPECTED TOOL CALLS: {delta.tool_calls}"

            return AssistantResponseChunk(type="thinking", content=delta.reasoning_content)


        def process_answer_chunk(delta: ChoiceDelta):
            return AssistantResponseChunk(type="answer", content=delta.content)

        def process_tool_call_chunk(delta: ChoiceDelta, tool_info_map: dict[int, ChoiceDeltaToolCall]):
            for tool_call in delta.tool_calls:
                if (
                    tool_call.function
                    and not tool_call.function.name
                    and not tool_call.function.arguments
                ):
                    break

                return AssistantResponseChunk(type="tool_call", content=tool_call)
            
            return None

        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                yield process_thinking_chunk(delta)

            else:
                if delta.content is not None and delta.content != "":
                    yield process_answer_chunk(delta)

                if delta.tool_calls is not None:
                    chunk = process_tool_call_chunk(delta, tool_info_map)
                    if chunk is not None:
                        yield chunk

    async def get_assistant_response(self, messages):

        while True:
            response = await self.get_chat_completion(messages)

            result = self.process_streamed_response(response)

            answer_content = ""
            reasoning_content = ""
            tool_call_tasks = []
            tool_call_message_params = {}
            tool_call_info = {}
            notified_calls = set()

            async for chunk in result:
                if chunk.type == "answer":
                    answer_content += chunk.content
                    yield AssistantResponseChunk(type="answer", content=chunk.content)
                elif chunk.type == "thinking":
                    reasoning_content += chunk.content
                    yield AssistantResponseChunk(type="thinking", content=chunk.content)
                elif chunk.type == "tool_call":
                    tool_call_param : ChoiceDeltaToolCall = chunk.content
                    index = tool_call_param.index
                    if index not in tool_call_message_params:
                        tool_call_message_params[index] = tool_call_param
                    else:
                        tool_call_message_params[index].function.arguments += tool_call_param.function.arguments

                    tool_call_param = tool_call_message_params[index]
                    if is_valid_json(tool_call_param.function.arguments):

                        tool_name = tool_call_param.function.name
                        tool_args = json.loads(tool_call_param.function.arguments)

                        task = asyncio.create_task(self.mcpClient.call_tool(tool_call_param.id, tool_name, tool_args))
                        tool_call_tasks.append(task)

                        tool_info = ToolCallInfo(id=tool_call_param.id, name=tool_name, args=tool_args, result=None)
                        tool_call_info[tool_call_param.id] = tool_info

                        notified_calls.add(tool_call_param.id)
                        yield AssistantResponseChunk(type="tool_call", content=tool_info)

            for id, info in tool_call_info.items():
                if id not in notified_calls:
                    logger.error(f"Malformed Tool call {info.name} with args {info.args}")


            assistant_msg_record = {
                "role": "assistant",
                "content": answer_content,
            }
            if tool_call_info:
                assistant_msg_record["tool_calls"] = tool_call_message_params.values()
            messages.append(assistant_msg_record)

            async for task in asyncio.as_completed(tool_call_tasks):
                result: ToolCallInfo = await task
                messages.append(self.get_tool_result_message(result.result, result.id))
                yield AssistantResponseChunk(type="tool_call_result", content=result)


            if not tool_call_info:
                break

class ChatApp:
    # def __init__(self):
    #     self.llmClient = LLMClient()

    async def chat_loop(self):
        """Run an interactive chat loop"""
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system",
                "content": load_system_prompt(),
            },
        ]


        async with LLMClient() as llmClient:
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
                    async for chunk in llmClient.get_assistant_response(messages):
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



def load_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()


if __name__ == "__main__":

    asyncio.run(ChatApp().chat_loop())
