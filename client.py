import asyncio
import json
import logging
import os
from abc import abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, CallToolResult
from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from rich import logging as rich_logging
from rich import print as rprint

MODEL_NAME = "qwen3-235b-a22b"
load_dotenv()  # load environment variables from .env


class ModelInterface:
    @abstractmethod
    async def get_chat_completion(self, messages):
        pass


class QwenModel(ModelInterface):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    async def get_chat_completion(self, messages, tools):
        response = await self.client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={"enable_thinking": True, "thinking_budget": 500},
            stream=True,
            parallel_tool_calls=True,
        )
        return response


class DoubaoModel(ModelInterface):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DOUBAO_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    async def get_chat_completion(self, messages, tools):
        response = await self.client.chat.completions.create(
            # thinking model
            model="doubao-1-5-thinking-pro-250415",
            # multi-modal model
            # doubao-1-5-thinking-vision-pro-250428
            # doubao-1-5-vision-pro-32k-250115
            # model="doubao-1-5-vision-pro-32k-250115",
            # max_tokens=1000,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            parallel_tool_calls=True,
        )
        return response


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(rich_logging.RichHandler())


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


SERVER_CONFIG_FILE = ".server_config.json"


class MCPClient:
    def __init__(self):
        self.mcpSessions = {}
        self.tools: list[Tool] = []
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

    async def initialize(self, config_file=SERVER_CONFIG_FILE):
        with open(config_file, "r") as f:
            self.mcpServersConfig = json.load(f)
        await self.connect_to_server(self.mcpServersConfig)

    async def call_tool(self, id, tool_name, tool_args) -> ToolCallInfo:
        logger.debug(f"call_tool: {tool_name} with args {str(tool_args)[:100]}...")
        session: ClientSession = self.mcpToolsSessionMap[tool_name]
        if session is None:
            return ToolCallInfo(
                id=id,
                name=tool_name,
                args=tool_args,
                result=CallToolResult(
                    content=[{"text": f"Cannot find servers for tool {tool_name}"}],
                    isError=True,
                ),
            )

        result = await session.call_tool(tool_name, tool_args)
        # logger.debug(
        #     f"[Calling tool {tool_name} with args {tool_args}], \n  result: {
        #         result.content}"
        # )
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
            logger.info(
                f"\nConnected to server with tools: {[tool.name for tool in tools]}"
            )

            self.mcpSessions[server_name] = session
            for tool in tools:
                self.mcpToolsSessionMap[tool.name] = session

            self.tools.extend(tools)

    def list_tools(self):
        return self.tools

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


@dataclass
class AssistantResponseChunk:
    type: str
    content: str | dict | ToolCallInfo


class LLMClient:
    def __init__(self, mcp_config_file=SERVER_CONFIG_FILE):
        self.available_tools = []
        self.tools: list[Tool] = []
        self.qwenClient = QwenModel()
        self.doubaoClient = DoubaoModel()
        self.mcpClient = MCPClient()
        self.mcp_config_file = mcp_config_file

    async def __aenter__(self):
        await self.mcpClient.initialize(config_file=self.mcp_config_file)
        self.tools = self.mcpClient.list_tools()
        self.available_tools = get_tools_format(self.tools, type="qwen")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.mcpClient.cleanup()

    async def get_chat_completion(self, messages):
        # return await self.qwenClient.get_chat_completion(messages, self.available_tools)
        logger.debug(
            f"call doubao model with tools {[tool['function']['name'] for tool in self.available_tools]}"
        )
        return await self.doubaoClient.get_chat_completion(
            messages, self.available_tools
        )

    def get_tool_result_message(
        self, result: CallToolResult | Any, tool_call_id: str, type="tool"
    ):
        if type == "tool":
            return {
                "content": (
                    result.content[0].text
                    if isinstance(result, CallToolResult)
                    else str(result)
                ),
                "role": "tool",
                "tool_call_id": tool_call_id,
            }
        if type == "user":
            return {"content": result.content, "role": "user", "name": "tool caller"}

    async def process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk]
    ):
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                if delta.tool_calls is not None:
                    assert False, f"UNEXPECTED TOOL CALLS: {delta.tool_calls}"

                yield AssistantResponseChunk(
                    type="thinking", content=delta.reasoning_content
                )

            else:
                if delta.content is not None and delta.content != "":
                    yield AssistantResponseChunk(type="answer", content=delta.content)

                if delta.tool_calls is not None:
                    for tool_call in delta.tool_calls:
                        if (
                            tool_call.function
                            and not tool_call.function.name
                            and not tool_call.function.arguments
                        ):
                            break

                        yield AssistantResponseChunk(
                            type="tool_call", content=tool_call
                        )

    async def get_assistant_response(self, messages):
        while True:
            response = await self.get_chat_completion(messages)

            result = self.process_streamed_response(response)

            answer_content = ""
            reasoning_content = ""
            tool_call_message_params: dict[int:ChoiceDeltaToolCall] = {}
            tool_call_tasks = []
            tool_call_info = {}
            notified_calls = set()

            async for chunk in result:
                if chunk.type == "answer":
                    answer_content += chunk.content
                    yield chunk
                elif chunk.type == "thinking":
                    reasoning_content += chunk.content
                    yield chunk
                elif chunk.type == "tool_call":
                    tool_call_param: ChoiceDeltaToolCall = chunk.content
                    index = tool_call_param.index
                    if index not in tool_call_message_params:
                        tool_call_message_params[index] = tool_call_param
                    else:
                        tool_call_message_params[
                            index
                        ].function.arguments += tool_call_param.function.arguments

                    tool_call_param = tool_call_message_params[index]
                    if is_valid_json(tool_call_param.function.arguments):
                        tool_name = tool_call_param.function.name
                        tool_args = json.loads(tool_call_param.function.arguments)

                        task = asyncio.create_task(
                            self.mcpClient.call_tool(
                                tool_call_param.id, tool_name, tool_args
                            )
                        )
                        tool_call_tasks.append(task)

                        tool_info = ToolCallInfo(
                            id=tool_call_param.id,
                            name=tool_name,
                            args=tool_args,
                            result=None,
                        )
                        tool_call_info[tool_call_param.id] = tool_info

                        notified_calls.add(tool_call_param.id)
                        yield AssistantResponseChunk(
                            type="tool_call", content=tool_info
                        )

            for id, info in tool_call_info.items():
                if id not in notified_calls:
                    logger.error(
                        f"Malformed Tool call {info.name} with args {info.args}"
                    )

            assistant_msg_record = {
                "role": "assistant",
                "content": answer_content,
            }
            if tool_call_info:
                assistant_msg_record["tool_calls"] = [
                    param.to_dict() for param in tool_call_message_params.values()
                ]
            messages.append(assistant_msg_record)

            async for task in asyncio.as_completed(tool_call_tasks):
                result: ToolCallInfo = await task
                messages.append(self.get_tool_result_message(result.result, result.id))
                yield AssistantResponseChunk(type="tool_call_result", content=result)

            if not tool_call_info:
                break
