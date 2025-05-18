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
from openai.types.chat.chat_completion_chunk import ChoiceDelta

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
    tool_info: list[dict] = field(default_factory=list)
    tool_call_tasks: list[asyncio.Task] = field(default_factory=list)
    is_answering: bool = False
    is_reasoning: bool = False
    is_tool_calling: bool = False
    reasoning_content: str = ""
    answer_content: str = ""


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
        response = await self.qwenClient.chat.completions.create(
            # model="qwen-turbo-2024-11-01",
            model="qwen-plus-2025-04-28",
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
            extra_body={"enable_thinking": False, "thinking_budget": 100},
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

        def process_reasoning_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            if not context.is_reasoning:
                context.is_reasoning = True
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
            print(delta.reasoning_content, end="", flush=True)
            context.reasoning_content += delta.reasoning_content

            if delta.tool_calls is not None:
                print("*" * 20 + "UNEXPECTED TOOL CALLS" + "*" * 20)
                print(delta.tool_calls)
                print("*" * 50)

        def process_answer_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            if not context.is_answering:
                context.is_answering = True
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                context.is_answering = True

            print(delta.content, end="", flush=True)
            context.answer_content += delta.content

        def process_tool_call_chunk(delta: ChoiceDelta, context: ChatResponseContext):
            if not context.is_tool_calling:
                print("\n" + "*" * 20 + "开始工具调用" + "*" * 20)
                context.is_tool_calling = True

            print(f"\ntool_call chunk: {delta.tool_calls}")
            for tool_call in delta.tool_calls:
                if (
                    tool_call.function
                    and not tool_call.function.name
                    and not tool_call.function.arguments
                ):
                    break

                index = tool_call.index
                if len(context.tool_info) <= index:
                    context.tool_info.append(
                        {"id": "", "name": "", "arguments": "", "result": None}
                    )

                if tool_call.id:
                    context.tool_info[index]["id"] = tool_call.id

                if tool_call.function and tool_call.function.name:
                    context.tool_info[index]["name"] += tool_call.function.name

                if tool_call.function and tool_call.function.arguments:
                    context.tool_info[index][
                        "arguments"
                    ] += tool_call.function.arguments

                if is_valid_json(context.tool_info[index]["arguments"]):
                    tool_name = context.tool_info[index]["name"]
                    tool_args = json.loads(context.tool_info[index]["arguments"])
                    tool_task = asyncio.create_task(
                        self.call_tool(tool_name, tool_args)
                    )
                    context.tool_call_tasks.append(tool_task)

        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                process_reasoning_chunk(delta, context)

            else:
                if delta.content is not None and delta.content != "":
                    process_answer_chunk(delta, context)

                if delta.tool_calls is not None:
                    process_tool_call_chunk(delta, context)

        if context.tool_call_tasks:
            print("\n" + "*" * 20 + "工具调用结果" + "*" * 20)
            for index, result in enumerate(
                await asyncio.gather(*context.tool_call_tasks, return_exceptions=True)
            ):
                print(f"tool_call_result: {result}")
                if isinstance(result, Exception):
                    context.tool_info[index]["result"] = {
                        "content": [
                            {
                                "text": f"Error calling tool {context.tool_info[index]['name']}: {result}"
                            }
                        ],
                        "isError": True,
                    }
                else:
                    context.tool_info[index]["result"] = result

        return {
            "answer_content": context.answer_content,
            "reasoning_content": context.reasoning_content,
            "tool_info": context.tool_info,
        }

    async def get_response_2(self, messages):
        final_text = []

        round = 1
        while True:
            print(
                f"{'=' * 20} Assistant round {round} {'=' * 20}",
            )
            round += 1

            response = await self.get_chat_completion(messages)

            result = await self.process_streamed_response(response)

            answer_content = result["answer_content"]
            tool_info = result["tool_info"]

            assistant_msg_record = {
                "role": "assistant",
                "content": answer_content,
            }
            tool_result_records = []
            for index in range(len(tool_info)):
                if not assistant_msg_record.get("tool_calls"):
                    assistant_msg_record["tool_calls"] = []

                info = tool_info[index]

                tool_call_record = {
                    "id": info["id"],
                    "function": {
                        "name": info["name"],
                        "arguments": info["arguments"],
                    },
                    "index": index,
                    "type": "function",
                }
                assistant_msg_record["tool_calls"].append(tool_call_record)
                tool_result_records.append(
                    {
                        "content": str(info["result"]),
                        "role": "tool",
                        "tool_call_id": info["id"],
                    }
                )

            messages.append(assistant_msg_record)

            # if there is no tool result, it means the tool calling is done
            # break the loop and return the final text
            # otherwise, continue the chat loop to get the next round of response
            if tool_result_records:
                messages.extend(tool_result_records)
            else:
                break

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
