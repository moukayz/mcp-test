from abc import abstractmethod
from typing import Any
from rich import print as rprint, logging as rich_logging
import asyncio
from dataclasses import dataclass, field
import json
import os
import logging

from contextlib import AsyncExitStack

from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

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
            # tools=tools,
            # tool_choice="auto",
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
            model="doubao-1-5-thinking-pro-250415",
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

SERVER_CONFIG_FILE = ".server_config.json"

@dataclass
class AssistantResponseChunk:
    type: str
    content: str | dict 


class LLMClient:
    def __init__(self):
        self.qwenClient = QwenModel()
        # self.doubaoClient = DoubaoModel()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    async def get_chat_completion(self, messages):
        return await self.qwenClient.get_chat_completion(messages, [])

    async def process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk]
    ):
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                yield AssistantResponseChunk(
                    type="thinking", content=delta.reasoning_content
                )

            else:
                if delta.content is not None and delta.content != "":
                    yield AssistantResponseChunk(type="answer", content=delta.content)

    async def get_assistant_response(self, messages):

        while True:
            response = await self.get_chat_completion(messages)

            result = self.process_streamed_response(response)

            answer_content = ""
            reasoning_content = ""

            async for chunk in result:
                if chunk.type == "answer":
                    answer_content += chunk.content
                    yield chunk
                elif chunk.type == "thinking":
                    reasoning_content += chunk.content
                    yield chunk


            assistant_msg_record = {
                "role": "assistant",
                "content": answer_content,
            }
            messages.append(assistant_msg_record)

            break


class ChatApp:

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
                                rprint( "[bold yellow]" + "\n" + "=" * 20 + "完整回复" + "=" * 20 + "[/bold yellow]\n")
                            rprint(chunk.content, end="", flush=True)
                        elif chunk.type == "thinking":
                            if current_type != "thinking":
                                current_type = "thinking"
                                rprint( "[bold cyan]" + "\n" + "=" * 20 + "思考过程" + "=" * 20 + "[/bold cyan]\n")
                            rprint(chunk.content, end="", flush=True)
                        

                except KeyboardInterrupt:
                    break


def load_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()


if __name__ == "__main__":

    asyncio.run(ChatApp().chat_loop())