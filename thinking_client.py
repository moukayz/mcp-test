import random
from openai import AsyncOpenAI, AsyncStream,OpenAI
from openai.types.chat import ChatCompletionChunk
import os
from dotenv import load_dotenv
import asyncio
import json
load_dotenv()

tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 无需参数
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]  # 必填参数
            }
        }
    }
]

async def dummy_tool_call(tool_name, tool_args):
    print(f"\nStart tool call: {tool_name} with arguments: {tool_args}")
    await asyncio.sleep(2)
    print(f"Finish tool call: {tool_name} with arguments: {tool_args}")
    result = random.choice(["天气晴朗", "天气多云", "天气下雨"])
    return {"content": result, "isError": False}

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

async def main():
    # 初始化OpenAI客户端
    client = AsyncOpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key = os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复

    messages = []
    conversation_idx = 1
    is_tool_streaming = False
    while True:
        tool_info = []
        tool_call_task = []
        is_answering = False   # 判断是否结束思考过程并开始回复
        tool_call_result = []

        if not is_tool_streaming:
            print(f"="*20+f"第{conversation_idx}轮对话"+"="*20)
            conversation_idx += 1
            user_msg = {"role": "user", "content": input("请输入你的消息：")}
            messages.append(user_msg)

        is_tool_streaming = False

            # 创建聊天完成请求
        completion : AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            # 您可以按需更换为其它深度思考模型
            model="qwen-plus-2025-04-28",
            messages=messages,
            # enable_thinking 参数开启思考过程，QwQ 与 DeepSeek-R1 模型总会进行思考，不支持该参数
            extra_body={"enable_thinking": True, "thinking_budget": 100},
            stream=True,
            tools=tools,
            parallel_tool_calls=True,
            # stream_options={
            #     "include_usage": True
            # }
        )
        print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
        chunk_index = 0
        async for chunk in completion:
            chunk_index += 1
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content

                if delta.tool_calls is not None:
                    print("*"*20+"UNEXPECTED TOOL CALLS"+"*"*20)
                    print(delta.tool_calls)
                    print("*"*50)

            else:
                # 开始回复
                if delta.content is not None and delta.content != "":
                    if not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True

                    # 打印回复过程
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content

                # 处理工具调用信息（支持并行工具调用）
                finish_tool_call_stream = False
                if delta.tool_calls is not None:
                    if not is_tool_streaming:
                        print("\n" + "*"*20+"开始工具调用"+"*"*20)
                        is_tool_streaming = True

                    print(f"\n[chunk_index: {chunk_index}] current tool_call chunk: {delta.tool_calls}")
                    for tool_call in delta.tool_calls:
                        if tool_call.function and not tool_call.function.name and not tool_call.function.arguments:
                            finish_tool_call_stream = True
                            break

                        # print(f"[chunk_index: {chunk_index}] current tool_call chunk: {tool_call}")
                        index = tool_call.index  # 工具调用索引，用于并行调用
                        
                        # 动态扩展工具信息存储列表
                        while len(tool_info) <= index:
                            tool_info.append({
                                'id': '',
                                'name': '',
                                'arguments': ''
                            })
                        
                        # 收集工具调用ID（用于后续函数调用）
                        if tool_call.id:
                            tool_info[index]['id'] = tool_call.id
                        
                        # 收集函数名称（用于后续路由到具体函数）
                        if tool_call.function and tool_call.function.name:
                            tool_info[index]['name'] = tool_info[index].get('name', '') + tool_call.function.name
                        
                        # 收集函数参数（JSON字符串格式，需要后续解析）
                        if tool_call.function and tool_call.function.arguments:
                            tool_info[index]['arguments'] = tool_info[index].get('arguments', '') + tool_call.function.arguments

                        if is_valid_json(tool_info[index]['arguments']):
                            tool_call_task.append(dummy_tool_call(tool_info[index]['name'], tool_info[index]['arguments']))

                    # print(f"\n"+"="*19+"工具调用信息"+"="*19)
                    # print(tool_info)
                    # print(f"="*50)

        if is_tool_streaming:
            print(f"finish_tool_call_stream: {finish_tool_call_stream}")
            tool_call_result = await asyncio.gather(*tool_call_task)
            print(f"tool_call_result: {tool_call_result}")

        # 将模型回复的content添加到上下文中

        assistant_msg_record = {
            "role": "assistant",
            "content": answer_content,
        }
        for index in range(len(tool_info)):
            if not assistant_msg_record.get("tool_calls"):
                assistant_msg_record["tool_calls"] = []
            tool_call_record = {
                    "id": tool_info[index]['id'],
                    "function": {
                        "name": tool_info[index]['name'],
                        "arguments": tool_info[index]['arguments'],

                    },
                    "index": index,
                    "type": "function"
                }
            assistant_msg_record["tool_calls"].append(tool_call_record)

        messages.append(assistant_msg_record)

        for idx, result in enumerate(tool_call_result):
            tool_call_record = {
                "content": str(result),
                "role": "tool",
                "tool_call_id": tool_info[idx]['id'],
            }
            messages.append(tool_call_record)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())