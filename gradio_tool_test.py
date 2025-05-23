import gradio as gr
from gradio import ChatMessage
import time
import asyncio

async def impl(history):
    history.append(
        ChatMessage(
            role="user", content="What is the weather in San Francisco right now?"
        )
    )
    yield history
    await asyncio.sleep(1)
    history.append(
        ChatMessage(
            role="assistant",
            content="In order to find the current weather in San Francisco, I will need to use my weather tool.",
        )
    )
    yield history
    await asyncio.sleep(1)

    history.append(
        ChatMessage(
            role="assistant",
            content="...",
            metadata={"title": "Using tools", "status": "pending"},
        )
    )
    yield history

    await asyncio.sleep(3)
    history[-1] = ChatMessage(
            role="assistant",
            content="API Error when connecting to weather service.",
            metadata={"title": "💥 Error using tool 'Weather'", "status": "done"},
        )
    
    yield history
    await asyncio.sleep(1)

    history.append(
        ChatMessage(
            role="assistant",
            content="I will try again",
        )
    )
    yield history
    await asyncio.sleep(1)

    history.append(
        ChatMessage(
            role="assistant",
            content="Weather 72 degrees Fahrenheit with 20% chance of rain.",
            metadata={"title": "🛠️ Used tool 'Weather'"},
        )
    )
    yield history
    await asyncio.sleep(1)

    history.append(
        ChatMessage(
            role="assistant",
            content="Now that the API succeeded I can complete my task.",
        )
    )
    yield history
    await asyncio.sleep(1)

    history.append(
        ChatMessage(
            role="assistant",
            content="It's a sunny day in San Francisco with a current temperature of 72 degrees Fahrenheit and a 20% chance of rain. Enjoy the weather!",
        )
    )
    yield history

async def func(history):
    await asyncio.sleep(2)
    return impl(history)

async def generate_response(history):
    async for response in await func(history):
        yield response

def like(evt: gr.LikeData):
    print("User liked the response")
    print(evt.index, evt.liked, evt.value)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", height=500, show_copy_button=True)
    button = gr.Button("Get San Francisco Weather")
    button.click(generate_response, chatbot, chatbot)
    chatbot.like(like)

if __name__ == "__main__":
    demo.launch()