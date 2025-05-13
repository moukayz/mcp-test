import gradio as gr
import random
import time
import asyncio

async def init():
    print("Initialization started", time.time())
    await asyncio.sleep(2)
    print("Initialization completed", time.time())

async def uninit():
    print("Uninitialization started", time.time())
    await asyncio.sleep(1)
    print("Uninitialization completed", time.time())

with gr.Blocks() as demo:
    demo.load(init)
    demo.unload(uninit)

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    async def bot(history: list):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history.append({"role": "assistant", "content": ""})
        for character in bot_message:
            history[-1]['content'] += character
            await asyncio.sleep(0.5)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()