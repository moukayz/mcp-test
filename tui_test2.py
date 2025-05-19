import asyncio
from typing import AsyncGenerator

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout, HSplit, VSplit
from prompt_toolkit.widgets import TextArea, Frame


async def processing_function(user_input: str) -> AsyncGenerator[str, None]:
    """
    Simulate an async processing that yields chunks of text.
    """
    for word in user_input.split():
        await asyncio.sleep(0.5)  # simulate I/O or compute
        yield word


def main() -> None:
    # --- Create the three areas ---
    input_area: TextArea = TextArea(
        height=1,
        prompt='> ',
        multiline=False,
        wrap_lines=False
    )

    output_ok: TextArea = TextArea(
        text='OK output:\n',
        read_only=True,
        scrollbar=True,
        wrap_lines=True
    )
    output_err: TextArea = TextArea(
        text='Error output:\n',
        read_only=True,
        scrollbar=True,
        wrap_lines=True
    )

    framed_input: Frame = Frame(input_area, title='Input')
    framed_ok: Frame    = Frame(output_ok,  title='Result')
    framed_err: Frame   = Frame(output_err, title='Errors')

    # --- Key bindings ---
    kb: KeyBindings = KeyBindings()

    @kb.add('enter')
    def _(event: KeyPressEvent) -> None:
        """
        Handler for Enter key: grab input, clear it, and schedule processing.
        """
        user_text: str = input_area.text.strip()
        if not user_text:
            return

        if user_text == "quit":
            event.app.exit()

        # Clear the input box immediately
        input_area.text = ''

        # Schedule the async processing in the background
        event.app.create_background_task(route_and_display(user_text))

    # --- Coroutine to drive the async generator and update UI ---
    async def route_and_display(text: str) -> None:
        """
        Consume the async generator and append each chunk to the proper pane.
        """
        async for chunk in processing_function(text):
            # Route by some condition (here: digits → error pane)
            target: TextArea = (
                output_err if any(c.isdigit() for c in chunk)
                else output_ok
            )
            target.buffer.text += chunk

            # Append chunk to buffer (preserves scroll position)
            # target.buffer.insert_text(chunk + '\n')

            # Request a repaint so it appears immediately
            app.invalidate()

    # --- Layout ---
    root = HSplit([
        framed_input,
        VSplit([framed_ok, framed_err], padding=1),
    ])

    # --- Build and run ---
    app: Application = Application(
        layout=Layout(root),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
    )
    app.run()


if __name__ == '__main__':
    main()
