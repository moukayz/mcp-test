import asyncio
from typing import AsyncGenerator

from textual.app import App, ComposeResult
from textual.widgets import Input, Log, Markdown
from textual.containers import Vertical, Horizontal


async def processing_function(user_input: str) -> AsyncGenerator[str, None]:
    """
    Example async generator: yields each word after a short delay.
    """
    for word in user_input.split():
        await asyncio.sleep(0.5)  # simulate I/O or computation
        yield word

EXAMPLE_MARKDOWN = """\
# Markdown Document

This is an example of Textual's `Markdown` widget.

## Features

Markdown syntax and extensions are supported.

- Typography *emphasis*, **strong**, `inline code` etc.
- Headers
- Lists (bullet and ordered)
- Syntax highlighted code blocks
- Tables!
"""

class TextualTUI(App):
    """
    A Textual application with:
      - Input at the top
      - Two Log panels side by side for routed output
    """

    def compose(self) -> ComposeResult:
        # Build layout: vertical stack with an input above a horizontal pair of logs.
        yield Vertical(
            Input(placeholder="Type something and press Enter", id="input"),
            Horizontal(
                Markdown(id="log_ok", name="result"),
                # Markdown( id="log_err", name="errors"),
                id="logs"
            ),
            id="root"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Fired when Enter is pressed in the Input widget.
        Clears the input and starts the async routing task.
        """
        user_input = event.value.strip()
        if not user_input:
            return

        if user_input == "quit":
            self.exit()

        # Clear the input field
        event.input.value = ""

        # Schedule background task—does not block the UI
        asyncio.create_task(self.route_and_display(EXAMPLE_MARKDOWN))

    async def route_and_display(self, text: str) -> None:
        """
        Consume the async generator and write each chunk to the appropriate Log.
        """
        # Note: using `async for` here to consume the generator properly
        async for chunk in processing_function(text):
            # Route based on a condition (digits → error pane)
            # if any(c.isdigit() for c in chunk):
            #     log: Markdown = self.query_one("#log_err", Markdown)
            # else:
            self.log.info(chunk)
            log: Markdown = self.query_one("#log_ok", Markdown)

            await log.update('' if log._markdown is None else log._markdown + chunk)

    # Optionally, you can override `on_mount` to set focus on the input:
    def on_mount(self) -> None:
        self.query_one(Input).focus()


if __name__ == "__main__":
    TextualTUI().run()
