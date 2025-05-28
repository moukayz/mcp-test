# Config
1. create `.env` from `.env_example`, add your api keys
2. create `.server_config.json` from `.server_config_example.json`, modify servers and add your brave search api key

# Install `UV`
https://docs.astral.sh/uv/getting-started/installation/
 
# Run client app
uv run cli_app.py

# Run gradio web app
uv run gradio_app.py

# Test any server
uv run mcp dev ./server/weather.py