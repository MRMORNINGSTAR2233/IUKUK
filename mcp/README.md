# MCP-Gymnasium: Universal Adapter for MCP Servers

> **Turn ANY MCP server into a Reinforcement Learning environment**

MCP-Gymnasium creates a universal adapter that turns any [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server into a standard [Gymnasium](https://gymnasium.farama.org/) environment, enabling you to run RL algorithms on it.

## 🎯 What is This?

This project implements a bridge between:
- **MCP (Model Context Protocol)**: A standard for LLM-tool interaction
- **Gymnasium**: The standard RL environment interface (successor to OpenAI Gym)

**Result**: Any MCP server (file systems, APIs, databases, web search, etc.) can be used as a training ground for autonomous agents!

## 🏗️ Architecture

```
┌─────────────────┐
│  Your RL Agent  │  (LLM-based or traditional RL)
│   (Gemini/Groq) │
└────────┬────────┘
         │
         ├─── Observations (tool results)
         └─── Actions (tool calls as JSON)
         │
┌────────▼────────┐
│   MCPEnv        │  Universal Gym Adapter
│  (mcp_gym.py)   │
└────────┬────────┘
         │
         ├─── list_tools()
         ├─── call_tool(name, args)
         │
┌────────▼────────┐
│   MCP Server    │  (ANY server: search, files, APIs...)
│ (ddg_server.py) │
└─────────────────┘
```

## 📁 Project Structure

```
mcp/
├── mcp_gym.py          # Universal Gym adapter (MCPEnv class)
├── ddg_server.py       # Example: DuckDuckGo search + file saving
├── search_agent.py     # Example: Autonomous LLM agent
├── rewards.py          # Reward functions (keyword & LLM-based)
├── .env               # API keys
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install mcp gymnasium groq ddgs python-dotenv colorama requests

# Or using pip
pip install mcp gymnasium groq ddgs python-dotenv colorama requests
```

### 2. Set Up Environment

Create `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Optional
```

### 3. Run the Example Agent

```bash
python search_agent.py
```

This will:
1. Search for Bitcoin & Ethereum prices
2. Save a report to `crypto_report.txt`
3. Demonstrate multi-step autonomous behavior

## 🎓 How to Use

### Creating a Custom MCP Server

```python
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("my-custom-server")

@mcp.tool()
def my_custom_tool(param1: str, param2: int) -> str:
    """Your tool description here."""
    # Your logic
    return json.dumps({"result": "success"})

if __name__ == "__main__":
    mcp.run(transport='stdio')
```

### Using Any MCP Server as a Gym Environment

```python
from mcp_gym import MCPEnv
from mcp import StdioServerParameters
from rewards import KeywordReward
import sys

# Configure your MCP server
server_params = StdioServerParameters(
    command=sys.executable,
    args=["path/to/your_server.py"],
    env=dict(os.environ)
)

# Define success criteria
reward_fn = KeywordReward(
    target_keywords=["success", "completed"],
    success_reward=10.0
)

# Create the Gym environment
env = MCPEnv(
    server_params=server_params,
    reward_function=reward_fn,
    max_steps=10
)

# Use like any Gym environment
obs, info = env.reset()
done = False

while not done:
    # Your agent decides action (JSON format)
    action = '{"name": "tool_name", "args": {"param": "value"}}'
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## 🧠 Reward Functions

### 1. Keyword-Based (Simple)

```python
from rewards import KeywordReward

critic = KeywordReward(
    target_keywords=["success", "found", "completed"],
    success_reward=10.0,
    step_penalty=-0.1
)
```

### 2. LLM-Based (Smart)

```python
from rewards import LLMReward

critic = LLMReward(
    mission_description="Find crypto prices and save to file",
    api_key="your_api_key",
    provider="groq"  # or "gemini"
)
```

The LLM critic evaluates if the observation truly satisfies the mission requirements.

## 🎯 Example Missions

### Mission 1: Web Search
```python
MISSION = "Find who won Super Bowl 2024 and the score"
SUCCESS_KEYWORDS = ["Chiefs", "49ers", "25", "22"]
```

### Mission 2: Multi-Step Task
```python
MISSION = "Find Bitcoin and Ethereum prices, save to crypto_report.txt"
SUCCESS_KEYWORDS = ["saved", "SUCCESS", "crypto_report.txt"]
```

### Mission 3: File Operations
```python
MISSION = "Save a test message 'Hello World' to test.txt"
SUCCESS_KEYWORDS = ["saved", "SUCCESS", "test.txt"]
```

## 🛠️ Technical Details

### MCPEnv Class Features

- ✅ **Async Bridge**: Runs MCP session in background thread
- ✅ **Persistent Session**: Single task manages entire lifecycle
- ✅ **Tool Discovery**: Auto-detects all available tools
- ✅ **Action Space**: Text-based JSON commands
- ✅ **Observation Space**: Text-based tool responses
- ✅ **Flexible Rewards**: Pluggable reward functions

### Key Innovation: Async/Sync Bridge

The core challenge: Gymnasium is synchronous, but MCP is async. Solution:

```python
class AsyncBridge:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_background_loop, daemon=True)
        self.thread.start()
    
    def _start_background_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
```

This allows sync Gym steps to trigger async MCP operations cleanly.

## 🧪 Current Status

**Working Features:**
- ✅ MCP server connection & tool discovery
- ✅ Gym-compliant environment interface
- ✅ LLM-based autonomous agents (Gemini/Groq)
- ✅ Keyword & LLM-based reward functions
- ✅ Multi-step mission execution
- ✅ File operations (save_report tool)
- ✅ Web search (DuckDuckGo integration)
- ✅ Graceful error handling & fallbacks

**Recent Fixes:**
- ✅ Fixed circular import in ddg_server.py
- ✅ Updated to use `ddgs` package (new DuckDuckGo client)
- ✅ Switched Gemini to REST API (avoids DNS/gRPC issues)
- ✅ Added Groq support with `openai/gpt-oss-120b` model
- ✅ Improved JSON extraction with regex
- ✅ Stricter LLM critic for multi-step missions

## 📊 Results

Example run (3 steps, ~8 seconds):
```
🎯 Mission: Find Bitcoin and Ethereum prices, save to crypto_report.txt

Step 1: web_search("Bitcoin price USD") → Reward: 2.4
Step 2: web_search("Ethereum price USD") → Reward: 2.4
Step 3: save_report("crypto_report.txt", "...") → Reward: 9.9

🎉 SUCCESS! Mission completed.
```

## 🔮 Future Enhancements

- [ ] Support for more MCP servers (filesystem, databases, APIs)
- [ ] Traditional RL algorithms (DQN, PPO, A3C)
- [ ] Multi-agent environments
- [ ] Benchmarks & evaluation suite
- [ ] Integration with LangChain/CrewAI
- [ ] Visual observation support (images from MCP)

## 📚 Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)

## 🤝 Contributing

This is a research project exploring the intersection of:
- Tool-use by LLMs (MCP)
- Reinforcement Learning (Gymnasium)
- Autonomous agents

Feel free to experiment and extend!

## 📝 License

See LICENSE file in repository root.

---

**Built with ❤️ to make MCP servers accessible to RL algorithms**
