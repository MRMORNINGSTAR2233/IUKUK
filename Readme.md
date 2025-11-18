# IUKUK - AI Research Projects

This repository contains experimental AI and machine learning research projects.

## 🎯 Featured Project: MCP-Gymnasium

**Turn ANY MCP server into a Reinforcement Learning environment**

A universal adapter that bridges the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) with [Gymnasium](https://gymnasium.farama.org/), enabling autonomous agents to interact with any tool/API through a standard RL interface.

[→ See full documentation in `/mcp/`](./mcp/README.md)

### Quick Demo

```bash
cd mcp
source ../.venv/bin/activate
python search_agent.py
```

This will run an autonomous agent that:
1. Searches for cryptocurrency prices
2. Saves results to a file
3. Uses LLM-based reasoning and rewards

## 📁 Repository Structure

```
├── mcp/                      # 🌟 MCP-Gymnasium implementation
│   ├── mcp_gym.py           # Universal Gym adapter
│   ├── ddg_server.py        # Example MCP server (search + file ops)
│   ├── search_agent.py      # Autonomous LLM agent demo
│   ├── rewards.py           # Reward functions (keyword & LLM-based)
│   └── README.md            # Full documentation
│
├── speedtest-llm/           # LLM inference benchmarks
│   └── specdec*.py          # Speculative decoding experiments
│
├── paper-implementation/    # Research paper reproductions
│
└── .venv/                   # Python virtual environment
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.11+
python --version

# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MRMORNINGSTAR2233/IUKUK.git
cd IUKUK

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install mcp gymnasium groq ddgs python-dotenv colorama requests
```

### Environment Setup

Create `mcp/.env`:
```bash
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key  # Optional
```

## 🎓 What Can You Build?

### 1. Autonomous Research Agent
```python
# Agent searches web, analyzes data, saves reports
MISSION = "Research topic X and create summary report"
```

### 2. File System Agent
```python
# Agent navigates directories, reads/writes files
MISSION = "Find all TODO comments and create task list"
```

### 3. API Integration Agent
```python
# Agent calls APIs, processes responses, takes actions
MISSION = "Monitor system metrics and alert if threshold exceeded"
```

### 4. Database Agent
```python
# Agent queries database, analyzes patterns, generates insights
MISSION = "Find anomalies in transaction data"
```

## 🧠 Key Features

- ✅ **Universal Adapter**: Works with ANY MCP server
- ✅ **LLM-Powered**: Uses Gemini/Groq for intelligent decision-making
- ✅ **Reward Engineering**: Keyword-based or LLM-based evaluation
- ✅ **Multi-Step Missions**: Complex task completion
- ✅ **Async Bridge**: Clean sync/async integration
- ✅ **Production Ready**: Error handling, timeouts, fallbacks

## 📊 Example Results

```
🎯 Mission: Find Bitcoin and Ethereum prices, save to file

⚡ Step 1: web_search("Bitcoin price") 
   → Found: $95,355 USD | Reward: +2.4

⚡ Step 2: web_search("Ethereum price")
   → Found: $3,107 USD | Reward: +2.4

⚡ Step 3: save_report("crypto_report.txt", summary)
   → Saved 104 chars | Reward: +9.9

🎉 SUCCESS in 3 steps (~8 seconds)
```

## 🔬 Research Applications

1. **Tool-Use Learning**: Train agents to use tools effectively
2. **Task Planning**: Multi-step reasoning and execution
3. **Reward Shaping**: Compare keyword vs LLM-based rewards
4. **Transfer Learning**: Train on one MCP server, test on others
5. **Agent Evaluation**: Benchmark different LLM architectures

## 🛠️ Technical Highlights

### Async/Sync Bridge
Novel solution to connect async MCP with sync Gymnasium:
```python
# Background event loop manages MCP session
# Main thread makes sync gym calls
# Communication via thread-safe futures
```

### Smart Rewards
LLM judges if observation satisfies mission:
```python
critic = LLMReward(
    mission="Find X and save to file",
    provider="groq"
)
# Returns 0-10 based on how well mission is completed
```

## 📚 Documentation

- [MCP-Gymnasium Full Docs](./mcp/README.md)
- [API Reference](./mcp/mcp_gym.py)
- [Example Servers](./mcp/ddg_server.py)
- [Agent Examples](./mcp/search_agent.py)

## 🤝 Contributing

This is a research project. Contributions welcome!

Areas of interest:
- New MCP server implementations
- RL algorithm integrations
- Benchmarking and evaluation
- Documentation improvements

## 📝 License

See [LICENSE](./LICENSE) file.

## 🙏 Acknowledgments

Built on top of:
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [FastMCP](https://github.com/jlowin/fastmcp)

---

**Status**: ✅ Working | 🧪 Experimental | 📚 Research

*Last Updated: November 2025*



