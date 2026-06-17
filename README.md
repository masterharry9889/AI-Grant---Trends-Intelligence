# AI Grant Trends Intelligence

An intelligent AI-powered agent system for researching, analyzing, and tracking AI grants and emerging trends in the AI funding landscape. This project combines agentic AI workflows with real-time trend analysis to provide actionable insights into grant opportunities and industry developments.

## Overview

**AI Grant Trends Intelligence** is a multi-agent AI system designed to:
- Autonomously research available AI grants and funding opportunities
- Analyze emerging trends in AI research and development
- Synthesize complex grant eligibility requirements and timelines
- Maintain persistent conversation history for context-aware interactions
- Provide intelligent recommendations and insights based on market data

This system leverages modern agentic AI frameworks to orchestrate multiple tools and data sources, enabling comprehensive grant intelligence gathering and trend analysis.

## Features

✨ **Core Capabilities**
- **Grant Discovery & Analysis**: Automated research of AI grants across multiple funding bodies
- **Trend Intelligence**: Real-time analysis of emerging trends in AI research and funding
- **Intelligent Agent Workflows**: Multi-step reasoning and tool orchestration for complex queries
- **Chat History Management**: Persistent conversation tracking with SQLite for context retention
- **Dynamic Prompting**: Flexible prompt engineering for specialized grant and trend analysis
- **Environment Management**: Secure API key and configuration handling via `.env` files

## Tech Stack

- **Language**: Python 3.x
- **AI Framework**: LLM-based agents (LangGraph/CrewAI compatible)
- **Database**: SQLite (for chat history and grant metadata)
- **Configuration**: Python-dotenv for environment management
- **Agent Orchestration**: Multi-step agentic workflows with tool integration

## Project Structure

```
AI-Grant---Trends-Intelligence/
├── Grant_agent.py          # Core grant research and analysis agent
├── run_grant_agent.py      # Entry point and agent runner
├── prompts.py              # Prompt templates for grant and trend analysis
├── grant_db/               # Grant database and metadata storage
├── chat_history.db         # SQLite database for conversation history
├── .env                    # Environment variables (API keys, config)
└── __pycache__/            # Python cache files
```

### Key Components

| File | Purpose |
|------|---------|
| `Grant_agent.py` | Implements the core grant research agent with tool definitions and reasoning logic |
| `run_grant_agent.py` | Main entry point; initializes and executes the agent with user queries |
| `prompts.py` | System prompts and templates for grant research and trend analysis tasks |
| `grant_db/` | Local database directory for storing grant data, metadata, and search results |
| `chat_history.db` | SQLite database maintaining multi-turn conversation history for context |

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/masterharry9889/AI-Grant---Trends-Intelligence.git
   cd AI-Grant---Trends-Intelligence
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: If `requirements.txt` is not present, install core dependencies:
   > ```bash
   > pip install python-dotenv langchain langchain-community langgraph anthropic
   > ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   # Add other required API keys and configuration
   ```

## Usage

### Running the Grant Research Agent

```bash
python run_grant_agent.py
```

This will start the agent and accept queries. Example interactions:

```
Query: What are the latest AI research grants available in 2025?
Query: Analyze trends in NLP funding across government and private sectors
Query: Find grants for sustainable AI projects
```

### Basic Agent Workflow

```python
from Grant_agent import create_grant_agent

# Initialize the agent
agent = create_grant_agent()

# Run a query
response = agent.run("What are emerging trends in AI grant funding?")
print(response)
```

### Accessing Chat History

The agent maintains conversation history in `chat_history.db`:

```python
import sqlite3

conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)
conn.close()
```

## Configuration

### Environment Variables

Create a `.env` file with the following structure:

```env
# API Configuration
ANTHROPIC_API_KEY=your_anthropic_key
# Add additional API keys as needed for external grant databases

# Database Configuration
GRANT_DB_PATH=./grant_db
CHAT_HISTORY_DB=./chat_history.db

# Agent Configuration
MAX_ITERATIONS=10
TEMPERATURE=0.7
```

### Custom Prompts

Modify `prompts.py` to customize agent behavior:

```python
GRANT_RESEARCH_PROMPT = """
You are an expert grant researcher specializing in AI funding opportunities.
Your task is to:
1. Identify relevant grants based on criteria
2. Analyze eligibility requirements
3. Synthesize trends across funding bodies
...
"""
```

## Architecture

### Multi-Agent Workflow

```
User Query
    ↓
Query Router Agent
    ├→ Grant Research Agent
    │   ├→ Web Search Tool
    │   ├→ Grant Database Tool
    │   └→ Analysis Tool
    │
    ├→ Trend Analysis Agent
    │   ├→ Data Aggregation Tool
    │   ├→ Pattern Recognition Tool
    │   └→ Insight Generation Tool
    │
    └→ Response Synthesis
        └→ Chat History Update
```

## Key Features in Detail

### 1. Agentic Grant Research
- Autonomous tool orchestration for grant discovery
- Multi-step reasoning for complex eligibility analysis
- Comparative analysis across funding bodies

### 2. Trend Intelligence
- Emerging pattern detection in AI funding
- Time-series analysis of grant trends
- Sector-specific insights

### 3. Persistent Context
- SQLite-backed conversation history
- Context retention across sessions
- Query enrichment with historical data

### 4. Flexible Prompting
- Dynamic prompt construction based on query type
- Specialized templates for different grant categories
- Few-shot examples for improved reasoning

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Tools

1. Define the tool in `Grant_agent.py`:
```python
def grant_search_tool(query: str) -> str:
    """Search grant databases"""
    # Implementation
    return results
```

2. Register with the agent:
```python
agent.add_tool(grant_search_tool)
```

3. Update prompts in `prompts.py` to reference the new tool

### Code Style

Follow PEP 8 conventions:
```bash
black .
flake8 .
```

## Performance Considerations

- **Chat History**: Database queries are indexed on timestamp; consider archiving old chats
- **Agent Iterations**: Configure `MAX_ITERATIONS` based on query complexity
- **API Calls**: Use response caching to minimize redundant API calls
- **Database Size**: Periodically vacuum `chat_history.db` for optimal performance

## Troubleshooting

### Agent Timeouts
- Increase `MAX_ITERATIONS` in `.env`
- Check API rate limits
- Verify network connectivity

### Database Errors
```bash
# Reset chat history (backup first!)
rm chat_history.db
python run_grant_agent.py  # Will recreate the database
```

### Missing API Keys
- Verify `.env` file exists and is in the project root
- Check that `ANTHROPIC_API_KEY` is set correctly
- Ensure no trailing whitespace in `.env` values

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Areas
- New grant data sources and integrations
- Improved prompt engineering for better analysis
- Additional agent tools and capabilities
- Documentation and examples
- Performance optimizations

## Roadmap

- [ ] Integration with major grant databases (NSF, NSERC, Horizon Europe)
- [ ] Real-time grant notifications based on user preferences
- [ ] Advanced analytics dashboard for grant trends
- [ ] Multi-language support for international grants
- [ ] Personalized recommendation engine
- [ ] Grant application assistance and drafting tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

- **Author**: [@masterharry9889](https://github.com/masterharry9889)
- **Issues**: [GitHub Issues](https://github.com/masterharry9889/AI-Grant---Trends-Intelligence/issues)
- **Portfolio**: [vermaaniket.vercel.app](https://vermaaniket.vercel.app)
- **LinkedIn**: [linkedin.com/in/aniket-verma-2034a3294](https://linkedin.com/in/aniket-verma-2034a3294)

## Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) for agentic workflows
- Powered by [Anthropic Claude API](https://www.anthropic.com/)
- Inspired by the need for better grant discovery and trend analysis in AI research

---

**Note**: This project is actively maintained. For the latest updates and features, visit the [GitHub repository](https://github.com/masterharry9889/AI-Grant---Trends-Intelligence).
