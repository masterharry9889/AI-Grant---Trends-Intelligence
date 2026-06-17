# 🤖 AI Grant & Trends Intelligence — GrantBot

A LangChain-powered ReAct agent that discovers and evaluates **grants, tenders, fellowships, and funding opportunities** from across the web — covering Indian government schemes, international organizations, startup accelerators, research bodies, and more.

---

## 🌟 Overview

GrantBot is an intelligent conversational agent built using **LangChain**, **Chroma**, and **Groq (LLaMA 3.1)**. It ingests and indexes funding data from curated web sources, then answers natural language queries about grants with structured, verified, source-linked responses.

> **Example query:** *"Find grants for early-stage Indian startups working on clean energy or sustainability."*

GrantBot will search its vector knowledge base, reason through the most relevant results using the ReAct framework, and return a structured breakdown of opportunities with eligibility, deadlines, amounts, and source links.

---

## ✨ Features

- **Multi-category search** — government, international, startup, research, scholarship, conference, and education funding
- **ReAct agent reasoning** — step-by-step Thought → Action → Observation loops before generating a final answer
- **RAG pipeline** — web pages are scraped, chunked, embedded, and stored in a persistent Chroma vector store
- **Persistent chat history** — conversations are saved to a local SQLite database for multi-turn context
- **NVIDIA embeddings** — uses `nvidia/llama-nemotron-embed-1b-v2` for high-quality semantic search
- **Structured output** — every result is formatted with Title, Source, Type, Eligibility, Amount, Deadline, Key Requirements, and Link
- **Query classification & fallback** — classifies queries to pick the right tool; gracefully handles zero-result cases
- **Condense question rewriting** — rewrites follow-up questions into standalone retrieval queries using chat history

---

## 📁 Project Structure

```
AI-Grant---Trends-Intelligence/
│
├── Grant_agent.py        # Core agent: vectorstore builder, retriever tools, agent factory
├── run_grant_agent.py    # Runner: loads vectorstore and runs a sample agent query
├── prompts.py            # All prompt templates (system, ReAct, RAG, classifier, fallback)
├── grant_db/             # Persisted Chroma vector store
├── chat_history.db       # SQLite conversation history
└── .env                  # API keys (GROQ_API_KEY, NVIDIA_API_KEY)
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
CONDENSE_QUESTION_PROMPT   ← rewrites follow-ups using chat history
    │
    ▼
CLASSIFIER_PROMPT          ← picks the right retrieval tool
    │
    ▼
Chroma Retriever (tool)    ← fetches semantically relevant docs
    │
    ▼
RAG_RETRIEVAL_PROMPT       ← generates grounded answer from context
    │
    ▼
REACT_AGENT_PROMPT         ← wraps with persona + tool rules
    │
    ▼
NO_RESULT_PROMPT           ← fallback if retriever returns nothing
    │
    ▼
Structured Output
```

---

## 🔍 Data Sources

| Category | Sources |
|---|---|
| **Government (India)** | myscheme.gov.in, patenvue.com, infoapp.com, letmespread.com |
| **International** | worldbank.org, undp.org, ec.europa.eu, oecd.org, erasmus-plus.ec.europa.eu |
| **Startups** | ycombinator.com, techstars.com, startup.google.com, microsoft.com |
| **Grants & Tenders** | mastersgrant.com, youthatlas.com, paperswithcode.com |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/masterharry9889/AI-Grant---Trends-Intelligence.git
cd AI-Grant---Trends-Intelligence
```

### 2. Install Dependencies

```bash
pip install langchain langchain-community langchain-groq langchain-nvidia-ai-endpoints chromadb python-dotenv
```

### 3. Set Up Environment Variables

Create a local `.env` file in the project root only for development. Do not commit it.

```env
GROQ_API_KEY=your_groq_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Get your keys from:
- [Groq Console](https://console.groq.com/) — free tier available
- [NVIDIA AI Endpoints](https://build.nvidia.com/) — free API credits available

### 4. Build the Vector Store

Before running the agent, ingest a grant source and index it into Chroma. Use `ingest.py`:

```bash
python ingest.py --url https://www.myscheme.gov.in/ --category government --max-depth 3
```

You can also ingest additional sources for categories like `nabard`, `dst`, `startup_india`, and `icar`.

### 5. Run the Agent

```bash
python run_grant_agent.py
```

This will load the persisted vector store, create the GrantBot agent, and run a sample query:

```python
from Grant_agent import get_loader, tag_documents, build_vectorstore

# Example: index government grants
loader = get_loader("https://www.myscheme.gov.in/", max_depth=3)
docs = loader.load()
docs = tag_documents(docs, category="government", source_url="https://www.myscheme.gov.in/")

vectorstore = build_vectorstore(docs)
print("Vector store built and persisted to grant_db/")
```

### 5. Run the Agent

```bash
python run_grant_agent.py
```

### 6. Install from requirements

If you want to install the environment via the repository dependencies:

```bash
pip install -r requirements.txt
```

This will load the persisted vector store, create the GrantBot agent, and run a sample query:

> *"Find grants or funding opportunities for early-stage Indian startups working on clean energy or sustainability."*

---

## 🧩 Using the Agent Programmatically

```python
from Grant_agent import create_agent, load_chat_history
from run_grant_agent import load_vectorstore
from pathlib import Path

# Load persisted vectorstore
vectorstore = load_vectorstore(Path("grant_db"))

# Create the ReAct agent
agent = create_agent(vectorstore)

# Load chat history for multi-turn conversations
history = load_chat_history(session_id="my_session")

# Run a query
response = agent.invoke({
    "input": "What scholarships are available for Indian engineering students?",
    "chat_history": []
})

print(response["output"])
```

---

## 🛠️ Key Components

### `Grant_agent.py`

| Function | Description |
|---|---|
| `get_loader(url, max_depth)` | Creates a `RecursiveUrlLoader` for scraping grant sources |
| `tag_documents(docs, category, url)` | Attaches category and metadata before indexing |
| `build_vectorstore(docs)` | Chunks documents and builds/persists a Chroma collection |
| `build_retriever_tool(name, desc, retriever)` | Wraps a Chroma retriever as a LangChain Tool |
| `build_tools(vectorstore)` | Creates one retriever tool per funding category |
| `create_agent(vectorstore)` | Assembles the full ReAct AgentExecutor |
| `load_chat_history(session_id)` | Returns a SQLite-backed chat history store |

### `prompts.py`

| Prompt | Purpose |
|---|---|
| `SYSTEM_PROMPT` | GrantBot's identity, tool guidance, and output format rules |
| `REACT_AGENT_PROMPT` | Main agent reasoning template (Thought/Action/Observation) |
| `CONDENSE_QUESTION_PROMPT` | Rewrites follow-up questions into standalone retrieval queries |
| `RAG_RETRIEVAL_PROMPT` | Grounded answer generation from retrieved context |
| `CLASSIFIER_PROMPT` | Routes query to the correct tool category |
| `NO_RESULT_PROMPT` | Graceful fallback when retrieval returns nothing |

---

## 🧪 Example Queries

```
"What government grants are available for women-led NGOs in India?"
"Find international research funding from UNDP or World Bank for climate projects."
"Are there any startup accelerators for EdTech companies in 2025?"
"Show me IEEE or ACM conference funding opportunities for PhD students."
"What scholarships are available from IITs or NITs for postgraduate students?"
```

---

## 🔧 Configuration

You can modify the funding categories and their descriptions in `Grant_agent.py`:

```python
CATEGORY_MAP = {
    "government": "Search Indian government grants from myscheme.gov.in and similar.",
    "international": "Search international tenders from World Bank, UNDP, EU, OECD.",
    "startup": "Search startup funding from YC, Techstars, Google for Startups.",
    "research": "Search research funding from NSF, NIH, DARPA.",
    "scholarship": "Search scholarships from IIT, IIM, NIT.",
    "conference": "Search conference funding from ACM, IEEE, Google.",
    "education": "Search education-related grants from Coursera, edX, Khan Academy.",
}
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.1 8B Instant |
| Embeddings | NVIDIA — llama-nemotron-embed-1b-v2 |
| Vector Store | ChromaDB (persistent) |
| Agent Framework | LangChain ReAct Agent |
| Web Scraping | LangChain `RecursiveUrlLoader` |
| Chat Memory | SQLite via `SQLChatMessageHistory` |
| Environment | Python 3.10+ |

---

## 🔒 Important Notes

- The `.env` file is committed to this repo for reference — **replace with your own API keys before running**.
- The `chat_history.db` and `grant_db/` are local files and will grow as you index more sources and have more conversations.
- This project uses `langchain_classic` — ensure you have the correct version of LangChain installed that includes this module.

---

## 🗺️ Roadmap

- [ ] Streamlit UI for a browser-based chat interface
- [ ] Scheduled scraping to keep the vector store up to date
- [ ] Email alerts for newly indexed opportunities matching saved queries
- [ ] Support for PDF grant documents (e.g. RFP/RFQ attachments)
- [ ] LangGraph migration for more robust multi-agent orchestration

---

## 👤 Author

**Aniket Verma**
AI/ML Engineer | B.Tech CSE, AKTU
[GitHub](https://github.com/masterharry9889) · [LinkedIn](https://linkedin.com/in/aniket-verma-2034a3294) · [Portfolio](https://vermaaniket.vercel.app)

---

## 📄 License

This project is open-source. Feel free to fork, extend, and build on it.
