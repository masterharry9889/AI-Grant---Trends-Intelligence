"""
Grant_agent.py — Agent orchestration for GrantBot
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from prompts import REACT_AGENT_PROMPT

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import Tool
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

DATA_DIR = Path.cwd()
PERSIST_DIRECTORY = DATA_DIR / "grant_db"
CHAT_HISTORY_DB = "sqlite:///chat_history.db"

CATEGORY_MAP = {
    "government": "Search Indian government grants from myscheme.gov.in and similar.",
    "international": "Search international tenders from World Bank, UNDP, EU, OECD.",
    "startup": "Search startup funding from YC, Techstars, Google for Startups.",
    "research": "Search research funding from NSF, NIH, DARPA.",
    "scholarship": "Search scholarships from IIT, IIM, NIT.",
    "conference": "Search conference funding from ACM, IEEE, Google.",
    "education": "Search education-related grants and scholarships from Coursera, edX, Khan Academy.",
}


def get_loader(url: str, max_depth: int = 3) -> RecursiveUrlLoader:
    """Create a web page loader for grant and funding sources."""
    return RecursiveUrlLoader(url=url, max_depth=max_depth)


def tag_documents(documents: list[Any], category: str, source_url: str) -> list[Any]:
    """Attach source metadata to documents before indexing."""
    for document in documents:
        document.metadata["category"] = category
        document.metadata["source_url"] = source_url
        document.metadata["last_scraped"] = datetime.utcnow().isoformat()
    return documents


def build_vectorstore(raw_documents: list[Any], persist_directory: Path = PERSIST_DIRECTORY) -> Chroma:
    """Build or restore a Chroma vector store from raw documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2"),
        persist_directory=str(persist_directory),
        collection_name="grants_tenders",
    )
    return vectorstore


def build_retriever_tool(name: str, description: str, retriever: Any) -> Tool:
    """Wrap a retriever in a tool that the agent can call."""

    def search(query: str) -> str:
        if hasattr(retriever, "get_relevant_documents"):
            documents = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "_get_relevant_documents"):
            documents = retriever._get_relevant_documents(query, run_manager=None)
        else:
            raise AttributeError(
                "Retriever does not support get_relevant_documents or _get_relevant_documents"
            )
        return "\n\n".join(document.page_content for document in documents)

    return Tool(name=name, description=description, func=search)


def build_tools(vectorstore: Chroma) -> list[Tool]:
    """Build the set of tools used by GrantBot."""
    tools = []
    for category, description in CATEGORY_MAP.items():
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"category": category}})
        tools.append(build_retriever_tool(name=f"search_{category}", description=description, retriever=retriever))
    return tools


def create_agent(vectorstore: Chroma, model: ChatGroq | None = None) -> AgentExecutor:
    """Create the React agent using the prompt defined in prompts.py."""
    if model is None:
        model = ChatGroq(model_name="llama-3.1-8b-instant")

    tools = build_tools(vectorstore)
    agent = create_react_agent(llm=model, tools=tools, prompt=REACT_AGENT_PROMPT)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def load_chat_history(session_id: str = "default_session") -> SQLChatMessageHistory:
    """Load or create a local chat history store."""
    return SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_DB)


def main() -> None:
    """Demonstrate how to construct the GrantBot agent."""
    print("GrantBot agent module loaded.")
    print("To use the agent, build a Chroma vectorstore from your documents and call create_agent(...).")


if __name__ == "__main__":
    main()
