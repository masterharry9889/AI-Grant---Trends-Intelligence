"""
Grant_agent.py — Agent orchestration for GrantBot
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from prompts import CLASSIFIER_PROMPT, PROPOSAL_PROMPT, REACT_AGENT_PROMPT

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_classic.chains.llm import LLMChain
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
    "nabard": "Search NABARD agricultural, rural development, and micro-enterprise funding in India.",
    "dst": "Search Department of Science and Technology funding programs for Indian research and innovation.",
    "startup_india": "Search Startup India grants, bootcamps, incubators, and funding initiatives.",
    "icar": "Search ICAR agricultural research, food systems, and agritech funding opportunities.",
}


def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {key}. "
            "Set it in your shell or in a local .env file and do not commit it."
        )
    return value


def create_nvidia_embeddings() -> NVIDIAEmbeddings:
    """Create NVIDIA embeddings with explicit API key validation."""
    return NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        nvidia_api_key=get_required_env("NVIDIA_API_KEY"),
    )


def create_groq_model() -> ChatGroq:
    """Create a Groq chat model with explicit API key validation."""
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=get_required_env("GROQ_API_KEY"))


def get_loader(url: str, max_depth: int = 3) -> RecursiveUrlLoader:
    """Create a web page loader for grant and funding sources."""
    return RecursiveUrlLoader(url=url, max_depth=max_depth)


def tag_documents(documents: list[Any], category: str, source_url: str) -> list[Any]:
    """Attach source metadata to documents before indexing."""
    for document in documents:
        document.metadata["category"] = category
        document.metadata["source_url"] = source_url
        document.metadata["last_scraped"] = datetime.now(timezone.utc).isoformat()
    return documents


def load_vectorstore(persist_directory: Path = PERSIST_DIRECTORY) -> Chroma:
    """Load an existing persisted Chroma vector store."""
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory=str(persist_directory),
        collection_name="grants_tenders",
        embedding_function=create_nvidia_embeddings(),
    )


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
        embedding=create_nvidia_embeddings(),
        persist_directory=str(persist_directory),
        collection_name="grants_tenders",
    )
    vectorstore.persist()
    return vectorstore


@dataclass
class ClientProfile:
    organization: str = ""
    contact_name: str = ""
    stage: str = "early-stage"
    industry: str = ""
    location: str = "India"
    team_size: str = ""
    funding_need: str = ""
    focus_areas: str = ""
    beneficiaries: str = ""
    impact_goals: str = ""
    experience: str = ""

    def format(self) -> str:
        profile = asdict(self)
        return "\n".join(
            f"{key.replace('_', ' ').title()}: {value or 'Not specified'}"
            for key, value in profile.items()
        )


def match_profile_to_categories(profile: ClientProfile) -> list[str]:
    """Recommend grant categories based on a client profile."""
    categories: list[str] = []
    industry = profile.industry.lower()
    stage = profile.stage.lower()
    focus = profile.focus_areas.lower()
    beneficiaries = profile.beneficiaries.lower()

    if "startup" in stage or "startup" in industry or "incubator" in focus:
        categories.extend(["startup", "startup_india"])
    if "research" in stage or "research" in industry or "innovation" in focus:
        categories.extend(["research", "dst"])
    if "agri" in industry or "farm" in beneficiaries or "rural" in focus:
        categories.extend(["nabard", "icar", "government"])
    if "international" in focus or "world bank" in beneficiaries or "undp" in focus:
        categories.append("international")
    if "education" in focus or "scholarship" in beneficiaries:
        categories.append("scholarship")
    if not categories:
        categories.append("government")
    return list(dict.fromkeys(categories))


def generate_proposal(grant_summary: str, profile: ClientProfile, model: ChatGroq | None = None) -> str:
    """Generate a proposal outline based on grant details and client profile."""
    if model is None:
        model = create_groq_model()
    chain = LLMChain(llm=model, prompt=PROPOSAL_PROMPT)
    result = chain.invoke(
        {
            "grant_details": grant_summary,
            "client_profile": profile.format(),
        },
        return_only_outputs=True,
    )
    return _extract_text_from_chain_output(result)


def build_retriever_tool(name: str, description: str, retriever: Any) -> Tool:
    """Wrap a retriever in a tool that the agent can call."""

    def search(query: str) -> str:
        try:
            documents = retriever.invoke(query)
        except AttributeError:
            if hasattr(retriever, "_get_relevant_documents"):
                documents = retriever._get_relevant_documents(query, run_manager=None)
            else:
                raise AttributeError(
                    "Retriever does not support invoke() or _get_relevant_documents()"
                )
        return "\n\n".join(document.page_content for document in documents)

    return Tool(name=name, description=description, func=search)


def _extract_text_from_chain_output(output: Any) -> str:
    """Extract a single text string from LLMChain outputs."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if "output" in output and isinstance(output["output"], str):
            return output["output"]
        if "text" in output and isinstance(output["text"], str):
            return output["text"]
        text_values = [value for value in output.values() if isinstance(value, str)]
        if len(text_values) == 1:
            return text_values[0]
        raise RuntimeError(
            "Could not extract text from chain output. "
            f"Chain returned keys: {list(output.keys())}"
        )
    raise TypeError(f"Unsupported chain output type: {type(output).__name__}")


def classify_query(query: str, model: ChatGroq | None = None) -> str:
    """Use the classifier prompt to route queries to the right grant category."""
    if model is None:
        model = create_groq_model()
    chain = LLMChain(llm=model, prompt=CLASSIFIER_PROMPT)
    result = chain.invoke({"query": query}, return_only_outputs=True)
    return _extract_text_from_chain_output(result).strip().lower()


def format_chat_history(chat_history: SQLChatMessageHistory) -> str:
    """Render SQLChatMessageHistory as a plain-text conversation string."""
    messages = chat_history.get_messages()
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or getattr(message, "type", None)
        content = getattr(message, "content", "")
        if role:
            role_lower = role.lower()
            if "human" in role_lower or "user" in role_lower:
                lines.append(f"Human: {content}")
            elif "ai" in role_lower or "assistant" in role_lower:
                lines.append(f"AI: {content}")
            else:
                lines.append(f"{role.title()}: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)


def build_classifier_tool(model: ChatGroq | None = None) -> Tool:
    def classify(query: str) -> str:
        return classify_query(query, model=model)

    return Tool(
        name="classify_query",
        description=(
            "Classify the user's query into one grant category: government, international, "
            "startup, research, scholarship, conference, education, mixed, or out_of_scope."
        ),
        func=classify,
    )


def build_tools(vectorstore: Chroma, model: ChatGroq | None = None) -> list[Tool]:
    """Build the set of tools used by GrantBot."""
    tools: list[Tool] = []
    for category, description in CATEGORY_MAP.items():
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"category": category}})
        tools.append(
            build_retriever_tool(
                name=f"search_{category}",
                description=description,
                retriever=retriever,
            )
        )
    tools.append(build_classifier_tool(model=model))
    return tools


def create_agent(vectorstore: Chroma, model: ChatGroq | None = None) -> AgentExecutor:
    """Create the React agent using the prompt defined in prompts.py."""
    if model is None:
        model = create_groq_model()

    tools = build_tools(vectorstore, model=model)
    agent = create_react_agent(llm=model, tools=tools, prompt=REACT_AGENT_PROMPT)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def load_chat_history(session_id: str = "default_session") -> SQLChatMessageHistory:
    """Load or create a local chat history store."""
    return SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_DB)


def main() -> None:
    """Demonstrate how to construct the GrantBot agent."""
    print("GrantBot agent module loaded.")
    print("To use the agent, build a Chroma vectorstore from your documents and call create_agent(...).")


if __name__ == "__main__":
    main()
