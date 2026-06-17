"""
run_grant_agent.py — Example runner for the GrantBot agent.

This file demonstrates how to load a persisted Chroma vector store, create the
GrantBot agent from Grant_agent.py, and run a simple user query.
"""

from pathlib import Path

from Grant_agent import create_agent, create_nvidia_embeddings, format_chat_history, load_chat_history


PERSIST_DIRECTORY = Path("grant_db")


def load_vectorstore(persist_directory: Path):
    """Load an existing Chroma vector store from disk."""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as exc:
        raise ImportError(
            "langchain_community is required to load the persisted vector store. "
            "Install it before running this script."
        ) from exc

    return Chroma(
        persist_directory=str(persist_directory),
        collection_name="grants_tenders",
        embedding_function=create_nvidia_embeddings(),
    )


def main() -> None:
    """Run a sample agent query against GrantBot."""

    if not PERSIST_DIRECTORY.exists():
        print(
            "Persisted vector store not found.\n"
            "Build the vector store first using Grant_agent.build_vectorstore(...)."
        )
        return

    vectorstore = load_vectorstore(PERSIST_DIRECTORY)
    agent = create_agent(vectorstore)
    chat_history = load_chat_history(session_id="default_session")
    formatted_history = format_chat_history(chat_history)

    user_query = (
        "Find grants or funding opportunities for early-stage Indian startups "
        "working on clean energy or sustainability."
    )

    print("=== GrantBot query ===")
    print(user_query)
    print("\n=== Agent response ===")

    try:
        response = agent.invoke({"input": user_query, "chat_history": formatted_history})
    except Exception as exc:
        print("Agent execution failed:", exc)
        return

    print(response["output"])

    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(response["output"])
    session_id = "default_session"

    print("\n=== Session saved to chat history ===")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()
