"""Ingest grant web pages into the Chroma vector store."""
import argparse
from pathlib import Path

from Grant_agent import build_vectorstore, get_loader, tag_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest grant sources into Chroma.")
    parser.add_argument("--url", required=True, help="Source URL to scrape")
    parser.add_argument("--category", required=True, help="Grant category to tag documents")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")
    parser.add_argument("--persist-dir", default="grant_db", help="Chroma persistence directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    persist_dir = Path(args.persist_dir)
    loader = get_loader(args.url, max_depth=args.max_depth)
    print(f"Loading documents from {args.url} (depth={args.max_depth})...")
    docs = loader.load()
    docs = tag_documents(docs, category=args.category, source_url=args.url)
    print(f"Indexing {len(docs)} documents into {persist_dir}...")
    vectorstore = build_vectorstore(docs, persist_directory=persist_dir)
    vectorstore.persist()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
