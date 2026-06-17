"""Scheduler to refresh ingested sources periodically."""
import schedule
import time
from Grant_agent import get_loader, tag_documents, build_vectorstore

SCHEDULED_FEEDS = [
    {
        "url": "https://www.myscheme.gov.in/",
        "category": "government",
        "max_depth": 3,
    },
    {
        "url": "https://www.worldbank.org/",
        "category": "international",
        "max_depth": 2,
    },
    {
        "url": "https://startup.google.com/",
        "category": "startup",
        "max_depth": 2,
    },
]


def refresh_sources() -> None:
    for feed in SCHEDULED_FEEDS:
        loader = get_loader(feed["url"], max_depth=feed["max_depth"])
        docs = loader.load()
        docs = tag_documents(docs, category=feed["category"], source_url=feed["url"])
        vectorstore = build_vectorstore(docs)
        vectorstore.persist()
        print(f"Refreshed {feed['category']} data from {feed['url']}")


def main() -> None:
    schedule.every().day.at("03:00").do(refresh_sources)
    print("Scheduler started. Refreshing sources daily at 03:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
