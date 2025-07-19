import requests
from bs4 import BeautifulSoup
import csv
import json

COLLECTION_URL = "https://huggingface.co/collections/The-Great-Genius/skynet-66366061cc7af105efb7e0ca"

def fetch_collection(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def parse_papers(html):
    soup = BeautifulSoup(html, "html.parser")
    # Entries are in <a> and <p> tags; we'll locate the container listing
    items = soup.select("div.sc-bdVaJa.kKHybU .paper-row, div.paper-row") or soup.select("p")
    papers = []
    for p in items:
        text = p.get_text(separator=" ").strip()
        # Example: "FLAME: Factuality-Aware Alignment for Large Language Models Paper • 2405.01525 • Published May 2, 2024 • 29"
        parts = [part.strip() for part in text.split("•")]
        if len(parts) < 3:
            continue
        title = parts[0].replace("Paper", "").strip()
        arxiv_id = parts[1]
        pub_date = parts[2].replace("Published", "").strip()
        papers.append({
            "title": title,
            "arxiv_id": arxiv_id,
            "publication_date": pub_date
        })
    return papers

def save_json(papers, path="skynet_papers.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
    print(f"[+] Saved JSON to {path}")

def save_csv(papers, path="skynet_papers.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "arxiv_id", "publication_date"])
        writer.writeheader()
        writer.writerows(papers)
    print(f"[+] Saved CSV to {path}")

def main():
    html = fetch_collection(COLLECTION_URL)
    papers = parse_papers(html)
    print(f"[+] Found {len(papers)} papers")
    save_json(papers)
    save_csv(papers)

if __name__ == "__main__":
    main()
