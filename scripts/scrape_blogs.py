"""
Scrapes engineering blog posts for the RAG pipeline corpus.
Run: python scripts/scrape_blogs.py
Saves cleaned text to data/blogs/<slug>.txt with metadata header.
"""
import re, time, requests
from pathlib import Path
from bs4 import BeautifulSoup

BLOGS = [
    {"url": "https://www.anthropic.com/news/contextual-retrieval",    "company": "Anthropic",   "topic": "RAG"},
    {"url": "https://huggingface.co/blog/evaluation-structured-outputs", "company": "HuggingFace", "topic": "eval"},
    {"url": "https://huggingface.co/blog/dpo-trl",                    "company": "HuggingFace", "topic": "fine-tuning"},
    {"url": "https://cohere.com/blog/rerank",                         "company": "Cohere",      "topic": "RAG"},
    {"url": "https://blog.vllm.ai/2023/06/20/vllm.html",             "company": "vLLM",        "topic": "inference"},
]

out = Path("data/blogs")
out.mkdir(parents=True, exist_ok=True)


def slug(url: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', url.lower().split("//")[-1])[:60]


for b in BLOGS:
    dest = out / f"{slug(b['url'])}.txt"
    if dest.exists():
        print(f"  skip {dest.name}")
        continue
    print(f"  scraping {b['url']}...")
    try:
        r = requests.get(b["url"], timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        header = f"# SOURCE: {b['url']}\n# COMPANY: {b['company']}\n# TOPIC: {b['topic']}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        time.sleep(2)
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nDone. {len(list(out.glob('*.txt')))} blogs in data/blogs/")
