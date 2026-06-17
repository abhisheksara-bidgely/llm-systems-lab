"""
Downloads arXiv papers for the RAG pipeline corpus.
Run: python scripts/download_papers.py
"""
import time, requests
from pathlib import Path

PAPERS = {
    "rag_lewis2020.pdf":          "https://arxiv.org/pdf/2005.11401",
    "hyde_gao2022.pdf":           "https://arxiv.org/pdf/2212.10496",
    "raptor_sarthi2024.pdf":      "https://arxiv.org/pdf/2401.18059",
    "self_rag_asai2023.pdf":      "https://arxiv.org/pdf/2310.11511",
    "ragas_es2023.pdf":           "https://arxiv.org/pdf/2309.15217",
    "frames_krishna2024.pdf":     "https://arxiv.org/pdf/2409.12941",
    "hotpotqa_yang2018.pdf":      "https://arxiv.org/pdf/1809.09600",
    "react_yao2022.pdf":          "https://arxiv.org/pdf/2210.03629",
    "instructgpt_ouyang2022.pdf": "https://arxiv.org/pdf/2203.02155",
    "dpo_rafailov2023.pdf":       "https://arxiv.org/pdf/2305.18290",
    "vllm_kwon2023.pdf":          "https://arxiv.org/pdf/2309.06180",
    "bge_m3_chen2024.pdf":        "https://arxiv.org/pdf/2402.03216",
    "llm_judge_zheng2023.pdf":    "https://arxiv.org/pdf/2306.05685",
    "speculative_decoding.pdf":   "https://arxiv.org/pdf/2211.17192",
}

out = Path("data/papers")
out.mkdir(parents=True, exist_ok=True)

for name, url in PAPERS.items():
    dest = out / name
    if dest.exists():
        print(f"  skip {name}")
        continue
    print(f"  downloading {name}...")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        time.sleep(1)  # be polite to arXiv
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nDone. {len(list(out.glob('*.pdf')))} PDFs in data/papers/")
