"""
Generates notebooks/llm_training_pipeline/05_evaluation.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_05_evaluation_notebook.py
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"}
}

cells = []

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())

# ─── INTRO ───────────────────────────────────────────────────────────────────
cells.append(md("""
# LLM Training Pipeline — Part 5: Evaluation

Stage 5 of 6. A measurement-only notebook — no new checkpoints are trained. Loads
`sft_model.pt`, `ppo_model.pt`, and `dpo_model.pt` and compares them via an independent
LLM-as-judge (`Qwen2.5-1.5B-Instruct`, run locally), then loads Part 3's
`ppo_training_log.json` to plot the reward-vs-KL overoptimization curve.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 8) for the full discussion of judge biases.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.
- **Note on the judge model:** `Qwen2.5-1.5B-Instruct` (~3GB) downloads from the
  HuggingFace Hub on first run — this is much larger than anything else in this
  pipeline (the ~14M-parameter pipeline model itself is a few tens of MB). The first
  cell that loads it may take a few minutes; this is expected, not a hang.

**Parts:**
1. LLM-as-Judge Pairwise Comparison
2. SFT vs PPO vs DPO Win-Rates
3. Reward-vs-KL Overoptimization Curve
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os, json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, '../..')
from src.llm_pipeline.model import GPTConfig, GPTModel
from src.llm_pipeline.data import TOPIC_KEYWORDS, format_sft_prompt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

CKPT_DIR = "../../data/checkpoints/llm_training_pipeline"
torch.manual_seed(0)

tokenizer = ByteLevelBPETokenizer(
    f"{CKPT_DIR}/tinystories_bpe-vocab.json",
    f"{CKPT_DIR}/tinystories_bpe-merges.txt",
)

def load_pipeline_model(name):
    ckpt = torch.load(f"{CKPT_DIR}/{name}", weights_only=False)
    m = GPTModel(ckpt['config']).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

sft_model = load_pipeline_model('sft_model.pt')
ppo_model = load_pipeline_model('ppo_model.pt')
dpo_model = load_pipeline_model('dpo_model.pt')
print(f"Loaded sft_model.pt, ppo_model.pt, dpo_model.pt — "
      f"{sum(p.numel() for p in sft_model.parameters()):,} params each")
"""))

# Parts 1-3 are appended here by Tasks 4-6.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/05_evaluation.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
