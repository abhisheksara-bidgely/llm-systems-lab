"""
Generates notebooks/llm_training_pipeline/04_dpo.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_04_dpo_notebook.py
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
# LLM Training Pipeline — Part 4: Direct Preference Optimization (DPO)

Stage 4 of 6. Loads `sft_model.pt` and the `preference_pairs.json` dataset Part 3 built
(the same pairs the reward model was trained on) and trains a policy directly against them
with the closed-form DPO loss — no reward model, no rollouts, no value function. Produces
`dpo_model.pt`, then compares SFT vs PPO vs DPO on held-out prompts.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 7) for the full derivation.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. DPO Loss
2. DPO Training Loop
3. SFT vs PPO vs DPO Comparison
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os, json, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, '../..')
from src.llm_pipeline.model import GPTConfig, GPTModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

CKPT_DIR = "../../data/checkpoints/llm_training_pipeline"
torch.manual_seed(0)

tokenizer = ByteLevelBPETokenizer(
    f"{CKPT_DIR}/tinystories_bpe-vocab.json",
    f"{CKPT_DIR}/tinystories_bpe-merges.txt",
)
EOT_ID = tokenizer.token_to_id('<|endoftext|>')

sft_ckpt = torch.load(f"{CKPT_DIR}/sft_model.pt", weights_only=False)
sft_cfg = sft_ckpt['config']
sft_model = GPTModel(sft_cfg).to(device)
sft_model.load_state_dict(sft_ckpt['model_state_dict'])
sft_model.eval()
BLOCK_SIZE = sft_cfg.block_size
print(f"Loaded sft_model.pt — {sum(p.numel() for p in sft_model.parameters()):,} params")

with open(f"{CKPT_DIR}/preference_pairs.json") as f:
    preference_pairs = json.load(f)
print(f"Loaded {len(preference_pairs)} preference pairs from Part 3")
"""))

# Parts 1-3 are appended here by Tasks 4-6.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/04_dpo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
