"""
Generates notebooks/llm_training_pipeline/02_sft.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_02_sft_notebook.py
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
# LLM Training Pipeline — Part 2: Supervised Fine-Tuning (SFT)

Stage 2 of 6. Loads `base_model.pt` from notebook 1 and fine-tunes it on an
instruction-formatted slice of TinyStories, using prompt-loss-masking so only
the response tokens contribute gradient. Produces `sft_model.pt`, the
checkpoint every later stage (reward model + PPO, DPO, evaluation,
RLVR/GRPO) fine-tunes or references as the frozen "reference" policy.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 4) for the full derivations.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. Instruction Dataset Construction
2. Prompt-Loss-Masking
3. SFT Training Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os
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

ckpt = torch.load(f"{CKPT_DIR}/base_model.pt", weights_only=False)
base_cfg = ckpt['config']
base_model = GPTModel(base_cfg).to(device)
base_model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded base_model.pt — {sum(p.numel() for p in base_model.parameters()):,} params")
"""))

# Parts 1-3 are appended here by Tasks 4-6.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/02_sft.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
