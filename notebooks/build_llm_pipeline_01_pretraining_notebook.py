"""
Generates notebooks/llm_training_pipeline/01_transformer_and_pretraining.ipynb
from cell definitions.
Run: python3 notebooks/build_llm_pipeline_01_pretraining_notebook.py
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
# LLM Training Pipeline — Part 1: Transformer Architecture & Pretraining

Stage 1 of 6 in `notebooks/llm_training_pipeline/`. Builds a ~14M-parameter
decoder-only transformer from scratch and pretrains it on TinyStories.
Later notebooks (SFT, reward model + PPO, DPO, evaluation, RLVR/GRPO) load
the checkpoint this notebook produces.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab for the full derivations.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself; that is the reflective part
  of this notebook.

**Parts:**
1. BPE Tokenizer
2. Causal Self-Attention, MLP, Transformer Block
3. Full GPT Model
4. Data Loading & Packing
5. Pretraining Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

CKPT_DIR = "../../data/checkpoints/llm_training_pipeline"
os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(0)
"""))

# ─── PART 1: TOKENIZER ───────────────────────────────────────────────────────
cells.append(md("""
---
## Part 1: BPE Tokenizer

Language models operate on integer token ids, not raw text. We train a
byte-level BPE tokenizer (same family as GPT-2's tokenizer) on a sample of
TinyStories, vocab size 8000. Byte-level BPE can represent *any* input
string (it falls back to raw bytes for unseen sequences), so there is no
"unknown token" problem.
"""))

cells.append(code("""
print("Loading TinyStories (train[:50000])...")
ds = load_dataset('roneneldan/TinyStories', split='train[:50000]')
texts = [x['text'] for x in ds]
print(f"{len(texts)} stories loaded")

tok_train_path = f"{CKPT_DIR}/tinystories_tok_train.txt"
with open(tok_train_path, 'w') as f:
    f.write('\\n'.join(texts[:20000]))

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[tok_train_path], vocab_size=8000, min_frequency=2,
    special_tokens=['<|endoftext|>'],
)
EOT_ID = tokenizer.token_to_id('<|endoftext|>')
print(f"Vocab size: {tokenizer.get_vocab_size()}, EOT id: {EOT_ID}")
"""))

cells.append(code("""
# TEST 1: tokenizer roundtrip + vocab size
test_strings = [
    "Once upon a time, there was a little girl named Lily.",
    "The dog ran to the park and played with a ball.",
    "\\"I am happy,\\" said Tom. \\"Let's go home!\\"",
]
for s in test_strings:
    ids = tokenizer.encode(s).ids
    decoded = tokenizer.decode(ids)
    assert decoded.strip() == s.strip(), f"roundtrip mismatch: {s!r} -> {decoded!r}"
    print(f"  OK ({len(ids)} tokens): {s[:40]}...")

assert tokenizer.get_vocab_size() == 8000
print("TEST 1 PASSED — tokenizer roundtrip and vocab size verified")
"""))

cells.append(md("""
### Question 1

**Why does a byte-level BPE tokenizer never need an "unknown token"?** What
would happen with a purely word-level tokenizer (split on whitespace, one id
per unique word) applied to text containing a word it never saw during
tokenizer training?

*Write your answer below (double-click this cell to edit):*

"""))

# Parts 2-5 are appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/01_transformer_and_pretraining.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
