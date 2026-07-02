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

# ─── PART 1: INSTRUCTION DATASET ─────────────────────────────────────────────
cells.append(md("""
---
## Part 1: Instruction Dataset Construction

TinyStories has no built-in `(prompt, response)` structure. We synthesize one:
for each story, we heuristically extract a "topic" (a keyword the story is
actually about, from a fixed vocabulary of ~40 common TinyStories nouns), then
frame the story as the answer to `"Write a short story about {topic}:\\n"`.
Stories where no keyword matches are dropped.
"""))

cells.append(code("""
from datasets import load_dataset

TOPIC_KEYWORDS = [
    "dog", "cat", "girl", "boy", "forest", "ball", "tree", "bird", "star",
    "friend", "monster", "princess", "dragon", "robot", "garden", "park",
    "school", "castle", "rabbit", "mouse", "flower", "boat", "river",
    "mountain", "farm", "toy", "bear", "fish", "sun", "moon", "rain",
    "snow", "house", "family", "birthday", "picnic", "adventure", "magic",
    "kite", "puppy",
]

def extract_topic(story):
    lower = story.lower()
    for kw in TOPIC_KEYWORDS:
        if kw in lower:
            return kw
    return None

def format_sft_prompt(topic):
    return f"Write a short story about {topic}:\\n"

print("Loading TinyStories (train[:50000]) for SFT pair construction...")
ds = load_dataset('roneneldan/TinyStories', split='train[:50000]')
sft_pairs = []
for x in ds:
    topic = extract_topic(x['text'])
    if topic is not None:
        sft_pairs.append((topic, x['text']))
print(f"{len(sft_pairs)} / {len(ds)} stories matched a topic keyword")
"""))

cells.append(code("""
# TEST 1: extraction determinism + prompt formatting + coverage sanity
assert extract_topic("A brave dog ran through the yard.") == "dog"
assert extract_topic("The weather was strange that day with no mentioned nouns.") is None
assert format_sft_prompt("dragon") == "Write a short story about dragon:\\n"
assert len(sft_pairs) > 10000, f"expected >10000 matched pairs, got {len(sft_pairs)}"
for topic, story in sft_pairs[:3]:
    assert topic in story.lower()
print(f"TEST 1 PASSED — {len(sft_pairs)} pairs, extraction verified on samples")
"""))

cells.append(md("""
### Question 1

`extract_topic` returns the **first** matching keyword found by scanning
`TOPIC_KEYWORDS` in a fixed order, even if a story matches several keywords
(e.g. a story about both a "dog" and a "park"). Is this a problem for
training a model to follow the `"Write a short story about {topic}:\\n"`
instruction? What would you check to find out?

*Write your answer below:*

"""))

# ─── PART 2: PROMPT-LOSS-MASKING ─────────────────────────────────────────────
cells.append(md("""
---
## Part 2: Prompt-Loss-Masking

`GPTModel.forward` already treats label value `-100` as "ignore this position"
(PyTorch's `F.cross_entropy` default `ignore_index`) — no model code changes
are needed. We only need a tokenizer function that builds the
`(input_ids, labels)` pair with prompt-token predictions masked out. This
follows the same shift-by-one convention as pretraining's `pack_into_blocks`:
`input_ids[i]` predicts `labels[i]`, where `labels[i]` is the *next* token
after `input_ids[i]` — so masking is applied to which *target* falls inside
the prompt, not which *input* position does.
See `docs/llm_training_pipeline_reference.html#s4` for why we mask at all.
"""))

cells.append(code("""
def tokenize_sft_example(topic, story, tokenizer, eot_id, block_size):
    prompt_ids = tokenizer.encode(format_sft_prompt(topic)).ids
    completion_ids = tokenizer.encode(story).ids + [eot_id]
    full_ids = (prompt_ids + completion_ids)[: block_size + 1]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_real = len(full_ids)

    pad_len = (block_size + 1) - n_real
    full_ids = full_ids + [eot_id] * pad_len

    input_ids = full_ids[:-1]           # length block_size
    targets_raw = full_ids[1:]          # length block_size, shifted by one

    labels = []
    for i in range(block_size):
        target_pos = i + 1              # position in the original (unshifted) sequence
        if target_pos < n_prompt or target_pos >= n_real:
            labels.append(-100)         # target is a prompt token, or padding
        else:
            labels.append(targets_raw[i])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

BLOCK_SIZE = 256
"""))

cells.append(code("""
# TEST 2: known mask boundary + masked-loss != full-loss
topic, story = "dog", "A dog ran fast."
prompt_len = len(tokenizer.encode(format_sft_prompt(topic)).ids)
input_ids, labels = tokenize_sft_example(topic, story, tokenizer, EOT_ID, BLOCK_SIZE)

assert input_ids.shape == (BLOCK_SIZE,)
assert labels.shape == (BLOCK_SIZE,)
# every target position before (prompt_len - 1) must be masked (predicting a prompt token)
assert torch.all(labels[: prompt_len - 1] == -100), "prompt-region targets not fully masked"
# the position right after the prompt should predict the first response token (not masked)
assert labels[prompt_len - 1].item() != -100, "first response token incorrectly masked"
print(f"TEST 2a PASSED — mask boundary correct (prompt_len={prompt_len})")

masked_logits, masked_loss = base_model(input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device))
unmasked_targets = torch.tensor(
    (tokenizer.encode(format_sft_prompt(topic)).ids + tokenizer.encode(story).ids + [EOT_ID])[1: BLOCK_SIZE + 1]
    + [EOT_ID] * max(0, BLOCK_SIZE - (len(tokenizer.encode(format_sft_prompt(topic)).ids + tokenizer.encode(story).ids + [EOT_ID]) - 1)),
    dtype=torch.long,
)[:BLOCK_SIZE].unsqueeze(0).to(device)
_, full_loss = base_model(input_ids.unsqueeze(0).to(device), unmasked_targets)

print(f"masked_loss={masked_loss.item():.4f}, full_loss(unmasked prompt+response)={full_loss.item():.4f}")
assert abs(masked_loss.item() - full_loss.item()) > 1e-4, "masked and unmasked loss should differ"
print("TEST 2b PASSED — masked loss differs from full (unmasked) loss")
"""))

cells.append(md("""
### Question 2

The masking rule above checks `target_pos < n_prompt`, i.e. it masks based on
where the **predicted** token falls, not where the **input** token falls.
Concretely: the input token at the last prompt position (`input_ids[n_prompt - 1]`,
the `\\n` after the prompt) is *not* masked as an input — it's fed into the
model normally. Why does that make sense, given the model needs to see the
whole prompt to generate the response?

*Write your answer below:*

"""))

# Part 3 is appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/02_sft.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
