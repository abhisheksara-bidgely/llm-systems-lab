"""
Generates notebooks/llm_training_pipeline/06_rlvr_grpo.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_06_rlvr_grpo_notebook.py
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
# LLM Training Pipeline — Part 6: RLVR / GRPO

Stage 6 of 6, the final notebook in the series. Trains `sft_model.pt` directly (no reward
model, no PPO/DPO checkpoint needed) against a rule-based verifiable reward — constrained
story-ending generation, where the reward is 1 if the continuation mentions a target word
within a token budget, else 0 — using GRPO's group-relative advantage in place of a learned
value function. Produces `grpo_model.pt`.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 9) for the full derivation.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. Verifiable Reward
2. GRPO Core (group-relative advantage)
3. GRPO Training Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, '../..')
from src.llm_pipeline.model import GPTConfig, GPTModel
from src.llm_pipeline.data import TOPIC_KEYWORDS, load_tinystories
from src.llm_pipeline.rlhf import ppo_clipped_loss

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
"""))

# ─── PART 1: VERIFIABLE REWARD ───────────────────────────────────────────────
cells.append(md("""
---
## Part 1: Verifiable Reward

The task: continue a story prefix such that the continuation mentions a specified target
word *and* stays under a token budget. Both conditions are mechanically checkable from the
generated text alone — no reward model, no classifier, no human judgment anywhere in this
reward function. See `docs/llm_training_pipeline_reference.html#s9` for why this removes
the need for the Section 5 reward-modeling apparatus entirely.
"""))

cells.append(code("""
def verifiable_reward(continuation_text, target_word, token_budget, tokenizer):
    n_tokens = len(tokenizer.encode(continuation_text).ids)
    contains_target = target_word.lower() in continuation_text.lower()
    within_budget = n_tokens <= token_budget
    return 1.0 if (contains_target and within_budget) else 0.0


def sample_grpo_prompt(story_text, tokenizer, prefix_tokens=30):
    \"\"\"Takes the first prefix_tokens tokens of story_text as context, picks a
    target word from TOPIC_KEYWORDS not already present in that prefix, and
    frames the task as an explicit instruction.\"\"\"
    prefix_ids = tokenizer.encode(story_text).ids[:prefix_tokens]
    prefix_text = tokenizer.decode(prefix_ids)
    candidates = [w for w in TOPIC_KEYWORDS if w not in prefix_text.lower()]
    target_word = candidates[torch.randint(0, len(candidates), (1,)).item()]
    prompt = f"{prefix_text}\\n(Continue the story above and be sure to mention the word '{target_word}' before you finish.)\\n"
    return prompt, target_word
"""))

cells.append(code("""
# TEST 1: verifiable_reward on synthetic examples covering all four condition combinations
assert verifiable_reward("The dog ran to the park and had fun.", "dog", token_budget=30, tokenizer=tokenizer) == 1.0
assert verifiable_reward("The cat sat on a mat quietly all day long.", "dog", token_budget=30, tokenizer=tokenizer) == 0.0, \\
    "missing target word must score 0 even under budget"
long_but_mentions = "The dog " + "walked and walked and walked and walked and walked and walked and walked. " * 3
assert verifiable_reward(long_but_mentions, "dog", token_budget=10, tokenizer=tokenizer) == 0.0, \\
    "over-budget completion must score 0 even if it mentions the target word"
assert verifiable_reward("Nothing relevant here at all whatsoever today unfortunately.", "dog", token_budget=5, tokenizer=tokenizer) == 0.0, \\
    "missing target AND over budget must score 0"
print("TEST 1 PASSED — verifiable_reward correct on all four contains/budget combinations")
"""))

cells.append(code("""
print("Loading a fresh TinyStories slice (not used by any earlier stage) for GRPO prompts...")
grpo_story_pool = load_tinystories('train[50000:52000]')
print(f"{len(grpo_story_pool)} stories loaded")

torch.manual_seed(0)
example_story = grpo_story_pool[0]
example_prompt, example_target = sample_grpo_prompt(example_story, tokenizer)
print(f"target word: {example_target!r}")
print("prompt:", example_prompt)
"""))

cells.append(code("""
# TEST 2: sample_grpo_prompt never picks a target word already present in its own prefix
for i in range(20):
    story = grpo_story_pool[i]
    prompt, target = sample_grpo_prompt(story, tokenizer)
    prefix_only = prompt.split('\\n(Continue')[0]
    assert target not in prefix_only.lower(), f"target {target!r} leaked into its own prefix"
print("TEST 2 PASSED — sampled target words never already appear in their own prefix (20 samples checked)")
"""))

cells.append(md("""
### Question 1

`verifiable_reward` is a strict AND of two conditions (contains target word, under token
budget) — a completion that mentions the target word using 31 tokens when the budget is 30
scores exactly the same (0) as a completion that never mentions it at all and rambles for
200 tokens. Is collapsing reward to a single bit like this a limitation for how much signal
GRPO's group-relative advantage can extract from a group of completions? What would change
if the reward were instead a continuous value (e.g. token count relative to budget)?

*Write your answer below:*

"""))

# Parts 2-3 are appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/06_rlvr_grpo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
