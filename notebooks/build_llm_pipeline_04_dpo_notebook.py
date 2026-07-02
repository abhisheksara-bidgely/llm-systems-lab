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

# ─── PART 1: DPO LOSS ────────────────────────────────────────────────────────
cells.append(md("""
---
## Part 1: DPO Loss

`tokenize_prompt_response` generalizes Part 2's `tokenize_sft_example` to arbitrary prompt
and response strings (not just the SFT topic template) — the mask boundary rule is
identical: a target token is masked (`-100`) iff it falls inside the prompt or padding
region. `sequence_logprob` sums the log-probability of the response tokens only, giving
`log pi(y|x)` for a whole completion. `dpo_loss` implements the closed-form loss from
`docs/llm_training_pipeline_reference.html#s7` directly.
"""))

cells.append(code("""
def tokenize_prompt_response(prompt, response, tokenizer, eot_id, block_size):
    prompt_ids = tokenizer.encode(prompt).ids
    completion_ids = tokenizer.encode(response).ids + [eot_id]
    full_ids = (prompt_ids + completion_ids)[: block_size + 1]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_real = len(full_ids)

    pad_len = (block_size + 1) - n_real
    full_ids = full_ids + [eot_id] * pad_len

    input_ids = full_ids[:-1]
    targets_raw = full_ids[1:]

    labels = []
    for i in range(block_size):
        target_pos = i + 1
        if target_pos < n_prompt or target_pos >= n_real:
            labels.append(-100)
        else:
            labels.append(targets_raw[i])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def sequence_logprob(model, input_ids, labels):
    \"\"\"Returns (B,): sum of log pi(token) over only the non-masked (response)
    positions in each sequence — log pi(y|x) for the whole completion.\"\"\"
    logits, _ = model(input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
"""))

cells.append(code("""
# TEST 1: DPO loss against a hand-computed toy example, plus a monotonicity sanity check
policy_chosen_lp = torch.tensor([-2.0])
policy_rejected_lp = torch.tensor([-3.0])
ref_chosen_lp = torch.tensor([-2.5])
ref_rejected_lp = torch.tensor([-2.5])
beta = 0.5

loss = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta)
# by hand: pi_logratios = -2.0 - (-3.0) = 1.0; ref_logratios = -2.5 - (-2.5) = 0.0
# logits = 0.5 * (1.0 - 0.0) = 0.5; loss = -log(sigmoid(0.5)), computed independently below
expected = -math.log(1.0 / (1.0 + math.exp(-0.5)))
assert abs(loss.item() - expected) < 1e-5, f"{loss.item()} != {expected}"
print(f"TEST 1a PASSED — DPO loss matches hand-computed value ({loss.item():.4f})")

# Monotonicity: loss should be lower when the policy prefers chosen over rejected
# *more strongly relative to the reference* than in a case where it prefers the opposite.
loss_good = dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
loss_bad = dpo_loss(torch.tensor([-3.0]), torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
assert loss_good.item() < loss_bad.item(), "DPO loss should be lower when the policy prefers chosen over rejected relative to the reference"
print(f"TEST 1b PASSED — loss_good ({loss_good.item():.4f}) < loss_bad ({loss_bad.item():.4f})")
"""))

cells.append(code("""
# TEST 2: tokenize_prompt_response mask boundary (same rule as SFT's tokenize_sft_example)
prompt, response = "Write a short story about dog:\\n", "A dog ran fast."
prompt_len = len(tokenizer.encode(prompt).ids)
input_ids, labels = tokenize_prompt_response(prompt, response, tokenizer, EOT_ID, BLOCK_SIZE)
assert input_ids.shape == (BLOCK_SIZE,) and labels.shape == (BLOCK_SIZE,)
assert torch.all(labels[: prompt_len - 1] == -100), "prompt-region targets not fully masked"
assert labels[prompt_len - 1].item() != -100, "first response token incorrectly masked"
print(f"TEST 2 PASSED — mask boundary correct (prompt_len={prompt_len})")
"""))

cells.append(md("""
### Question 1

`sequence_logprob` sums log-probabilities over the response tokens rather than averaging
them. Suppose `y_w` (chosen) is a much longer response than `y_l` (rejected) for the same
prompt. Could summing (rather than averaging) systematically bias which response the DPO
loss favors, independent of which one is actually better? What would change if
`sequence_logprob` divided by the number of response tokens instead?

*Write your answer below:*

"""))

# Parts 2-3 are appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/04_dpo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
