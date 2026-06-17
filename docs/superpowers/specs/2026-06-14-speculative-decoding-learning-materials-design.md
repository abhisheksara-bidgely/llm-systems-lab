# Speculative Decoding Learning Materials — Design Spec
Date: 2026-06-14

## Goal
Build two learning artifacts for deeply understanding Leviathan et al. 2023 ("Fast Inference from Transformers via Speculative Decoding", arxiv 2211.17192):
1. A standalone HTML reference document (full mathematical treatment)
2. A tutorial Jupyter notebook (theory → questions → code blanks → tests)

The notebook is the primary learning experience. The HTML is a companion reference kept open alongside it.

## Learner Profile
- Comfortable with PyTorch basics
- Understands attention conceptually, has not built transformers from scratch
- Needs autoregressive generation mechanics explained before the algorithm
- GPU available with ≤8GB VRAM
- Wants full mathematical treatment — re-derivable proofs

## Models
- **Draft model**: GPT-2 small (`gpt2`) via HuggingFace
- **Target model**: GPT-2 XL (`gpt2-xl`) via HuggingFace
- Same tokenizer (BPE), compatible architectures, fit in ≤8GB VRAM combined

---

## Artifact 1: HTML Reference Document

**File**: `docs/speculative_decoding_reference.html`
**Format**: Single self-contained HTML file (inline CSS; MathJax CDN for LaTeX rendering — requires internet connection)
**Purpose**: Dense reference covering full paper — every proof, every theorem, every formula. Open alongside the notebook.

### Sections

**1. Why Autoregressive Decoding Is Slow**
- Serial token generation: K tokens = K forward passes
- Memory bandwidth bottleneck (not compute): weights must be loaded from HBM for every token; GPU arithmetic units sit idle
- Implication: additional parallel compute is often available "for free" during inference

**2. Speculative Execution**
- Origin in CPU architecture (branch prediction)
- Generalization to stochastic setting: task might be needed with probability p
- Intuition: use a cheap approximation to guess, verify with the expensive model in parallel

**3. Standardized Sampling**
- All sampling methods (argmax, top-k, nucleus, temperature) reduce to sampling from an adjusted probability distribution
- Proof sketch for argmax case (zero out non-max, normalize)
- Why this matters: the algorithm only needs to reason about distributions p(x) and q(x)

**4. Algorithm 1 — Line-by-Line Walkthrough**
- Annotated pseudocode with prose explanation for each step
- Variable glossary: Mp (target), Mq (draft), γ (lookahead), prefix, qi(x), pi(x)
- The batched forward pass: how γ+1 prefixes become one parallel call to Mp
- Rejection loop: what n is, why it's a `min`, what gets discarded
- The adjusted distribution p'(x) = norm(max(0, p - q)): what it means geometrically
- Bonus token: why we always get at least one new token per step

**5. Speculative Sampling — Full Mathematical Treatment**
- Rejection sampling intuition (accept-reject method)
- Lemma: if x ~ q and we accept with prob min(1, p(x)/q(x)), accepted samples are distributed as p
  - Full proof via normalizing constant argument
- DLK divergence: Definition 3.2, Lemma 3.3, Corollary 3.4
- Theorem 3.5: β = 1 − DLK(p, q) — full proof
- Corollary 3.6: α = E[min(p, q)] — what this means intuitively

**6. Analysis**
- Definition 3.1: acceptance rate β
- E[tokens per iteration]: capped geometric variable derivation, Equation 1
- Theorem 3.8: walltime improvement factor — full proof
- Definition 3.7: cost coefficient c
- Corollary 3.9: when does speculative decoding help? (condition α > c)
- Theorem 3.11: arithmetic operations — full proof
- Section 3.5: optimal γ — derivation, worked numerical example (α=0.75, c=0.03)

**7. Experiments Summary**
- Observed α values per task (Table 3 from paper)
- 2-3x speedup on T5-XXL
- Effect of γ on empirical speedup

### Format Details
- Each section ends with a **"Key facts"** callout box (3-4 bullets)
- Theorems/lemmas in styled boxes with proof toggle (click to expand)
- Inline LaTeX rendered via MathJax CDN
- Color-coded: definitions (blue), theorems (green), proofs (grey), key facts (yellow)

---

## Artifact 2: Tutorial Notebook

**File**: `notebooks/speculative_decoding_tutorial.ipynb`
**Format**: Jupyter notebook
**Structure per part**: theory markdown → written questions → code blanks → automated tests

Tests use `assert` statements — pass or fail, no ambiguity. Later parts reuse earlier implementations, so bugs surface early.

### Part 1 — Setup & Baseline Autoregressive Decoding
**Theory**: How transformer generation works token-by-token. What a forward pass returns. What input_ids and logits are. KV cache mention (not implemented here — just acknowledged).

**Written Q1**: "Why does generating K tokens require exactly K serial forward passes? What property of the architecture causes this? What would need to change for this not to be true?"

**Code blank**: `autoregressive_decode(model, input_ids, max_new_tokens) -> Tensor`
- Must use raw logits, no `.generate()`
- Greedy (argmax) decoding only — deterministic, no sampling
- Scaffold: model loading, tokenizer, output loop structure provided

**Test**: token-for-token match with HuggingFace `.generate(do_sample=False)` on 5 prompts

---

### Part 2 — The Bottleneck & Parallelism
**Theory**: Memory bandwidth vs. compute. Why the GPU is underutilized during autoregressive decoding. Why running γ+1 forward passes simultaneously doesn't cost γ+1x walltime (up to a point).

**Written Q2**: "A target model takes 200ms per forward pass. A draft model takes 4ms. What is the cost coefficient c? What is the maximum theoretical tokens/sec improvement if α=0.8 and γ=4? State your assumptions."

**Benchmark cell**: time `autoregressive_decode` on 3 prompts × 50 tokens. Record tokens/sec as `BASELINE_TOKENS_PER_SEC`.

---

### Part 3 — Speculative Sampling
**Theory**: Rejection sampling from scratch. Intuition (accept cheap sample if target agrees, else resample). Full proof that accepted samples are distributed as p. DLK divergence. Theorem 3.5.

**Written Q3**: "Fill in the missing steps of the proof: we sample x ~ q(x). We accept x with probability min(1, p(x)/q(x)). Show that P(output = x) = p(x) for all x."

**Written Q4**: "What does DLK(p, q) = 0 mean? What does DLK(p, q) = 1 mean? Give an example of distributions p and q for each case."

**Code blank**: `speculative_sample(p_probs, q_probs, draft_token_id) -> (token_id, accepted: bool)`
- p_probs, q_probs are full vocabulary distributions (torch.Tensor, shape [vocab_size])
- draft_token_id is the token Mq sampled
- Returns: accepted token (may differ from draft) and whether draft was accepted

**Test**: Run 50k samples with known p and q (small vocab, e.g. 10 tokens). Assert empirical distribution over outputs has KL divergence < 0.01 from p.

---

### Part 4 — Algorithm 1
**Theory**: Full Algorithm 1 walkthrough. The batched forward pass (all γ+1 prefixes in one call). Rejection loop. Adjusted distribution. Bonus token.

**Written Q5**: "What happens when all γ draft tokens are accepted? Where does the (γ+1)th token come from and why is it needed for the distribution to be correct?"

**Written Q6**: "When token xᵢ is rejected, we discard the forward pass result for xᵢ₊₁. Why? Could we reuse it?"

**Code blank**: `speculative_decode_step(target, draft, input_ids, gamma) -> Tensor`
- Scaffold: draft autoregressive loop provided, batched target forward pass structure provided
- Blank: rejection loop (computing n), adjusted distribution, bonus token sampling

**Test**: Run 2000 steps with fixed seeds. Compare token frequency distribution to pure target-model autoregressive. Chi-squared test p-value > 0.05.

---

### Part 5 — Measuring Acceptance Rate
**Theory**: Definition of β and α. Connection to DLK. Equation 1 (E[tokens per iteration]) — derivation from geometric distribution. What α tells you about model pair quality.

**Written Q7**: "Derive E[tokens per iteration] = (1 - α^(γ+1)) / (1 - α) from first principles. Hint: model the number of accepted draft tokens as a geometric random variable with a cap."

**Written Q8**: "If α = 0.5 and γ = 4, what is E[tokens per iteration]? What is it when α → 1?"

**Code blank**: `measure_alpha(target, draft, prompts, gamma) -> float`
- Run speculative_decode_step on each prompt, record accept/reject per position
- Return mean acceptance rate across all positions and prompts

**Experiment cell**: Plot α per token position for 3 prompt types (factual, creative, code). Observation question: "Where is α highest/lowest and why?"

---

### Part 6 — Full Generation Loop & Benchmark
**Theory**: Outer loop over speculative_decode_step. When to stop. Handling variable-length outputs.

**Code blank**: `speculative_decode(target, draft, input_ids, max_new_tokens, gamma) -> Tensor`
- Calls speculative_decode_step in a loop
- Handles EOS token
- Scaffold: loop structure, EOS check provided; blank: assembling output, stopping condition

**Test**: Same outputs as `autoregressive_decode` on 10 prompts with same seed and greedy decoding (deterministic case, γ=1 forces equivalence).

**Benchmark cell**: tokens/sec vs `BASELINE_TOKENS_PER_SEC` from Part 2. Compute actual speedup factor. Compare to Equation 1 prediction given measured α.

---

### Part 7 — Optimal γ
**Theory**: Theorem 3.8 (walltime improvement as function of α, c, γ). Derivation of optimal γ. Corollary 3.9 (condition for any improvement).

**Written Q9**: "Given α=0.75 and c=0.03, derive the optimal γ analytically using Theorem 3.8. Show your work."

**Written Q10**: "What does Corollary 3.9 say in plain English? Give a practical example of when speculative decoding would NOT help."

**Code blank**: `theoretical_speedup(alpha, c, gamma) -> float`
- Implements Theorem 3.8 formula directly

**Plot cell**: walltime improvement vs γ for α ∈ {0.5, 0.7, 0.8, 0.9}, c=0.03

**Experiment cell**: Sweep γ ∈ {1, 2, 3, 4, 5, 7, 10} on GPT-2 small/XL pair. Plot empirical tokens/sec. Mark empirical optimum. Overlay theoretical curve. Observation question: "Does the empirical optimum match the theoretical prediction? If not, why might they differ?"

---

## Directory Structure

```
SpeculativeDecoding/
  papers/
    speculative_decoding_2211.17192.pdf
    speculative_decoding_2603.03251.pdf
  docs/
    speculative_decoding_reference.html
    superpowers/
      specs/
        2026-06-14-speculative-decoding-learning-materials-design.md
  notebooks/
    speculative_decoding_tutorial.ipynb
```

## Dependencies
```
torch
transformers
jupyter
matplotlib
scipy          # for chi-squared test in Part 4
numpy
```

## Implementation Order
1. HTML reference doc (no code dependencies, can be built from paper alone)
2. Notebook Part 1-2 (setup, baseline — needed to measure BASELINE_TOKENS_PER_SEC)
3. Notebook Part 3 (speculative_sample — tested in isolation, no model needed)
4. Notebook Part 4 (speculative_decode_step — depends on Part 3)
5. Notebook Part 5 (measure_alpha — depends on Part 4)
6. Notebook Part 6 (full loop — depends on Parts 3-5)
7. Notebook Part 7 (analysis — depends on Part 6)
