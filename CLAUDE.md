# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What we are building

A from-scratch reimplementation of **LUQ (Long-text Uncertainty Quantification)** that runs locally on Apple Silicon (no CUDA/vLLM), validated on an open dataset and wrapped as an MCP tool.

Core idea: split a long-form answer into sentences, run an NLI judge between each sentence and each sibling sample, softmax over {entailment, contradiction} only, average → a scalar uncertainty. Then correlate that uncertainty against human factuality labels.

## Key locations

- [references/execution_steps.md](references/execution_steps.md) — the staged build plan (Stage 0 setup → Stage 6 report). **Start here.**
- [references/LUQ_research_plan.md](references/LUQ_research_plan.md) — full feasibility assessment: dataset choice, model choice, throughput, MCP design, risks, expected results. The source of truth for decisions.
- [LUQ-main/luq_vllm.py](LUQ-main/luq_vllm.py) — original LUQ reference (CUDA/vLLM, **not runnable here**). Reuse the *algorithm* only, not the runtime.
- [LUQ-main/README.md](LUQ-main/README.md) — upstream readme.
- `src/` — where all new code goes (currently empty).

## Fixed decisions (from the research plan)

- **Dataset**: `potsawee/wiki_bio_gpt3_hallucination` (238 rows, pre-sampled 20 siblings + human sentence-level labels, $0 API spend).
- **NLI judge**: `cross-encoder/nli-deberta-v3-base` on CPU first; fall back to MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`. FP32 only (no fp16/bf16 — DeBERTa-v3 is unstable in low precision).
- **Entailment direction**: sibling sample = premise, main-answer sentence = hypothesis. Reversing kills the signal.
- **MCP**: FastMCP from `mcp[cli]` (pin `mcp>=1.2`), primary tool `luq_score(passage, samples) -> {uncertainty, confidence}`.
- **Checkpoint**: after Stage 4 (clean negative correlation, Pearson ≈ −0.4 or stronger). Everything after is packaging.

## Setup (not yet run)

```bash
uv venv && uv add "mcp[cli]" transformers torch sentence-transformers datasets scipy spacy
python -m spacy download en_core_web_sm
```

## Engineering preferences

DRY is important — flag repetition aggressively. Code modularity is crucial.

"Engineered enough" — not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity).

Err on the side of handling more edge cases, not fewer. Thoughtfulness > speed. We want to solve issues robustly.

Bias toward explicit over clever. If anything can be handled in a simpler way, point this out clearly and ask for input.

When implementing methods, we want types and docstrings with Args and Returns where possible and appropriate.

Prioritize simple over complex wherever possible, only iterating towards more complex if proven to be necessary.

Use uv for dependency management.