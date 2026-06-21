# Beyond Attention — Anatomy of a Modern Transformer

Code companion to [Beyond Attention: Anatomy of a Modern Transformer](https://tanulsingh.github.io/blog/transformer-anatomy).

The capstone of the series. Everything outside attention — FFN, normalization, residuals, embeddings — and how to assemble them into a full transformer model.

## What you'll build

- **SwiGLU FFN** — the gated FFN used by Llama, Mistral, Qwen, DeepSeek (and the standard ReLU/GELU FFN it replaces, for comparison)
- **RMSNorm** — the simplified normalization used by every modern LLM (replacing LayerNorm)
- **A complete decoder-only transformer model** — token embedding → N transformer blocks → final RMSNorm → output projection

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/transformer-anatomy) first.
2. **Pre-requisites:** complete `blog/04-attention-mechanisms/` (you'll use `MultiHeadAttention` here) and at least the absolute positional encoding folder.
3. Work through:
   - `01_swiglu_ffn_exercise.ipynb` — the FFN evolution
   - `02_rmsnorm_exercise.ipynb` — RMSNorm vs LayerNorm
   - `03_transformer_model_exercise.ipynb` — the capstone: full model assembly

## Files

| File | Purpose |
|------|---------|
| `01_swiglu_ffn_exercise.ipynb` | Build standard FFN, GELU FFN, SwiGLU FFN |
| `01_swiglu_ffn_solution.ipynb` | Reference implementation |
| `02_rmsnorm_exercise.ipynb` | Build LayerNorm and RMSNorm |
| `02_rmsnorm_solution.ipynb` | Reference implementation |
| `03_transformer_model_exercise.ipynb` | Assemble full decoder-only LLM |
| `03_transformer_model_solution.ipynb` | Reference implementation |
| `tests.py` | Test functions |

## The Modern Recipe (2026)

By the end of this folder, you'll have built a transformer matching the de-facto standard:

```
Decoder-only, causal attention
RMSNorm (pre-norm), final RMSNorm before LM head
SwiGLU FFN (~8/3 × d_model hidden)
No biases anywhere, no dropout
RoPE for positional info (from blog/03)
Untied input/output embeddings (at scale)
```

This is essentially what Llama 3, Mistral, Qwen 3, and DeepSeek-V3 all converge on.
