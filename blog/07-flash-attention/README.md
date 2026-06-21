# Attention Part 4 — Flash Attention

Code companion to [Attention Part 4 — Flash Attention: Making GPUs Actually Work](https://tanulsingh.github.io/blog/flash-attention).

> **Important caveat.** Flash Attention's actual speedup comes from a hand-tuned CUDA kernel that exploits SRAM/HBM mechanics. You can't reproduce the wall-clock speedup in pure Python. **What you can do here:** implement the *algorithm* — online softmax + tiled attention — and verify that the output is bit-equivalent to standard attention. That's the pedagogically interesting part.

## What you'll build

- **Online softmax** — the running-max + correction trick that lets you compute softmax block-by-block instead of materializing the whole row
- **Tiled attention** — combine online softmax with block-wise QK^T and PV matmuls; the output matches standard attention exactly

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/flash-attention) first.
2. **Pre-requisite:** complete `blog/04-attention-mechanisms/` (you'll compare against your standard attention).
3. Work through:
   - `01_online_softmax_exercise.ipynb` — the core trick
   - `02_tiled_attention_exercise.ipynb` — assemble Flash Attention's algorithm

## Files

| File | Purpose |
|------|---------|
| `01_online_softmax_exercise.ipynb` | Build online softmax (running max + correction) |
| `01_online_softmax_solution.ipynb` | Reference implementation |
| `02_tiled_attention_exercise.ipynb` | Tiled attention using online softmax |
| `02_tiled_attention_solution.ipynb` | Reference implementation |
| `tests.py` | Test functions |

## What to take away

- **Online softmax** is the algorithmic enabler — without it, you can't tile.
- **Kernel fusion** is the actual contribution — keeping the N×N matrix in SRAM throughout.
- **Recomputation** is free when you're memory-bound — that's why the backward pass recomputes P instead of storing it.
- In practice: use `torch.nn.functional.scaled_dot_product_attention` or `flash-attn` — never roll your own CUDA.
