# Attention Part 3 — The Sparsity Lineage

Code companion to [Attention Part 3 — The Sparsity Lineage: Making Attention Sub-Quadratic](https://tanulsingh.github.io/blog/attention-sparsity).

## What you'll build

- **Sliding window attention** (Mistral-style) — each token attends to a fixed-size local window
- **Global + Local hybrid** (Gemma 3 style) — alternating local/global layers, where most layers use sliding window and a few use full attention
- A **receptive field demo** — visualize how information from a distant token reaches the current token through stacked sliding-window layers

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/attention-sparsity) first.
2. **Pre-requisite:** complete `blog/04-attention-mechanisms/` (you'll extend the masking and attention from there).
3. Work through:
   - `01_sliding_window_exercise.ipynb` — bounded local attention
   - `02_hybrid_global_local_exercise.ipynb` — alternating layer pattern
4. Run `receptive_field_demo.ipynb` to see how information propagates.

## Files

| File | Purpose |
|------|---------|
| `01_sliding_window_exercise.ipynb` | Build sliding window attention |
| `01_sliding_window_solution.ipynb` | Reference implementation |
| `02_hybrid_global_local_exercise.ipynb` | Build hybrid (some layers local, some global) |
| `02_hybrid_global_local_solution.ipynb` | Reference implementation |
| `receptive_field_demo.ipynb` | Visualize multi-layer information flow |
| `tests.py` | Test functions |

## Note on Sparse Transformer and NSA

The blog post discusses Sparse Transformer's strided/fixed patterns (historical, not used in production) and Native Sparse Attention (covered in a dedicated paper post). We focus on the techniques actually used in production LLMs today: sliding window and global/local hybrids.
