# Positional Encoding Part 1 — Absolute Methods

Code companion to [Positional Encoding Part 1 — Why Transformers Need to Know Where Words Are](https://tanulsingh.github.io/blog/positional-encoding-absolute).

## What you'll build

- A demo of why self-attention is permutation invariant (motivating the need for PE)
- Sinusoidal positional encoding (Vaswani et al., 2017) — from scratch
- Learned positional embeddings (BERT/GPT-2 style) — from scratch
- A side-by-side comparison

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/positional-encoding-absolute) first.
2. Run `00_permutation_invariance_demo.ipynb` — feel the problem.
3. Open `01_sinusoidal_exercise.ipynb` and work through the TODOs.
   - Verify your work with the test cell at the end.
   - Compare against `01_sinusoidal_solution.ipynb`.
4. Repeat with `02_learned_exercise.ipynb`.
5. Open `comparison.ipynb` to see them side by side.

## Files

| File | Purpose |
|------|---------|
| `00_permutation_invariance_demo.ipynb` | Demo: self-attention is order-blind without PE |
| `01_sinusoidal_exercise.ipynb` | Build sinusoidal PE — your turn |
| `01_sinusoidal_solution.ipynb` | Reference implementation |
| `02_learned_exercise.ipynb` | Build learned positional embeddings |
| `02_learned_solution.ipynb` | Reference implementation |
| `comparison.ipynb` | Visualize and compare both methods |
| `tests.py` | Test functions imported by exercise notebooks |
