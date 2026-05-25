# Positional Encoding Part 2 — Relative Methods

Code companion to [Positional Encoding Part 2 — RoPE, ALiBi, and the Quest for Length Generalization](https://tanulsingh.github.io/blog/positional-encoding-relative).

> **Do Part 1 first.** This folder builds on `blog/02-positional-encoding-absolute/`. The rotation property you verified there is the seed of RoPE.

## What you'll build

- ALiBi — Attention with Linear Biases (Press et al., 2022)
- RoPE — Rotary Position Embedding (Su et al., 2021), the method that won
- A length-generalization comparison across four methods (sinusoidal, learned, ALiBi, RoPE)

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/positional-encoding-relative) first.
2. Work through each exercise in order:
   - `02_alibi_exercise.ipynb` — fixed linear penalty on attention scores
   - `03_rope_exercise.ipynb` — rotation applied to Q and K (the big one)
3. For each exercise:
   - Work through the TODOs.
   - Compare with the `*_solution.ipynb`.
4. Run `comparison.ipynb` for the length-generalization showpiece.

## Files

| File | Purpose |
|------|---------|
| `02_alibi_exercise.ipynb` | Build ALiBi with geometric head slopes |
| `02_alibi_solution.ipynb` | Reference implementation |
| `03_rope_exercise.ipynb` | Build RoPE — the most important method |
| `03_rope_solution.ipynb` | Reference implementation |
| `comparison.ipynb` | Length-generalization comparison across methods |

## A note on the comparison

The `comparison.ipynb` swaps positional encoding methods inside a tiny transformer (see `common/mini_transformer.py`) and trains each on a short-context task, then evaluates on a longer context. This is the experimental basis for "RoPE generalizes; absolute methods don't" claims in the blog.
