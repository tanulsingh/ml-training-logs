# Attention Part 1 — The Mechanism

Code companion to [Attention Part 1 — The Mechanism That Changed Everything](https://tanulsingh.github.io/blog/attention-mechanisms).

## What you'll build

- Scaled dot-product attention from scratch (Q, K, V projections, dot product, scaling, softmax, weighted sum)
- Causal masking for autoregressive (decoder-only) attention
- Multi-head attention via the reshape-then-split pattern (the GPU-efficient version)
- A demo of the quadratic cost wall — visualizing why attention scales poorly to long sequences

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/attention-mechanisms) first.
2. Work through each exercise in order:
   - `01_scaled_dot_product_attention_exercise.ipynb` — bidirectional self-attention
   - `02_causal_mask_exercise.ipynb` — masked / causal self-attention
   - `03_multi_head_attention_exercise.ipynb` — multi-head with reshape
3. Run `quadratic_cost_demo.ipynb` to see the O(n²) wall in plot form.

## Files

| File | Purpose |
|------|---------|
| `01_scaled_dot_product_attention_exercise.ipynb` | Build single-head self-attention |
| `01_scaled_dot_product_attention_solution.ipynb` | Reference implementation |
| `02_causal_mask_exercise.ipynb` | Add causal masking |
| `02_causal_mask_solution.ipynb` | Reference implementation |
| `03_multi_head_attention_exercise.ipynb` | Multi-head with reshape |
| `03_multi_head_attention_solution.ipynb` | Reference implementation |
| `quadratic_cost_demo.ipynb` | Visualize O(n²) cost growth |
| `tests.py` | Test functions imported by exercise notebooks |

## Connection to other parts

- The `MultiHeadAttention` you build here is the foundation for Parts 5–7.
- The quadratic cost demo motivates Parts 5 (MQA/GQA/MLA) and 7 (Flash Attention).
- Causal attention is what every modern decoder-only LLM uses.
