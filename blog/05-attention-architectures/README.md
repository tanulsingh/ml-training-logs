# Attention Part 2 — The Sharing Lineage (MQA, GQA, MLA)

Code companion to [Attention Part 2 — The Sharing Lineage: From Multi-Query to Multi-Latent](https://tanulsingh.github.io/blog/attention-architectures).

## What you'll build

- **Multi-Query Attention (MQA)** — 1 K and 1 V head shared across all Q heads (Shazeer, 2019)
- **Grouped-Query Attention (GQA)** — groups of Q heads share K, V (Ainslie et al., 2023)
- **Multi-Latent Attention (MLA)** — low-rank latent compression of K, V (DeepSeek-V2, 2024)
- A KV-cache comparison showing memory savings across MHA / MQA / GQA / MLA

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/attention-architectures) first.
2. **Pre-requisite:** complete `blog/04-attention-mechanisms/` first — these exercises extend `MultiHeadAttention`.
3. Work through each exercise:
   - `01_mqa_exercise.ipynb` — strip down to 1 KV head
   - `02_gqa_exercise.ipynb` — generalize to G groups
   - `03_mla_exercise.ipynb` — latent compression (the hardest, optional but most impactful)
4. Run `kv_cache_comparison.ipynb` to see the memory savings plotted.

## Files

| File | Purpose |
|------|---------|
| `01_mqa_exercise.ipynb` | Build MQA from MHA |
| `01_mqa_solution.ipynb` | Reference implementation |
| `02_gqa_exercise.ipynb` | Build GQA (groups of Q heads share KV) |
| `02_gqa_solution.ipynb` | Reference implementation |
| `03_mla_exercise.ipynb` | Build MLA latent compression |
| `03_mla_solution.ipynb` | Reference implementation |
| `kv_cache_comparison.ipynb` | KV cache size: MHA vs MQA vs GQA vs MLA |
| `tests.py` | Test functions |

## What models use these

- **MQA**: PaLM, Falcon, StarCoder
- **GQA**: Llama 2/3/4, Mistral, Gemma 1-3, Qwen 2.5/3 — the de-facto standard
- **MLA**: DeepSeek-V2/V3/R1, GLM-5, Mistral Large 3 — the next frontier
