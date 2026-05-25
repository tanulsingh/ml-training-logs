# Tokenization — From Scratch and With Libraries

Code companion to [Tokenization — The First Gradient Descent Step](https://tanulsingh.github.io/blog/tokenization).

## What you'll build

- **Byte Pair Encoding (BPE)** from scratch — the algorithm behind GPT-2/3/4, Llama, and most modern LLMs (Sennrich et al., 2016)
- **Byte-level BPE** — the multilingual extension that ships in cl100k (GPT-4) and tiktoken
- **WordPiece** from scratch — the PMI-based variant powering BERT (Schuster & Nakajima, 2012)
- A demo of using **tiktoken** (GPT-4's tokenizer)

## How to use this folder

1. Read the [blog post](https://tanulsingh.github.io/blog/tokenization) first.
2. Build the from-scratch tokenizers in order — each builds on the previous:
   - `01_bpe_exercise.ipynb` — character-level BPE
   - `02_byte_level_bpe_exercise.ipynb` — extend to UTF-8 bytes for multilingual support
   - `03_wordpiece_exercise.ipynb` — modify the merge criterion to PMI
3. Try the production library:
   - `04_using_tiktoken_demo.ipynb` — what GPT-4 uses

## Files

| File | Purpose |
|------|---------|
| `01_bpe_exercise.ipynb` | Build BPE from scratch (character-level) |
| `01_bpe_solution.ipynb` | Reference implementation |
| `02_byte_level_bpe_exercise.ipynb` | Extend BPE to operate on UTF-8 bytes |
| `02_byte_level_bpe_solution.ipynb` | Reference implementation |
| `03_wordpiece_exercise.ipynb` | Build WordPiece (BPE with PMI scoring) |
| `03_wordpiece_solution.ipynb` | Reference implementation |
| `04_using_tiktoken_demo.ipynb` | Library demo: GPT-4's tokenizer |
| `tests.py` | Test functions imported by exercise notebooks |

## A note on Unigram and SentencePiece

The blog post covers Unigram and SentencePiece, but we don't implement either from scratch here — the EM algorithm and Viterbi segmentation involve enough complexity that the educational ROI is low. If you need a production tokenizer trained on your own data, reach for the [SentencePiece library](https://github.com/google/sentencepiece) directly.

## Why this matters

Tokenization is the **first** thing any LLM does to your input. The choice of algorithm shapes vocabulary size, sequence length, multilingual fairness, and even API pricing. Building these from scratch makes you appreciate why the modern stack converged on Byte-Level BPE with large vocabularies, and why production tokenizers like tiktoken and SentencePiece exist as separate libraries.
