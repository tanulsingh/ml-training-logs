"""Tests for the flash attention exercises."""

import math
import torch


def _run_checks(label, checks):
    print(f"Running {label}...")
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            msg = str(e) or type(e).__name__
            print(f"  ✗ {name} — {msg}")
    total = len(checks)
    if passed == total:
        print(f"\n  All {total} checks passed ✓\n")
    else:
        print(f"\n  {passed}/{total} checks passed — fix the ✗ above\n")
    return passed == total


def run_online_softmax_tests(online_softmax_fn):
    """Test online softmax against torch.softmax."""

    torch.manual_seed(0)

    def t_basic():
        x = torch.tensor([2.0, 8.0, 1.0, 9.0, 3.0, 7.0])
        ref = torch.softmax(x, dim=-1)
        out = online_softmax_fn(x, chunk_size=3)
        assert torch.allclose(out, ref, atol=1e-6), f"diff = {(out - ref).abs().max()}"

    def t_chunk_size_one():
        x = torch.randn(50)
        ref = torch.softmax(x, dim=-1)
        out = online_softmax_fn(x, chunk_size=1)
        assert torch.allclose(out, ref, atol=1e-6)

    def t_chunk_size_full():
        x = torch.randn(50)
        ref = torch.softmax(x, dim=-1)
        out = online_softmax_fn(x, chunk_size=50)
        assert torch.allclose(out, ref, atol=1e-6)

    def t_random_lengths():
        for _ in range(20):
            n = torch.randint(10, 200, (1,)).item()
            x = torch.randn(n) * 5
            ref = torch.softmax(x, dim=-1)
            for chunk in [1, 4, 16, n]:
                out = online_softmax_fn(x, chunk_size=chunk)
                assert torch.allclose(out, ref, atol=1e-5), \
                    f"failed at n={n}, chunk={chunk}, max diff {(out - ref).abs().max()}"

    def t_extreme_values():
        x = torch.tensor([100.0, -100.0, 50.0, -50.0])  # large dynamic range
        ref = torch.softmax(x, dim=-1)
        out = online_softmax_fn(x, chunk_size=2)
        assert torch.allclose(out, ref, atol=1e-6)

    return _run_checks("online_softmax", [
        ("Matches blog's example [2,8,1,9,3,7]",  t_basic),
        ("chunk_size=1 (every-element)",          t_chunk_size_one),
        ("chunk_size=full row (no chunking)",     t_chunk_size_full),
        ("Random lengths × chunk sizes",          t_random_lengths),
        ("Numerically stable on extreme values",  t_extreme_values),
    ])


def run_flash_attention_tests(flash_attn_fn):
    """Test that flash attention output matches standard attention."""

    torch.manual_seed(0)

    def reference_attention(Q, K, V, causal=False):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if causal:
            n = Q.shape[-2]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return weights @ V

    def t_match_no_causal():
        n, d = 256, 64
        Q = torch.randn(n, d); K = torch.randn(n, d); V = torch.randn(n, d)
        ref = reference_attention(Q, K, V)
        for bs in [16, 32, 64, 128]:
            out = flash_attn_fn(Q, K, V, block_size=bs)
            assert torch.allclose(out, ref, atol=1e-4), \
                f"block_size={bs}: diff = {(out - ref).abs().max()}"

    def t_block_size_independence():
        # Output should be the same regardless of block_size
        n, d = 96, 16
        Q = torch.randn(n, d); K = torch.randn(n, d); V = torch.randn(n, d)
        outs = [flash_attn_fn(Q, K, V, block_size=bs) for bs in [8, 16, 32, 48]]
        for i in range(1, len(outs)):
            assert torch.allclose(outs[0], outs[i], atol=1e-5), \
                "different block_sizes gave different outputs"

    return _run_checks("flash_attention", [
        ("Matches reference (non-causal)",           t_match_no_causal),
        ("Output independent of block_size",         t_block_size_independence),
    ])
