"""Tests for the attention mechanism exercises.

Each `run_*_tests` function takes the user's implementation and runs all checks,
printing ✓ for each pass and ✗ with a reason for each failure.
"""

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


def run_scaled_dot_product_tests(attn_fn, SelfAttention):
    """Test scaled_dot_product_attention(Q, K, V) and SelfAttention(d_model)."""

    torch.manual_seed(0)

    def t_shape():
        Q = torch.randn(4, 8); K = torch.randn(4, 8); V = torch.randn(4, 8)
        out = attn_fn(Q, K, V)
        assert out.shape == (4, 8), f"got {tuple(out.shape)}"

    def t_uniform_q():
        # When all Q rows are identical, every output row is the same weighted sum of V
        Q = torch.ones(3, 4)
        K = torch.randn(3, 4)
        V = torch.randn(3, 4)
        out = attn_fn(Q, K, V)
        assert torch.allclose(out[0], out[1], atol=1e-5) and torch.allclose(out[0], out[2], atol=1e-5), \
            "uniform Q should produce identical output rows"

    def t_batched():
        Q = torch.randn(2, 4, 8); K = torch.randn(2, 4, 8); V = torch.randn(2, 4, 8)
        out = attn_fn(Q, K, V)
        assert out.shape == (2, 4, 8), f"got {tuple(out.shape)}"

    def t_module_shape():
        m = SelfAttention(d_model=64)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64), f"got {tuple(out.shape)}"

    def t_module_grad():
        m = SelfAttention(d_model=32)
        out = m(torch.randn(1, 4, 32))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert has_grad, "no gradients on SelfAttention parameters"

    return _run_checks("scaled dot-product attention", [
        ("Output shape (4, 8)",                t_shape),
        ("Batched: (2, 4, 8) input ok",        t_batched),
        ("Uniform Q → identical output rows",  t_uniform_q),
        ("SelfAttention output shape correct", t_module_shape),
        ("Gradients flow through SelfAttention", t_module_grad),
    ])


def run_causal_mask_tests(causal_mask_fn, attn_fn):
    """Test causal_mask(seq_len) and scaled_dot_product_attention(Q, K, V, mask)."""

    torch.manual_seed(0)

    def t_mask_shape():
        m = causal_mask_fn(4)
        assert m.shape == (4, 4), f"got {tuple(m.shape)}"

    def t_mask_diagonal():
        m = causal_mask_fn(5)
        assert all(m[i, i].item() for i in range(5)), "diagonal must be True (token sees itself)"

    def t_mask_upper_triangle():
        m = causal_mask_fn(5)
        assert m[0, 1].item() is False, "token 0 must NOT see token 1"
        assert m[2, 3].item() is False, "token 2 must NOT see token 3"

    def t_mask_lower_triangle():
        m = causal_mask_fn(5)
        assert m[3, 1].item() is True, "token 3 must see token 1"
        assert m[4, 0].item() is True, "token 4 must see token 0"

    def t_mask_zeros_future():
        Q = torch.randn(4, 8); K = torch.randn(4, 8); V = torch.randn(4, 8)
        mask = causal_mask_fn(4)
        # Modify V's last row; first 3 outputs must be unchanged
        out_a = attn_fn(Q, K, V, mask)
        V_modified = V.clone()
        V_modified[3] = torch.randn(8)
        out_b = attn_fn(Q, K, V_modified, mask)
        assert torch.allclose(out_a[:3], out_b[:3], atol=1e-5), \
            "changing V[3] should not affect output[0..2] under causal mask"
        assert not torch.allclose(out_a[3], out_b[3]), \
            "changing V[3] should affect output[3]"

    return _run_checks("causal mask + masked attention", [
        ("Mask shape (4, 4)",                       t_mask_shape),
        ("Mask diagonal is True (self-attention)",  t_mask_diagonal),
        ("Mask upper triangle is False",            t_mask_upper_triangle),
        ("Mask lower triangle is True",             t_mask_lower_triangle),
        ("Future tokens don't leak past",           t_mask_zeros_future),
    ])


def run_multi_head_tests(MultiHeadAttention):
    """Test MultiHeadAttention(d_model, n_heads, causal=False) module."""

    torch.manual_seed(0)

    def t_shape():
        m = MultiHeadAttention(d_model=64, n_heads=8)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64), f"got {tuple(out.shape)}"

    def t_param_count():
        m = MultiHeadAttention(d_model=64, n_heads=8)
        n = sum(p.numel() for p in m.parameters())
        # 4 linear layers (W_q, W_k, W_v, W_o), each 64*64 + 64 (with bias) = 4160
        # Without bias: 4 * 64 * 64 = 16384
        # Allow either convention — just check it's in a reasonable range
        assert 16000 <= n <= 17000, f"unexpected param count {n}; expected ~16384 (no bias) or ~16640 (with bias)"

    def t_grad_flow():
        m = MultiHeadAttention(d_model=32, n_heads=4)
        out = m(torch.randn(1, 4, 32))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert has_grad, "no gradients on MultiHeadAttention parameters"

    def t_causal():
        m = MultiHeadAttention(d_model=32, n_heads=4, causal=True)
        x_a = torch.randn(1, 5, 32)
        x_b = x_a.clone()
        x_b[0, 4] = torch.randn(32)  # change last token
        out_a = m(x_a); out_b = m(x_b)
        assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5), \
            "causal: changing token 4 must not affect outputs at positions 0..3"

    def t_assertion_on_bad_d_model():
        try:
            MultiHeadAttention(d_model=63, n_heads=8)  # not divisible
            ok = False
        except (AssertionError, ValueError):
            ok = True
        assert ok, "d_model not divisible by n_heads should raise"

    return _run_checks("MultiHeadAttention", [
        ("Output shape correct",                  t_shape),
        ("Parameter count in expected range",     t_param_count),
        ("Gradients flow",                        t_grad_flow),
        ("Causal mode: future doesn't leak",      t_causal),
        ("Asserts d_model divisible by n_heads",  t_assertion_on_bad_d_model),
    ])
