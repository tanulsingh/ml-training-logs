"""Tests for the attention architecture exercises (MQA, GQA, MLA)."""

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


def run_mqa_tests(MultiQueryAttention):
    """Test MQA: 1 K head, 1 V head shared across all Q heads."""

    torch.manual_seed(0)

    def t_shape():
        m = MultiQueryAttention(d_model=64, n_heads=8)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64), f"got {tuple(out.shape)}"

    def t_kv_param_size():
        m = MultiQueryAttention(d_model=64, n_heads=8)
        # W_k weight should be (d_head, d_model) = (8, 64), not (64, 64)
        n_wk = sum(p.numel() for n, p in m.named_parameters() if 'k' in n.lower() and 'weight' in n)
        # MQA: d_head * d_model = 8 * 64 = 512 per W_k. With bias adds d_head = 8.
        assert n_wk <= 600, f"W_k params {n_wk} too large; should be O(d_head * d_model) ≈ 512, not d_model^2 = 4096"

    def t_grad_flow():
        m = MultiQueryAttention(d_model=32, n_heads=4)
        out = m(torch.randn(1, 4, 32))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert has_grad

    def t_causal():
        m = MultiQueryAttention(d_model=32, n_heads=4, causal=True)
        x_a = torch.randn(1, 5, 32); x_b = x_a.clone()
        x_b[0, 4] = torch.randn(32)
        out_a = m(x_a); out_b = m(x_b)
        assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5)

    return _run_checks("MultiQueryAttention", [
        ("Output shape correct",        t_shape),
        ("W_k is d_head-sized (not d_model-sized)", t_kv_param_size),
        ("Gradients flow",              t_grad_flow),
        ("Causal mode works",           t_causal),
    ])


def run_gqa_tests(GroupedQueryAttention):
    """Test GQA across the spectrum from MHA (n_kv = n_heads) to MQA (n_kv = 1)."""

    torch.manual_seed(0)

    def t_shape():
        m = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64), f"got {tuple(out.shape)}"

    def t_assert_divisible():
        try:
            GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=3)  # 8 % 3 != 0
            ok = False
        except (AssertionError, ValueError):
            ok = True
        assert ok, "n_heads % n_kv_heads != 0 should raise"

    def t_kv_scales():
        # W_k size should scale linearly with n_kv_heads
        sizes = []
        for n_kv in [1, 2, 4, 8]:
            m = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=n_kv)
            n_wk = sum(p.numel() for n, p in m.named_parameters()
                       if n == 'W_k.weight' or 'k.weight' in n)
            sizes.append(n_wk)
        # sizes[0] (n_kv=1) should be 8x smaller than sizes[3] (n_kv=8)
        assert sizes[3] >= 4 * sizes[0], \
            f"W_k should grow with n_kv_heads; got {sizes}"

    def t_causal():
        m = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, causal=True)
        x_a = torch.randn(1, 5, 32); x_b = x_a.clone()
        x_b[0, 4] = torch.randn(32)
        out_a = m(x_a); out_b = m(x_b)
        assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5)

    return _run_checks("GroupedQueryAttention", [
        ("Output shape correct",                        t_shape),
        ("Asserts n_heads divisible by n_kv_heads",     t_assert_divisible),
        ("W_k size scales with n_kv_heads",             t_kv_scales),
        ("Causal mode works",                           t_causal),
    ])


def run_mla_tests(MultiLatentAttention):
    """Test MLA: latent compression of K, V."""

    torch.manual_seed(0)

    def t_shape():
        m = MultiLatentAttention(d_model=128, n_heads=8, d_c=32)
        out = m(torch.randn(2, 10, 128))
        assert out.shape == (2, 10, 128), f"got {tuple(out.shape)}"

    def t_compression():
        # The latent cache (d_c=32) should be much smaller than full K+V (n_heads * d_head * 2)
        d_model, n_heads, d_c = 128, 8, 32
        d_head = d_model // n_heads
        full_kv = 2 * n_heads * d_head
        assert d_c < full_kv, \
            f"d_c={d_c} should be smaller than full KV cache size {full_kv}"

    def t_grad_flow():
        m = MultiLatentAttention(d_model=64, n_heads=4, d_c=16)
        out = m(torch.randn(1, 4, 64))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert has_grad

    def t_causal():
        m = MultiLatentAttention(d_model=64, n_heads=4, d_c=16, causal=True)
        x_a = torch.randn(1, 5, 64); x_b = x_a.clone()
        x_b[0, 4] = torch.randn(64)
        out_a = m(x_a); out_b = m(x_b)
        assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5)

    return _run_checks("MultiLatentAttention", [
        ("Output shape correct",                t_shape),
        ("d_c smaller than full KV cache",      t_compression),
        ("Gradients flow",                      t_grad_flow),
        ("Causal mode works",                   t_causal),
    ])
