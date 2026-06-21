"""Tests for the attention sparsity exercises."""

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


def run_sliding_window_tests(mask_fn, SlidingWindowAttention):
    """Test sliding window mask and module."""

    torch.manual_seed(0)

    def t_mask_shape():
        m = mask_fn(6, 3)
        assert m.shape == (6, 6), f"got {tuple(m.shape)}"

    def t_mask_diagonal():
        m = mask_fn(6, 3)
        assert all(m[i, i].item() for i in range(6)), "diagonal must be True"

    def t_mask_window():
        m = mask_fn(6, 3)
        # position 3 should see positions 1, 2, 3 (within window of 3) but not 0
        assert m[3, 1].item() and m[3, 2].item() and m[3, 3].item(), \
            "position 3 should see [1, 2, 3]"
        assert not m[3, 0].item(), "position 3 should NOT see position 0 (distance 3 == window)"

    def t_mask_no_future():
        m = mask_fn(6, 3)
        assert not m[2, 4].item() and not m[2, 5].item(), "no future tokens"

    def t_module_shape():
        swa = SlidingWindowAttention(d_model=64, n_heads=8, window_size=4, max_len=32)
        out = swa(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64), f"got {tuple(out.shape)}"

    def t_locality():
        swa = SlidingWindowAttention(d_model=64, n_heads=8, window_size=3 , max_len=32)
        x_a = torch.randn(1, 8, 64); x_b = x_a.clone()
        x_b[0, 2] = torch.randn(64)  # change pos 2
        out_a = swa(x_a); out_b = swa(x_b)
        # Position 5 has window covering positions [3, 4, 5] → can't see position 2
        assert torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-5), \
            "position 5 should not depend on position 2 (window_size=3)"
        # Position 3 has window covering [1, 2, 3] → CAN see position 2
        assert not torch.allclose(out_a[0, 3], out_b[0, 3]), \
            "position 3 should depend on position 2"

    return _run_checks("SlidingWindowAttention", [
        ("Mask shape correct",                  t_mask_shape),
        ("Mask diagonal is True",               t_mask_diagonal),
        ("Mask respects window size",           t_mask_window),
        ("Mask is causal (no future)",          t_mask_no_future),
        ("Module output shape correct",         t_module_shape),
        ("Locality: distant tokens don't leak", t_locality),
    ])


def run_hybrid_tests(HybridAttentionStack):
    """Test hybrid global + local attention stack."""

    torch.manual_seed(0)

    def t_shape():
        s = HybridAttentionStack(d_model=64, n_heads=8, n_layers=6,max_len = 32,
                                 window_size=3, global_every=3)
        out = s(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64), f"got {tuple(out.shape)}"

    def t_param_count_reasonable():
        # Hybrid should have more params than all-sliding-window because
        # global layers have full d_model x d_model W_k, W_v
        s = HybridAttentionStack(d_model=64, n_heads=8, n_layers=4,max_len = 32,
                                 window_size=3, global_every=2)
        n = sum(p.numel() for p in s.parameters())
        assert n > 0, "no parameters?"

    def t_grad_flow():
        s = HybridAttentionStack(d_model=32, n_heads=4, n_layers=4,max_len = 32,
                                 window_size=3, global_every=2)
        out = s(torch.randn(1, 8, 32))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in s.parameters())
        assert has_grad

    return _run_checks("HybridAttentionStack", [
        ("Output shape correct",   t_shape),
        ("Has parameters",         t_param_count_reasonable),
        ("Gradients flow",         t_grad_flow),
    ])
