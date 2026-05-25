"""Tests for the absolute positional encoding exercises.

Each `run_*_tests` function takes the user's implementation and runs all checks,
printing ✓ for each pass and ✗ with a reason for each failure. Designed so the
user sees the *full picture* of what's working — not just the first thing that
broke.
"""

import torch


def _run_checks(label, checks):
    """Run a list of (name, callable) checks. Print pass/fail for each, then a summary."""
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


def run_sinusoidal_tests(pe_fn):
    """Test a sinusoidal_pe(max_len, d_model) -> (max_len, d_model) implementation."""

    pe = pe_fn(10, 64)
    pe_long = pe_fn(1000, 64)

    def t_shape():
        assert pe.shape == (10, 64), f"got {tuple(pe.shape)}"

    def t_pos0_sin():
        assert torch.allclose(pe[0, 0::2], torch.zeros(32), atol=1e-6), \
            "PE[0, even cols] should all be sin(0) = 0"

    def t_pos0_cos():
        assert torch.allclose(pe[0, 1::2], torch.ones(32), atol=1e-6), \
            "PE[0, odd cols] should all be cos(0) = 1"

    def t_bounded():
        m = pe.abs().max().item()
        assert m <= 1.0 + 1e-6, f"max |value| = {m:.4f}, expected <= 1"

    def t_freq_ordering():
        var_low = pe_long[:, 0].var().item()
        var_high = pe_long[:, 62].var().item()
        assert var_low > var_high, \
            f"dim 0 should oscillate faster than dim 62 (var {var_low:.4f} vs {var_high:.4f})"

    def t_distinct_positions():
        assert not torch.allclose(pe[0], pe[1]), "PE[0] and PE[1] are equal"

    def t_rotation():
        d_model = 64
        local_pe = pe_fn(20, d_model)
        pos, k = 5, 7
        omegas = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Build block-diagonal R_k for the [sin, cos] convention: [[cos, sin], [-sin, cos]]
        R_k = torch.zeros(d_model, d_model)
        for f, omega in enumerate(omegas):
            angle = omega * k
            c, s = torch.cos(angle), torch.sin(angle)
            R_k[2 * f,     2 * f]     = c
            R_k[2 * f,     2 * f + 1] = s
            R_k[2 * f + 1, 2 * f]     = -s
            R_k[2 * f + 1, 2 * f + 1] = c
        rotated = R_k @ local_pe[pos]
        target = local_pe[pos + k]
        diff = (rotated - target).abs().max().item()
        assert torch.allclose(rotated, target, atol=1e-4), \
            f"R_k @ PE[{pos}] != PE[{pos + k}], max diff {diff:.2e}"

    return _run_checks("sinusoidal_pe", [
        ("Shape == (10, 64)",                          t_shape),
        ("PE[0, even cols] all 0 (sin of zero angle)", t_pos0_sin),
        ("PE[0, odd cols] all 1 (cos of zero angle)",  t_pos0_cos),
        ("All values bounded in [-1, 1]",              t_bounded),
        ("Low dims oscillate faster than high dims",   t_freq_ordering),
        ("Different positions have distinct codes",    t_distinct_positions),
        ("Rotation property: R_k @ PE[pos] == PE[pos+k]", t_rotation),
    ])


def run_sinusoidal_module_tests(SinusoidalPE):
    """Test a SinusoidalPositionalEmbedding(max_len, d_model) module."""

    max_len, d_model = 128, 64
    module = SinusoidalPE(max_len=max_len, d_model=d_model)

    def t_no_params():
        n = sum(p.numel() for p in module.parameters())
        assert n == 0, f"got {n} params; PE should be a buffer, not nn.Parameter"

    def t_buffer_in_state_dict():
        keys = list(module.state_dict().keys())
        assert len(keys) > 0, "state_dict() is empty — did you call register_buffer?"

    def t_forward_short():
        out = module(torch.zeros(2, 10, d_model))
        assert out.shape == (2, 10, d_model), \
            f"forward returned {tuple(out.shape)} for input (2, 10, {d_model})"

    def t_forward_full():
        out = module(torch.zeros(1, max_len, d_model))
        assert out.shape == (1, max_len, d_model), \
            f"forward at max_len={max_len} returned {tuple(out.shape)}"

    def t_correctness_at_zero_input():
        out = module(torch.zeros(1, 5, d_model))
        # At pos=0, even dims should be sin(0)=0 and odd dims cos(0)=1
        assert torch.allclose(out[0, 0, 0::2], torch.zeros(d_model // 2), atol=1e-6), \
            "forward(zeros)[batch=0, pos=0, even dims] != 0"
        assert torch.allclose(out[0, 0, 1::2], torch.ones(d_model // 2), atol=1e-6), \
            "forward(zeros)[batch=0, pos=0, odd dims] != 1"

    def t_device_move():
        module.to('cpu')
        for buf in module.buffers():
            assert buf.device.type == 'cpu', "Buffer didn't move with module.to('cpu')"

    return _run_checks("SinusoidalPositionalEmbedding", [
        ("Zero learnable parameters",                  t_no_params),
        ("Buffer registered in state_dict",            t_buffer_in_state_dict),
        ("Forward shape correct for seq_len < max_len", t_forward_short),
        ("Forward shape correct at full max_len",      t_forward_full),
        ("forward(zeros) returns the PE values",       t_correctness_at_zero_input),
        ("Buffer moves with module.to(device)",        t_device_move),
    ])


def run_learned_tests(LearnedPE):
    """Test a LearnedPositionalEmbedding(max_len, d_model) module."""

    max_len, d_model = 128, 64
    module = LearnedPE(max_len=max_len, d_model=d_model)

    def t_param_count():
        n = sum(p.numel() for p in module.parameters())
        assert n == max_len * d_model, \
            f"expected {max_len * d_model} params (max_len * d_model), got {n}"

    def t_forward_shape():
        out = module(torch.zeros(2, 10, d_model))
        assert out.shape == (2, 10, d_model), f"got {tuple(out.shape)}"

    def t_gradients_flow():
        module.zero_grad()
        out = module(torch.randn(2, 10, d_model))
        out.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in module.parameters()
        )
        assert has_grad, "no non-zero gradients on any parameter after backward"

    def t_boundary_failure():
        x_too_long = torch.zeros(1, max_len + 100, d_model)
        silently_ok = False
        try:
            out = module(x_too_long)
            silently_ok = (out.shape == (1, max_len + 100, d_model))
        except (IndexError, RuntimeError, AssertionError):
            pass  # expected — implementation correctly fails
        assert not silently_ok, \
            "forward(seq_len > max_len) returned a correct-shaped tensor; " \
            "should raise or produce wrong shape"

    return _run_checks("LearnedPositionalEmbedding", [
        ("Param count == max_len * d_model",       t_param_count),
        ("Forward shape correct",                   t_forward_shape),
        ("Gradients flow through position table",   t_gradients_flow),
        ("Boundary fails for seq_len > max_len",    t_boundary_failure),
    ])
