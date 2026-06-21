"""Tests for the transformer anatomy exercises."""

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


def run_ffn_tests(StandardFFN, GELUFFN, SwiGLU):
    """Test FFN variants — shapes match, param counts roughly equal."""

    torch.manual_seed(0)

    def t_shape_standard():
        m = StandardFFN(d_model=64)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64), f"got {tuple(out.shape)}"

    def t_shape_gelu():
        m = GELUFFN(d_model=64)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def t_shape_swiglu():
        m = SwiGLU(d_model=64)
        out = m(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def t_swiglu_three_matrices():
        m = SwiGLU(d_model=64)
        n_linear = sum(1 for mod in m.modules() if isinstance(mod, torch.nn.Linear))
        assert n_linear == 3, f"SwiGLU should have 3 Linear layers, got {n_linear}"

    def t_param_counts_similar():
        d = 256
        n_std = sum(p.numel() for p in StandardFFN(d).parameters())
        n_swi = sum(p.numel() for p in SwiGLU(d).parameters())
        # Should be within 10-20% of each other (SwiGLU's d_ff is shrunk)
        ratio = n_swi / n_std
        assert 0.7 < ratio < 1.3, \
            f"SwiGLU param count ({n_swi}) should be similar to StandardFFN ({n_std}), got ratio {ratio:.2f}"

    def t_grad_flow():
        for cls, name in [(StandardFFN, "Standard"), (GELUFFN, "GELU"), (SwiGLU, "SwiGLU")]:
            m = cls(d_model=32)
            out = m(torch.randn(1, 4, 32))
            out.sum().backward()
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
            assert has_grad, f"{name}: no grads"

    return _run_checks("FFN variants", [
        ("StandardFFN shape correct",     t_shape_standard),
        ("GELUFFN shape correct",         t_shape_gelu),
        ("SwiGLU shape correct",          t_shape_swiglu),
        ("SwiGLU has 3 Linear layers",    t_swiglu_three_matrices),
        ("Param counts roughly equal",    t_param_counts_similar),
        ("Gradients flow in all variants", t_grad_flow),
    ])


def run_norm_tests(LayerNorm, RMSNorm):
    """Test LayerNorm and RMSNorm."""

    torch.manual_seed(0)

    def t_layernorm_shape():
        ln = LayerNorm(64)
        out = ln(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def t_layernorm_zero_mean():
        ln = LayerNorm(64)
        x = torch.randn(2, 10, 64) * 5 + 3
        out = ln(x)
        means = out.mean(dim=-1)
        assert (means.abs() < 1e-5).all(), f"output should have zero mean per token, got max abs {means.abs().max()}"

    def t_layernorm_unit_var():
        ln = LayerNorm(64)
        x = torch.randn(2, 10, 64) * 5 + 3
        out = ln(x)
        vars = out.var(dim=-1, unbiased=False)
        assert (vars - 1).abs().max() < 1e-3, f"output should have unit variance, got {vars}"

    def t_rmsnorm_shape():
        rn = RMSNorm(64)
        out = rn(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def t_rmsnorm_unit_rms():
        rn = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 5
        out = rn(x)
        # RMS along last dim should be ~1
        rms = (out ** 2).mean(dim=-1).sqrt()
        assert (rms - 1).abs().max() < 1e-3, f"RMS should be ~1, got {rms}"

    def t_rmsnorm_fewer_params():
        ln = LayerNorm(64)
        rn = RMSNorm(64)
        n_ln = sum(p.numel() for p in ln.parameters())
        n_rn = sum(p.numel() for p in rn.parameters())
        assert n_rn < n_ln, f"RMSNorm should have fewer params (no beta); got {n_rn} vs LayerNorm's {n_ln}"

    return _run_checks("Normalization", [
        ("LayerNorm shape correct",            t_layernorm_shape),
        ("LayerNorm output: zero mean",        t_layernorm_zero_mean),
        ("LayerNorm output: unit variance",    t_layernorm_unit_var),
        ("RMSNorm shape correct",              t_rmsnorm_shape),
        ("RMSNorm output: unit RMS",           t_rmsnorm_unit_rms),
        ("RMSNorm has fewer params (no beta)", t_rmsnorm_fewer_params),
    ])


def run_transformer_tests(TransformerBlock, TransformerLM):
    """Test transformer block and full model."""

    torch.manual_seed(0)

    def t_block_shape():
        b = TransformerBlock(d_model=64, n_heads=8)
        out = b(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def t_block_grad():
        b = TransformerBlock(d_model=32, n_heads=4)
        out = b(torch.randn(1, 4, 32))
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters())
        assert has_grad

    def t_model_shape():
        m = TransformerLM(vocab_size=1000, d_model=64, n_heads=4, n_layers=2)
        tokens = torch.randint(0, 1000, (2, 16))
        logits = m(tokens)
        assert logits.shape == (2, 16, 1000), f"got {tuple(logits.shape)}"

    def t_model_grad():
        m = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
        tokens = torch.randint(0, 100, (1, 8))
        logits = m(tokens)
        logits.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert has_grad

    def t_causality():
        m = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
        tokens_a = torch.randint(0, 100, (1, 16))
        tokens_b = tokens_a.clone()
        tokens_b[0, 10:] = 99  # change future tokens
        logits_a = m(tokens_a)
        logits_b = m(tokens_b)
        assert torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-5), \
            "causality broken: changing future tokens affected past logits"

    return _run_checks("Transformer model", [
        ("Block output shape correct",  t_block_shape),
        ("Block gradients flow",        t_block_grad),
        ("Model output shape correct",  t_model_shape),
        ("Model gradients flow",        t_model_grad),
        ("Causality holds end-to-end",  t_causality),
    ])
