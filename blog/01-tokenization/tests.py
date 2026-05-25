"""Tests for the tokenization exercises.

Each `run_*_tests` function takes the user's implementation and runs all checks,
printing ✓ for each pass and ✗ with a reason for each failure.

Designed to be imported and called from the exercise notebooks.
"""

from collections import Counter


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


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------

def run_bpe_tests(BPETokenizer, pre_tokenize, get_pair_counts, merge_pair):
    """Test BPE building blocks and the full tokenizer."""

    def t_pretokenize_basic():
        out = pre_tokenize(["the cat sat", "the cat ate"])
        assert out.get(('t', 'h', 'e', '</w>'), 0) == 2, \
            f"'the' appears twice; got freq {out.get(('t', 'h', 'e', '</w>'), 0)}"
        assert out.get(('c', 'a', 't', '</w>'), 0) == 2, "'cat' should appear twice"
        assert out.get(('s', 'a', 't', '</w>'), 0) == 1, "'sat' should appear once"
        assert out.get(('a', 't', 'e', '</w>'), 0) == 1, "'ate' should appear once"

    def t_pretokenize_eow_marker():
        out = pre_tokenize(["hi"])
        word = list(out.keys())[0]
        assert word[-1] == '</w>', f"end-of-word marker missing; word ends with {word[-1]!r}"

    def t_pair_counts_basic():
        wf = {('c', 'a', 't', '</w>'): 2, ('s', 'a', 't', '</w>'): 1}
        out = get_pair_counts(wf)
        assert out.get(('a', 't'), 0) == 3, f"'a','t' should be 3 (2 from 'cat', 1 from 'sat'); got {out.get(('a','t'), 0)}"
        assert out.get(('c', 'a'), 0) == 2, "'c','a' should be 2"
        assert out.get(('s', 'a'), 0) == 1, "'s','a' should be 1"
        assert out.get(('t', '</w>'), 0) == 3, "'t','</w>' should be 3"

    def t_merge_pair_basic():
        wf = {('c', 'a', 't', '</w>'): 2, ('s', 'a', 't', '</w>'): 1, ('e', '</w>'): 1}
        out = merge_pair(('a', 't'), wf)
        assert out.get(('c', 'at', '</w>'), 0) == 2, f"after merging ('a','t'), 'cat' should become ('c','at','</w>'): 2"
        assert out.get(('s', 'at', '</w>'), 0) == 1, "'sat' should become ('s','at','</w>'): 1"
        # Words without the pair should be untouched
        assert out.get(('e', '</w>'), 0) == 1, "unrelated word should be unchanged"

    def t_merge_pair_no_match():
        wf = {('a', 'b', 'c'): 1}
        out = merge_pair(('x', 'y'), wf)
        assert out == wf, "merging a non-existent pair should leave word_freqs unchanged"

    def t_train_vocab_size():
        corpus = ["the cat sat", "the bat ran", "the dog ate"] * 5
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=20)
        assert len(tok.vocab) <= 20, f"vocab exceeded target: {len(tok.vocab)}"

    def t_train_includes_chars():
        corpus = ["abcabcabc"]
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=10)
        for c in 'abc':
            assert c in tok.vocab, f"single character {c!r} should be in vocab"

    def t_train_produces_merges():
        corpus = ["the cat sat", "the bat ran"] * 10
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=30)
        assert len(tok.merges) > 0, "no merges happened — training loop broken?"

    def t_encode_returns_list_of_strings():
        corpus = ["the cat sat", "the bat ran"] * 5
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=25)
        result = tok.encode("the cat")
        assert isinstance(result, list), "encode should return a list"
        assert all(isinstance(x, str) for x in result), "encode should return list of strings"

    def t_encode_compression():
        corpus = ["the cat sat", "the bat ran"] * 20
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=40)
        text = "the cat ran"
        tokens = tok.encode(text)
        # After training, encoding should produce fewer tokens than raw chars
        assert len(tokens) < len(text), \
            f"encoded length ({len(tokens)}) should be smaller than raw chars ({len(text)})"

    return _run_checks("BPE", [
        ("pre_tokenize: basic word frequencies",       t_pretokenize_basic),
        ("pre_tokenize: end-of-word marker present",   t_pretokenize_eow_marker),
        ("get_pair_counts: weighted by word frequency", t_pair_counts_basic),
        ("merge_pair: replaces pairs in matching words", t_merge_pair_basic),
        ("merge_pair: leaves non-matching words alone", t_merge_pair_no_match),
        ("train: vocab size respects target",          t_train_vocab_size),
        ("train: includes all single characters",      t_train_includes_chars),
        ("train: produces at least one merge",         t_train_produces_merges),
        ("encode: returns list of strings",            t_encode_returns_list_of_strings),
        ("encode: compresses below char count",        t_encode_compression),
    ])


# ---------------------------------------------------------------------------
# Byte-level BPE
# ---------------------------------------------------------------------------

def run_byte_level_bpe_tests(ByteLevelBPETokenizer):
    """Test the byte-level BPE tokenizer."""

    def t_initial_vocab_size():
        tok = ByteLevelBPETokenizer()
        assert len(tok.vocab) == 256, f"initial vocab should be 256 bytes; got {len(tok.vocab)}"
        for i in range(256):
            assert i in tok.vocab, f"byte {i} missing from initial vocab"

    def t_train_grows_vocab():
        tok = ByteLevelBPETokenizer()
        tok.train(["hello world"] * 20, vocab_size=300)
        assert len(tok.vocab) > 256, "training should add new merged tokens to vocab"
        assert len(tok.vocab) <= 300, f"vocab exceeded target 300: {len(tok.vocab)}"

    def t_encode_returns_ints():
        tok = ByteLevelBPETokenizer()
        tok.train(["hello world"] * 20, vocab_size=280)
        ids = tok.encode("hello")
        assert isinstance(ids, list), "encode should return a list"
        assert all(isinstance(x, int) for x in ids), "encode should return list of ints (token IDs)"

    def t_encode_handles_unicode():
        tok = ByteLevelBPETokenizer()
        tok.train(["hello world"], vocab_size=260)
        # Multi-byte input should still work — even if it fragments, no exceptions
        try:
            ids = tok.encode("नमस्ते 猫 🤖")
            assert all(isinstance(x, int) for x in ids)
        except Exception as e:
            raise AssertionError(f"failed on multilingual input: {e}")

    return _run_checks("Byte-Level BPE", [
        ("initial vocab is exactly 256 bytes",   t_initial_vocab_size),
        ("training grows vocab (within target)", t_train_grows_vocab),
        ("encode returns list of ints",          t_encode_returns_ints),
        ("encode handles multilingual input",    t_encode_handles_unicode),
    ])


# ---------------------------------------------------------------------------
# WordPiece
# ---------------------------------------------------------------------------

def run_wordpiece_tests(WordPieceTokenizer, get_pmi_scores):
    """Test WordPiece tokenizer."""

    def t_pmi_prefers_strong_association():
        # A pair where 'q' almost always precedes 'u' should score higher than
        # a pair where 'a' and 't' are common but only loosely associated.
        word_freqs = {
            # 'q' appears 5 times, 'u' appears 5 times, ('q','u') appears 5 times — perfect coupling
            ('q', 'u', 'e', 'e', 'n', '</w>'): 3,
            ('q', 'u', 'i', 'c', 'k', '</w>'): 2,
            # 'a' and 't' are common but appear in many contexts
            ('c', 'a', 't', '</w>'): 5,
            ('s', 'a', 't', '</w>'): 5,
            ('a', 't', 'e', '</w>'): 5,
            ('t', 'h', 'e', '</w>'): 10,
        }
        scores = get_pmi_scores(word_freqs)
        qu_score = scores.get(('q', 'u'), 0)
        at_score = scores.get(('a', 't'), 0)
        assert qu_score > at_score, \
            f"PMI should prefer ('q','u') over ('a','t'); got qu={qu_score}, at={at_score}"

    def t_train_populates_vocab():
        tok = WordPieceTokenizer()
        # vocab_size=32 leaves room for merges on top of the (~22-entry) seed
        # of bare + '##' forms for every character in the corpus.
        tok.train(["the cat sat", "the dog ran", "the cat sat"] * 10, vocab_size=32)
        assert len(tok.vocab) > 0, "training should populate vocab"
        assert len(tok.vocab) <= 32, f"vocab exceeded target: {len(tok.vocab)}"
        for c in 'tcdaeshnogr':  # chars actually present in the corpus
            assert c in tok.vocab, f"character {c!r} should be in vocab"

    def t_encode_returns_strings():
        tok = WordPieceTokenizer()
        tok.train(["the cat", "the dog"] * 10, vocab_size=15)
        out = tok.encode("the cat")
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)

    def t_encode_uses_continuation_prefix():
        # Train to get pieces; encode a multi-piece word and confirm '##' marks continuations
        tok = WordPieceTokenizer()
        tok.train(["unhappy unhappy happy happy hap"] * 5, vocab_size=12)
        out = tok.encode("unhappy")
        # If unhappy splits into multiple pieces, only the first should lack '##'
        if len(out) > 1:
            assert not out[0].startswith('##'), "first piece should not have '##' prefix"
            assert any(p.startswith('##') for p in out[1:]), \
                "continuation pieces should have '##' prefix"

    return _run_checks("WordPiece", [
        ("PMI prefers strong association over raw frequency", t_pmi_prefers_strong_association),
        ("train: vocab populated within target",              t_train_populates_vocab),
        ("encode: returns list of strings",                   t_encode_returns_strings),
        ("encode: uses '##' for continuation pieces",         t_encode_uses_continuation_prefix),
    ])
