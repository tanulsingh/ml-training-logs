# The Template Method Pattern (with no-op defaults)

> A note for future-me. The pattern that makes the PE-method-as-`nn.Module` design clean.

## What is a "no-op"?

**No-op** = "no operation". A function that does nothing useful — returns its input unchanged, or returns nothing at all.

```python
def no_op(x):
    return x        # does nothing, just hands back what it got
```

Why would you want a function that does nothing? Because **sometimes "do nothing" IS the right behavior**. Having a default no-op lets you call the function unconditionally, without checking whether anything needs to happen.

```python
# Without a no-op default — branching at every call site:
if some_step is not None:
    x = some_step(x)

# With a no-op default — just call:
x = some_step(x)        # if some_step is no-op, this is harmless
```

The second style is cleaner once you have many such optional steps.

---

## The Template Method Pattern

A classic OOP pattern. Use it when:

1. You have multiple variants of a process
2. The variants share the same overall structure
3. They differ only at **specific, well-defined points**

The pattern:
- Base class defines the **fixed algorithm structure** (the "template")
- Within that structure, it calls **hook methods** that subclasses override
- Hooks have **default implementations** (often no-ops) so subclasses only override what they actually change

### A simple example: document processing

You're processing documents. The structure is always the same:

```
load → clean → save
```

But CSV, JSON, and XML each do these steps differently. And XML doesn't actually need cleaning.

### The naive (bad) approach

```python
def process_document(doc_type, path):
    if doc_type == "csv":
        data = load_csv(path)
        cleaned = clean_csv(data)
        save_csv(cleaned, path + ".out")
    elif doc_type == "json":
        data = load_json(path)
        cleaned = clean_json(data)
        save_json(cleaned, path + ".out")
    elif doc_type == "xml":
        data = load_xml(path)
        # no cleaning needed for xml
        save_xml(data, path + ".out")
```

Problems:
- Fixed structure is buried in `if/elif`
- Adding a 4th type means editing this function
- XML branch is awkward — has to skip the clean step

### The Template Method approach

Pull the **structure** into a base class. Let subclasses fill in the *steps*.

```python
class DocProcessor:
    # The TEMPLATE — fixed algorithm. Never overridden.
    def process(self, path):
        data = self.load(path)
        cleaned = self.clean(data)
        self.save(cleaned, path + ".out")

    # Hooks — subclasses override these
    def load(self, path):
        raise NotImplementedError("subclasses must implement load")

    def clean(self, data):
        return data            # ← NO-OP DEFAULT: pass through

    def save(self, data, path):
        raise NotImplementedError("subclasses must implement save")
```

Subclasses:

```python
class CsvProcessor(DocProcessor):
    def load(self, path):
        return pd.read_csv(path)

    def clean(self, data):
        return data.dropna()    # CSV-specific cleaning

    def save(self, data, path):
        data.to_csv(path)


class XmlProcessor(DocProcessor):
    def load(self, path):
        return ET.parse(path)

    # ← clean NOT overridden. XmlProcessor inherits the no-op.

    def save(self, data, path):
        data.write(path)
```

Caller code is identical for all variants:

```python
processor = CsvProcessor()       # or XmlProcessor()
processor.process("data.csv")
```

No branching. Each subclass is self-contained.

### What's happening

| Element | Role |
|---|---|
| `process()` | The **template** — fixed structure. Calls hooks in order. |
| `load`, `clean`, `save` | **Hooks** — variable steps subclasses override. |
| `clean`'s no-op default | Optional hook — XmlProcessor doesn't need to override it. |
| `load`, `save` raising NotImplementedError | Mandatory hooks — every subclass must provide. |

---

## Why no-op defaults are clever

Without no-op defaults, every subclass would have to override every hook — even when there's nothing to do. XmlProcessor would need:

```python
def clean(self, data):
    return data    # nothing to do here
```

Boilerplate in every subclass.

With the no-op default in the base class, XmlProcessor inherits "do nothing" for free. The base class essentially says: *"here's a safe default — pass through unchanged. Override only if you actually need something different."*

This makes the no-op pattern extra valuable when:
- Many subclasses don't need the hook
- The hook applies to "optional decoration" of the algorithm
- You want each subclass's code to highlight only its **differences**, not restate sameness

---

## How this applied to the PE-method design

Same pattern, different domain.

The **fixed structure** (template) is TinyTransformer's forward pass:

```python
def forward(self, token_ids):
    positions = torch.arange(token_ids.shape[1])

    x = self.token_embedding(token_ids)
    x = self.pe_method.apply_to_input(x, positions)             # hook 1

    for layer in self.layers:
        # inside attention:
        # Q, K = self.pe_method.apply_to_qk(Q, K, positions)        # hook 2
        # scores = self.pe_method.apply_to_scores(scores, positions) # hook 3
        x = layer(x)

    return self.lm_head(x)
```

Three **hooks**, all with no-op defaults:

```python
class PEMethod(nn.Module):
    def apply_to_input(self, token_emb, positions):
        return token_emb        # no-op default
    def apply_to_qk(self, q, k, positions):
        return q, k             # no-op default
    def apply_to_scores(self, scores, positions):
        return scores           # no-op default
```

Subclasses override **only the hook their math intervenes at**:

| PE method | Overrides | Reason |
|---|---|---|
| `Sinusoidal` | `apply_to_input` | Adds PE to embeddings; doesn't touch attention |
| `Learned` | `apply_to_input` | Same as sinusoidal |
| `RoPE` | `apply_to_qk` | Rotates Q and K; doesn't touch embeddings or scores |
| `ALiBi` | `apply_to_scores` | Adds bias to scores; doesn't touch embeddings or Q, K |

`PEMethod()` itself (no overrides) is a complete, valid PE method — it's "no positional encoding at all," useful as a baseline.

---

## Why the pattern wins here

1. **TinyTransformer's forward never branches on PE type.** No `if isinstance(pe, RoPE)`. Ever.
2. **Adding a new PE method is local.** Subclass `PEMethod`, override the relevant hook. Zero changes to TinyTransformer or any other PE method.
3. **The taxonomy of PE methods becomes enforceable in code.** "Input-side vs attention-side" is an organizational principle in the blog post — the interface makes that taxonomy structural. You literally *can't* mix up which slot RoPE goes in because it overrides `apply_to_qk`, not `apply_to_input`.
4. **Eliminates `if pe_method is None` checks.** The no-op `PEMethod()` is always safe to call.

---

## When to use this pattern

**Good fit:**
- All variants share the same overall structure
- Variants differ only at specific, well-known points
- The set of hook points is small and stable
- Many subclasses won't need every hook

**Bad fit:**
- Variants have fundamentally different control flow (e.g., one needs an extra loop)
- Hook points keep changing as you add variants
- Only one or two variants exist (overkill — just write two functions)
- The structure itself isn't really fixed

---

## Related patterns (one-liners)

- **Strategy pattern**: like Template Method but you compose with the variant object instead of inheriting from it. `pe_method` as a constructor arg is actually the Strategy pattern; the *internal* hook structure is Template Method. They compose well.
- **Decorator pattern**: for *layering* behavior, not *substituting* it. If you wanted to chain multiple PE methods on top of each other, that'd be Decorator territory.
- **Hook system / plugin system**: a generalization of Template Method where the hook points are dynamically registered rather than declared in a base class. Overkill for our PE case.

---

## TL;DR

A **template method** is a method whose structure is fixed but whose individual steps can be customized by subclasses. Steps with **no-op defaults** are optional — subclasses only override the ones they need. Result: each subclass focuses on its own *differences*, not on restating the shared *sameness*. Calling code stays uniform regardless of which variant is in use.

This is exactly why the PE-method design works without if/else branches in TinyTransformer. The "where does this PE intervene?" decision lives in the PE class, not in the transformer.
