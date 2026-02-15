# Tutorial Writing Guide — KAN Picbreedr

This guide defines how tutorials are written for this project. Follow these rules exactly.

---

## Audience

**Profile:** A developer who deeply understands ML theory, KAN math, CPPN concepts, and evolutionary computation, but is learning PyTorch syntax and idioms. They can read math notation fluently and know NumPy well.

**Adaptation rules for intermediate-learning-PyTorch level:**

- **Never explain:** neural network theory, what a loss function is, what backpropagation does conceptually, spline math, Kolmogorov-Arnold theorem, CPPN architecture concepts, evolutionary algorithms, NumPy operations, Python syntax
- **Always explain:** Every PyTorch-specific class, function, and idiom on first use. Give the "why" — the engineering reason PyTorch does it this way, not a hand-wave.
- **Bridge format:** When introducing a PyTorch concept, show the math or NumPy equivalent alongside it. Use comparison tables where helpful.
- **Code context:** Code fragments are OK when the surrounding context is clear, but every new PyTorch API should get a minimal isolated example (3-10 lines) before appearing in larger code.

---

## Section Weights

| Focus Area | Weight | What to Write |
|---|---|---|
| **conceptual** | heavy | Dedicated section. State what we're building in plain English. Relate to ML/KAN/CPPN concepts the reader already knows. Explain the "what" and "why" before any code. |
| **design_choices** | heavy | Dedicated section. For every PyTorch pattern: explain *why* PyTorch does it that way vs the math on paper. Give the real engineering reason. |
| **implementation** | heavy | Dedicated section. Line-by-line annotated code using the CONCEPT/WHY comment format. Walk through architecture with pseudocode mapping math notation to PyTorch calls. |
| **connections** | normal | Standard paragraph coverage. Show "math vs PyTorch" comparison tables. Mention NumPy equivalents. Link to previous tutorials when building on prior concepts. |
| **examples** | heavy | Dedicated section. Every new PyTorch concept gets a minimal isolated example (3-10 lines) that is runnable in isolation. Show input/output. |

---

## Style Rules (Narrative)

- Use a narrative walkthrough style: "First we... then we... because..."
- Structure as a learning journey, not a reference manual
- Bold the **key insight** in each section
- Use analogies to concepts the reader knows (math, ML theory, NumPy)
- When the reader might ask "why?" — answer it immediately, don't defer
- Prefer explicit over clever — write readable code even if slightly more verbose

---

## Annotated Code Format

When writing implementation walkthroughs, use this comment style:

```python
# CONCEPT: nn.Module is PyTorch's base class for anything with learnable parameters
# WHY: It auto-tracks parameters, handles GPU moves, save/load, etc.
class MyCPPN(nn.Module):
    # CONCEPT: __init__ defines the structure (layers, sizes)
    # The actual computation happens in forward()
    def __init__(self, ...):
```

Every code block in the "implementation" sections should use this format.

---

## Frontmatter

Every tutorial must start with YAML frontmatter:

```yaml
---
date: YYYY-MM-DD
summary: "One-line description of what this tutorial covers"
---
```

---

## Naming Convention

Pattern: `{nn}_{slug}.md` where `nn` is zero-padded sequence number and `slug` is kebab-case-with-underscores.

Examples from this project:
- `01_pytorch_foundations_for_cppns.md`
- `02_what_is_a_cppn_and_building_one_in_pytorch.md`
- `03_kolmogorov_arnold_networks_in_pytorch.md`

---

## Gotchas Section

Every tutorial must end with a "Gotchas & Common Mistakes" section listing things that are counterintuitive or easy to mess up. Format as a numbered list with bold headers.

---

## Quality Bar

After reading a tutorial, the reader should be able to:
1. **Explain** every PyTorch concept that appeared, in their own words
2. **Write** the implementation code from scratch (with the tutorial closed) for simple cases
3. **Debug** common mistakes related to the concepts covered
4. **Connect** the PyTorch code back to the underlying math
