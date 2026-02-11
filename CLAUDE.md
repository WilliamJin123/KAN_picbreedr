# Project: KAN Picbreedr

## PyTorch Tutorial-First Rule (MANDATORY)

Before writing ANY PyTorch implementation code, you MUST first produce a **tutorial writeup** that:

1. **States what we're building** in plain English, relating it to ML/KAN/CPPN concepts the user already knows
2. **Explains every new PyTorch concept** that will appear in the implementation:
   - What the class/function does conceptually
   - Why PyTorch does it that way (vs. the math on paper)
   - A minimal isolated example (3-10 lines) showing JUST that concept
3. **Walks through the architecture** with a diagram or pseudocode mapping math notation to PyTorch calls
4. **Annotates the planned code** line-by-line before writing the real file - use this format:
   ```python
   # CONCEPT: nn.Module is PyTorch's base class for anything with learnable parameters
   # WHY: It auto-tracks parameters, handles GPU moves, save/load, etc.
   class MyCPPN(nn.Module):
       # CONCEPT: __init__ defines the structure (layers, sizes)
       # The actual computation happens in forward()
       def __init__(self, ...):
   ```
5. **Calls out gotchas** - things that are counterintuitive or easy to mess up

Save tutorial writeups to `tutorials/` directory with descriptive names (e.g., `tutorials/01_pytorch_tensors_and_modules.md`). Tutorials should also include the date written and a one-line summary as a yaml frontmatter.

Only after the user has read and acknowledged the tutorial should you proceed to write implementation code.

## Tutorial Style Guidelines

- Use analogies to concepts the user knows (math, ML theory, NumPy if helpful)
- Always show "what this looks like in math" vs "what this looks like in PyTorch"
- Keep code snippets short and runnable in isolation
- Bold the **key insight** in each section
- When the user asks a question, answer it thoroughly with examples - don't rush to get back to implementation
- Number tutorials sequentially so there's a clear learning path

## Project Context

- This project explores Fractured Entangled Representations (FER) using CPPNs and KANs
- Reference implementation is in `references/fer/` (JAX/Flax) - we are porting to PyTorch
- KAN reference code is in `references/Kolmogorov-Arnold_Networks_(KAN).ipynb` (already PyTorch)
- The user understands: neural network theory, KAN math (spline functions, Kolmogorov-Arnold theorem), CPPN concepts, swarm/memetic algorithms, evolutionary computation
- The user is learning: PyTorch syntax, PyTorch idioms, translating math to PyTorch code

## General Rules

- When the user asks "why does PyTorch do X this way?" - give the real engineering reason, not a hand-wave
- When the user asks about a code pattern, show the minimal example first, then the full context
- If a PyTorch concept has a NumPy equivalent, mention it
- Prefer explicit over clever - write readable code even if slightly more verbose
