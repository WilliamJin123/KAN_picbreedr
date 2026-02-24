# KAN Picbreeder Analysis Suite — Design Document

**Date:** 2026-02-24
**Goal:** Comprehensive analysis and visualization comparing KAN variants, standard NNs, and CPPNs.

## Output Format

- **Primary:** Jupyter notebook (`notebooks/kan_analysis.ipynb`) with inline plots and narrative
- **Secondary:** Auto-generated HTML report via nbconvert for sharing

## File Structure

```
analysis/
  __init__.py
  benchmark.py        — Train all methods, collect timing/loss/iteration data
  spline_inspector.py — Extract learned spline curves, fit known functions
  comparison.py       — MSE, SSIM, feature similarity metrics
  text_prototype.py   — 1D sequence generation with KAN-CPPN
notebooks/
  kan_analysis.ipynb  — Main analysis notebook (6 sections)
  export_html.py      — nbconvert wrapper
```

## Notebook Sections

### Section 1: Method Comparison (Who Wins?)

**Methods trained on each genome (skull, butterfly, apple):**
1. Standard MLP-CPPN + SGD (fresh, from scratch)
2. KAN-CPPN + SGD
3. SwarmKAN (PSO + SGD)
4. MemeticKAN (NES + SGD)
5. Pre-trained picbreeder/SGD genomes (reference bounds)

**Metrics:** MSE over iterations, wall-clock time, final SSIM
**Visuals:** Convergence curves (overlaid), bar charts (final metrics), time-to-threshold

### Section 2: How Close Are the Learned Functions?

- Extract learned spline curve for every (layer, in, out) edge
- Compare against CPPN activation functions (gaussian, sin, sigmoid, tanh, identity)
- **Metrics:** L2 distance between spline and known activation over [-3, 3]
- **Visuals:** Grid of learned vs. known function plots, heatmap of best-fit activations

### Section 3: What Did the Splines Learn? (Visual + Numerical)

- Top-K most impactful spline edges (by sweep variance)
- Plot spline shape, closest known function, residual, image impact strip
- **Visuals:** Multi-panel figure per edge

### Section 4: Can This Scale to Text?

- Theoretical analysis of spatial-to-sequential constraints
- Toy prototype: KAN-CPPN generating 1D character probability distribution
- Discussion of positional encoding, attention vs. coordinate inputs
- **Visuals:** 1D signal output, coordinate system comparison

### Section 5: How Swarm & Memetic KAN Work

- Plain-English walkthrough with algorithm flowcharts
- SwarmKAN: PSO velocity update interleaved with SGD
- MemeticKAN: Antithetic NES gradient + SGD refinement
- Spline type: Linear interpolation on fixed grid (20 knots)
- **Visuals:** Algorithm flowcharts, parameter evolution snapshots

### Section 6: Design Choices That Mattered (Transferable Insights)

**Inductive biases with ablation evidence:**
1. Orthogonal init for base_weight
2. Residual base path (base + spline)
3. Sigmoid normalization for grid lookup
4. Gradient normalization in SGD
5. Excluding base_weight from ES
6. Antithetic sampling for variance reduction
7. Fresh optimizer per generation

**For each:** what it does, why it helps, when to use elsewhere
**Visuals:** Ablation plots, signal propagation heatmaps

## Benchmarking Details

- Identical conditions per method: target image, hidden size, layers, random seed
- Wall-clock timing via `time.perf_counter()`
- Loss recorded every N iterations
- 3 random seeds for variance bars

## Spline Inspection Details

- Extract coefficients from `KANCPPNLayer.coeffs`
- Reconstruct curve at 1000 points in [0,1] via sigmoid normalization
- Fit against: identity, sin, cos, tanh, sigmoid, gaussian, relu
- Report best-fit + L2 residual

## Metrics

- MSE (pixel-level)
- SSIM (structural similarity)
- Feature map cosine similarity (internal representations)

## Scope Exclusions

- No changes to core `src/` modules unless bugs found
- Old `experiments/` scripts left in place (superseded by notebook)
- No backward compatibility concerns
