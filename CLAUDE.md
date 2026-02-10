# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Current State (as of 2026-02-10)

**What works (30 tests passing):**
- `Lattice2D` class with rectangular and circular constructors
- Forward mapping (world V → lattice L) and inverse mapping (L → V), fully vectorized
- Bilinear resampling in both directions via `grid_sample`
- Gradient magnitude energy (Sobel filters)
- Greedy and multi-greedy seam algorithms
- Seam removal
- End-to-end traditional carving (`carve_image_traditional`)
- End-to-end lattice-guided carving (`carve_image_lattice_guided`) — but see caveat below

**Known issue — naive double interpolation:**
The current `carve_image_lattice_guided` uses the naive approach: resample pixels
V→L, carve seams in L, resample L→V. This causes blurring from two rounds of
bilinear interpolation. The paper warns against this explicitly (Section 3.3, Fig. 6).

**What's not implemented yet (priority order):**
1. **"Carving the mapping" (Section 3.3)** — The paper's actual method. Instead of
   resampling pixel data twice, you: (a) map energy V→L, (b) find seam in L,
   (c) seam removal modifies the lattice mapping itself to produce g*, (d) final
   image pixels come from a single lookup: V*(p) = V_copy(g*(f(p))). This is the
   critical next step.
2. **Seam pairs (Section 3.6)** — Resize a local region without changing global
   boundaries. Two windows: region of interest shrinks, pair region expands
   (or vice versa). This is how you'd shrink a bagel hole while expanding background.
3. **Cyclic lattices (Section 3.5)** — For closed shapes (rings, tubes). Connect last
   lattice plane back to first. Cyclic greedy uses inverted Gaussian energy guide to
   steer seam back to start (Section 4.0.1, Fig. 12).
4. **Forward energy (Rubinstein et al. 2008)** — Better energy function that considers
   cost of introducing new edges after seam removal. Stub exists in `energy.py`.

## Coding Preferences

- Python with PyTorch (GPU-accelerated tensor ops)
- Vectorized operations over Python loops wherever possible
- Conda environment: `lattice-carving`
- Run commands with: `conda run -n lattice-carving ...`

## Testing

- pytest test suite in `tests/test_lattice.py`
- Run with: `conda run -n lattice-carving python -m pytest tests/ -v`
- Tests should verify algorithmic correctness, not just shapes
- Use deterministic seeds (`torch.manual_seed`) for reproducibility

## Documentation

- Keep DEVLOG.md updated with progress, decisions, and problems encountered
- Document algorithmic choices and their rationale
- Include references to the background paper
- Commit often with descriptive messages

## Architecture

- `src/lattice.py` — Lattice2D class: construction, forward/inverse mapping, resampling
- `src/energy.py` — Energy functions (gradient magnitude, forward energy)
- `src/seam.py` — Greedy seam computation, seam removal
- `src/carving.py` — High-level carving orchestration
- `examples/` — Visual test scripts (not automated)
- `tests/` — pytest suite (30 tests, all passing)

## Key Algorithmic Notes

- Greedy seam is sensitive to first-row energy (border artifacts from Sobel padding).
  Multi-greedy helps by trying multiple starting points.
- Circular lattice forward_mapping needs tangent-projection penalty to disambiguate
  between a radial scanline and its 180° opposite. Already implemented.
- The lattice forward/inverse mappings are fully vectorized (batched tensor ops),
  replacing earlier O(N*n_lines) Python loops.
