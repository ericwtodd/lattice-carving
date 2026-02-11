# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Current State (as of 2026-02-11)

**What works (30 tests passing):**
- `Lattice2D` class with rectangular and circular constructors
- Forward mapping (world V → lattice L) and inverse mapping (L → V), fully vectorized
- Bilinear resampling in both directions via `grid_sample`
- Gradient magnitude energy (Sobel filters)
- Greedy and multi-greedy seam algorithms
- Seam removal
- End-to-end traditional carving (`carve_image_traditional`)
- End-to-end lattice-guided carving (`carve_image_lattice_guided`) — but see caveat below

**Known issues:**
1. ✅ FIXED: Cumulative shift bug - was computing shifts against original u_map instead
   of accounting for previous shifts. Now tracks cumulative_shift correctly.
2. ⚠️ DEBUGGING: Lattice-guided carving still produces distorted output despite bug fix
   - Visualization framework created to debug step-by-step
   - Need to verify: lattice structure, energy resampling, seam interpolation, warping
3. ⚠️ Tests don't validate correctness - they only check shapes/types, pass even when
   output is severely distorted

**What's implemented but needs debugging:**
1. ✅ **"Carving the mapping" (Section 3.3)** — Implemented in `carve_image_lattice_guided()`
   - Maps energy V→L, finds seam in L, shifts u-coordinates, single pixel resample
   - Has cumulative shift bug fix, but still produces distorted output
2. ✅ **Seam pairs (Section 3.6)** — Implemented in `carve_seam_pairs()`
   - Two windowed regions: ROI (shrink) and pair (expand)
   - Same cumulative shift issue, needs debugging
3. ✅ **Curved lattice support** — Added `Lattice2D.from_horizontal_curve()`
   - For rivers, roads, or features following a curve
   - Creates scanlines perpendicular to centerline curve

**What's not implemented yet:**
1. **ROI-bounded lattices** — Lattice should cover region of interest only, not entire image
2. **Cyclic lattices (Section 3.5)** — For closed shapes (rings, tubes)
3. **Forward energy (Rubinstein et al. 2008)** — Stub exists in `energy.py`
4. **Concentric circle lattice** — May or may not be needed (radial lattice might work)

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
