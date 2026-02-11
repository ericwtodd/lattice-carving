# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Current State (as of 2026-02-11 - Evening)

**What works (lattice construction is SOLID):**
- ✅ **Lattice2D.from_curve_points()** - Build lattice from user point list (Figure 9)
  - Arc length resampling for uniform spacing
  - Symmetric scanline coverage (±perp_extent from centerline)
  - Origins aligned with user-specified curve points
  - Forward/inverse mapping with u_offset for centering
- ✅ `Lattice2D.rectangular()` and `Lattice2D.circular()` constructors
- ✅ Forward mapping (world V → lattice L) and inverse mapping (L → V), vectorized
- ✅ Gradient magnitude energy (Sobel filters)
- ✅ Greedy and multi-greedy seam algorithms
- ✅ Seam removal and windowed seam finding

**Recent critical fixes:**
1. ✅ **Iterative warping bug** - Was sampling warped image repeatedly (compounds blur)
   - Fix: Sample from ORIGINAL image once at end using cumulative shifts
2. ✅ **Cumulative shift bug** - Shifts computed against fixed u_map
   - Fix: Track cumulative_shift across iterations
3. ✅ **Asymmetric scanlines** - Only extended one side of curve
   - Fix: Apply u_offset in both forward AND inverse mapping
4. ✅ **Origins not aligned** - Were offset from user points
   - Fix: Keep origins on centerline, use u_offset for symmetric coverage

**Current status:**
- Lattice construction: **SOLID** ✓
- Carving algorithms: **FIXED BUT UNTESTED** ⚠️
- Need validation on real test cases with proper lattices

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
  - `rectangular()` - straight horizontal scanlines
  - `circular()` - radial scanlines from center
  - `from_horizontal_curve()` - curved scanlines following a path
- `src/energy.py` — Energy functions (gradient magnitude, forward energy)
- `src/seam.py` — Greedy seam computation, seam removal
- `src/carving.py` — High-level carving orchestration
  - `carve_image_lattice_guided()` - carving the mapping approach
  - `carve_seam_pairs()` - local region resizing
- `examples/debug_lattice_carving.py` — Debug visualization (bagel, river, arch)
- `tests/` — pytest suite (30 tests, but only check shapes not correctness)

## Key Algorithmic Notes

- Greedy seam is sensitive to first-row energy (border artifacts from Sobel padding).
  Multi-greedy helps by trying multiple starting points.
- Circular lattice forward_mapping needs tangent-projection penalty to disambiguate
  between a radial scanline and its 180° opposite. Already implemented.
- The lattice forward/inverse mappings are fully vectorized (batched tensor ops),
  replacing earlier O(N*n_lines) Python loops.

## Test Cases for Validation

Four test cases, each with centerline defined as points:

1. **Sine Wave**: Simple sinusoidal curve for basic validation
   - Centerline: y = 200 + 40*sin(2πx/400)
   - Tests symmetric coverage, arc length resampling

2. **Arch (Figure 3)**: Semicircular arch
   - Centerline: semicircle from paper's Figure 3
   - Traditional should squish, lattice-guided should preserve

3. **River**: Horizontally-flowing sinusoidal river
   - Centerline: y = 256 + 50*sin(3πx/512)
   - Scanlines perpendicular to flow

4. **Bagel (seam pairs)**: Circular centerline around hole
   - Centerline: circle at middle radius between hole and edge
   - Test seam pairs: shrink hole, expand background

**Current test:** `conda run -n lattice-carving python examples/test_lattice_visualization.py`
- Validates lattice construction only (no carving yet)
- Generates: lattice_sine.png, lattice_arch.png, lattice_river.png, lattice_bagel.png
