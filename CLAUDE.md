# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Current State (as of 2026-02-12)

**Core pipeline — all implemented, tested, and visually validated:**
- ✅ **Lattice construction**: `rectangular()`, `circular()`, `from_curve_points()`, `from_horizontal_curve()`
  - Arc length resampling, symmetric scanlines, cyclic support (Section 3.5)
  - Forward/inverse mapping fully vectorized
  - **Bug fix**: Cyclic forward_mapping tangent projection penalty disabled for cyclic lattices
  - **Bug fix**: Fractional n wraps around for cyclic lattices (modular arithmetic)
- ✅ **Lattice smoothing** (Section 3.4.2): `Lattice2D.smooth()`
  - Iterative mean filter on origins, overlap detection, convergence stopping
- ✅ **Energy functions**: gradient magnitude (Eq. 6), forward energy (Rubinstein 2008), normalization to [0,1]
  - Color images: RGB→grayscale via luma weights (0.299R + 0.587G + 0.114B), then Sobel
- ✅ **Seam algorithms**: greedy, multi-greedy, DP, windowed, cyclic (Section 4.0.1)
  - DP seam finding (Avidan & Shamir 2007) for globally optimal seams
  - `dp_seam`, `dp_seam_windowed`, `dp_seam_cyclic` in `src/seam.py`
- ✅ **Carving the mapping** (Section 3.3): iterative warping from current state
  - **Bug fix (2026-02-12)**: Replaced incorrect cumulative shift accumulation with
    iterative warping. Old approach composed shifts additively, causing off-by-one
    errors at seam boundaries. New approach matches the paper exactly — each iteration
    warps from V_c (copy of current state). See DEVLOG.md for full analysis.
  - **Bug fix**: `_interpolate_seam` now uses modular arithmetic for cyclic lattices
- ✅ **Seam pairs** (Section 3.6): ROI shrink + pair expand, boundary preservation
- ✅ **ROI-bounded carving**: `roi_bounds` parameter, validity masking
- ✅ **Visual validation**: `examples/reproduce_figures.py` generates 5 figure comparisons

**Known limitation — sawtooth artifacts at low resolution:**
- Sawtooth/scalloping on synthetic bagel demos is a resolution/aliasing artifact
- At 200px / 256 scanlines, the discrete lattice grid creates visible angular stepping
- Resolution sweep (DEVLOG.md) confirms: 600px / 1024 scanlines produces smooth results
- DP seams are correct and optimal — the artifacts are from lattice discretization, not seam quality

**What still needs work:**
- ⚠️ **SAM-based ROI definition** — use SAM2 for precise object boundary extraction
- ⚠️ **Browser demo** — not started

## Coding Preferences

- Python with PyTorch (GPU-accelerated tensor ops)
- Vectorized operations over Python loops wherever possible
- Conda environment: `lattice-carving`
- Run commands with: `conda run -n lattice-carving ...`

## Testing

- Test suite split across focused modules:
  - `tests/conftest.py` — shared fixtures (`make_gradient_image`, `make_ring_image`)
  - `tests/test_lattice.py` — construction, mapping, resampling (17 tests)
  - `tests/test_energy.py` — gradient magnitude, normalization, forward energy (14 tests)
  - `tests/test_seam.py` — greedy, windowed, multi-greedy, removal (12 tests)
  - `tests/test_carving.py` — traditional, lattice-guided, ROI bounds, seam pairs (18 tests)
  - `tests/test_smoothing.py` — overlap detection, smoothing convergence (7 tests)
- Run with: `conda run -n lattice-carving python -m pytest tests/ -v`
- **69 tests total, all passing**
- Tests verify observable behavior (identity, sanity, boundary preservation)
- Use deterministic seeds (`torch.manual_seed`) for reproducibility

## Architecture

- `src/lattice.py` — Lattice2D class: construction, forward/inverse mapping, resampling, smoothing
  - `rectangular()`, `circular()`, `from_curve_points()`, `from_horizontal_curve()`
  - `smooth()` — Section 3.4.2 iterative mean filter
- `src/energy.py` — Energy functions
  - `gradient_magnitude_energy()` — Eq. 6, L1 gradient norm
  - `normalize_energy()` — remap to [0, 1] (paper page 10)
  - `forward_energy()` — Rubinstein et al. 2008, three-cost DP
- `src/seam.py` — Seam algorithms: greedy, DP, windowed, cyclic, multi-greedy, removal
  - `dp_seam`, `dp_seam_windowed`, `dp_seam_cyclic` for optimal seam finding
  - All seam functions support `n_candidates` for multi-greedy (Section 4.0.1)
- `src/carving.py` — High-level carving orchestration
  - `carve_image_traditional()` — standard rectangular seam carving
  - `carve_image_lattice_guided()` — iterative warping (Section 3.3), with ROI bounds
  - `carve_seam_pairs()` — local region resizing (Section 3.6), iterative warping
  - `_carve_image_lattice_naive()` — naive double-interpolation (kept for comparison)
- `examples/reproduce_figures.py` — Visual validation: traditional, arch, no-blur, synthetic bagel, real bagel
- `pyproject.toml` — package config for `pip install -e .`

## Key Algorithmic Notes

- **Seam quality hierarchy** (worst → best):
  1. **Greedy** — O(n), picks locally optimal neighbor at each step. Wanders in flat energy.
  2. **Multi-greedy** — O(n * k), tries k starting points, keeps best. Helps with structured energy, NOT with flat regions.
  3. **DP** (dynamic programming) — O(n * W), finds globally optimal seam. For 2D images, equivalent to graph-cut.
  4. **Graph-cut** — paper's primary method. For 2D images, identical to DP. Needed for 3D volumes.
- Circular lattice forward_mapping needs tangent-projection penalty to disambiguate
  between a radial scanline and its 180° opposite — BUT this penalty must be
  **disabled for cyclic lattices** (from_curve_points with cyclic=True) because
  pixels inside a closed curve have negative tangent projection to ALL scanlines.
- The lattice forward/inverse mappings are fully vectorized (batched tensor ops).
- **Iterative warping** (Section 3.3): Each seam iteration warps from the current
  image state (V_c → V*), not from the original. This correctly handles the
  composition of multiple g* mappings, which is NOT simply additive. The old
  cumulative-shift-from-original approach had off-by-one errors at seam boundaries.
- Energy normalization is applied in all carving pipelines per paper specification.
- Forward energy uses cumulative DP — values increase top-to-bottom.
- Color images: energy computed on grayscale (luma-weighted), carving applied to all channels.
