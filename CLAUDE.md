# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Current State (as of 2026-02-11)

**Core pipeline — all implemented and tested:**
- ✅ **Lattice construction**: `rectangular()`, `circular()`, `from_curve_points()`, `from_horizontal_curve()`
  - Arc length resampling, symmetric scanlines, cyclic support (Section 3.5)
  - Forward/inverse mapping fully vectorized
- ✅ **Lattice smoothing** (Section 3.4.2): `Lattice2D.smooth()`
  - Iterative mean filter on origins, overlap detection, convergence stopping
- ✅ **Energy functions**: gradient magnitude (Eq. 6), forward energy (Rubinstein 2008), normalization to [0,1]
- ✅ **Seam algorithms**: greedy, multi-greedy, windowed, cyclic (Section 4.0.1)
- ✅ **Carving the mapping** (Section 3.3): single-interpolation approach, cumulative shifts
- ✅ **Seam pairs** (Section 3.6): ROI shrink + pair expand, boundary preservation
- ✅ **ROI-bounded carving**: `roi_bounds` parameter, validity masking

**What still needs work:**
- ⚠️ **Carving correctness tests** — weak tests removed, real ones not yet written (Task 7)
- ⚠️ **Visual validation** — no reproduce_figures.py yet (Task 8)
- ⚠️ **Browser demo** — not started (Task 9)

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
  - `tests/test_carving.py` — traditional + lattice-guided + seam pairs (5 meaningful tests)
  - `tests/test_smoothing.py` — overlap detection, smoothing convergence (7 tests)
- Run with: `conda run -n lattice-carving python -m pytest tests/ -v`
- **55 tests total, all passing**
- Tests verify algorithmic correctness, not just shapes
- Use deterministic seeds (`torch.manual_seed`) for reproducibility

## Architecture

- `src/lattice.py` — Lattice2D class: construction, forward/inverse mapping, resampling, smoothing
  - `rectangular()`, `circular()`, `from_curve_points()`, `from_horizontal_curve()`
  - `smooth()` — Section 3.4.2 iterative mean filter
- `src/energy.py` — Energy functions
  - `gradient_magnitude_energy()` — Eq. 6, L1 gradient norm
  - `normalize_energy()` — remap to [0, 1] (paper page 10)
  - `forward_energy()` — Rubinstein et al. 2008, three-cost DP
- `src/seam.py` — Seam algorithms: greedy, windowed, cyclic, multi-greedy, removal
- `src/carving.py` — High-level carving orchestration
  - `carve_image_traditional()` — standard rectangular seam carving
  - `carve_image_lattice_guided()` — carving the mapping (Section 3.3), with ROI bounds
  - `carve_seam_pairs()` — local region resizing (Section 3.6)
- `pyproject.toml` — package config for `pip install -e .`

## Key Algorithmic Notes

- Greedy seam is sensitive to first-row energy (border artifacts from Sobel padding).
  Multi-greedy helps by trying multiple starting points.
- Circular lattice forward_mapping needs tangent-projection penalty to disambiguate
  between a radial scanline and its 180° opposite. Already implemented.
- The lattice forward/inverse mappings are fully vectorized (batched tensor ops).
- Energy normalization is applied in all carving pipelines per paper specification.
- Forward energy uses cumulative DP — values increase top-to-bottom.
