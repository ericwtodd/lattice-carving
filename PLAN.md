# Plan: Faithful Generalized Fluid Carving Implementation

## Context

We're implementing "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation" (Flynn et al., 2021) in Python/JS. The Python implementation exists but progress has been slow because tasks weren't small enough and tests don't validate correctness well. The goal is: (1) a faithful Python implementation of the paper's pipeline, (2) solid tests for each component, (3) an interactive browser demo.

**Current state (updated 2026-02-11 evening)**: Phases 0–3 are COMPLETE. All paper algorithms are implemented and tested (68 tests). Visual validation via `reproduce_figures.py` confirms: traditional carving works, no-blur approach preserves sharpness, cyclic lattice mapping is correct. **Main open issue**: sawtooth artifacts on bagel seam pairs due to greedy seam wandering in flat-energy regions — DP seam finding is the fix. Browser demo (Phase 4) is not started.

---

## Repo Reorganization

```
lattice-carving/
  src/
    __init__.py          # Already uses relative imports ✓
    lattice.py           # Lattice2D class (existing, needs smooth() method)
    energy.py            # Energy functions (needs forward energy + normalization)
    seam.py              # Seam algorithms (existing)
    carving.py           # Carving orchestration (fix imports, add ROI bounds)
  tests/
    conftest.py          # NEW: shared fixtures (test images, lattices)
    test_lattice.py      # Keep: lattice construction + mapping tests
    test_energy.py       # NEW: energy function tests
    test_seam.py         # NEW: seam algorithm tests (extract from test_lattice.py)
    test_carving.py      # NEW: carving correctness tests (the critical gap)
    test_smoothing.py    # NEW: lattice smoothing tests
  examples/
    create_test_images.py     # Existing: generate synthetic test images
    reproduce_figures.py      # NEW: reproduce paper Figures 3, 6, 10, 22
    generate_visualization.py # NEW: generate demo data for browser
  web/                   # NEW: browser demo (static site)
    index.html
    app.js
    style.css
  demo_data/             # NEW: pre-computed JSON for browser demo
```

**Key change**: Fix `src/carving.py` imports from bare (`from lattice import`) to relative (`from .lattice import`) so the package works consistently. All example scripts use `sys.path.insert(0, src/)` pattern.

---

## Task Breakdown

### Phase 0: Infrastructure

**Task 0.1: Fix imports and split tests** ✅ DONE (imports + pyproject.toml)
- Fix `src/carving.py` to use relative imports (match `__init__.py`)
- Create `tests/conftest.py` with shared fixtures (`make_gradient_image`, `make_ring_image`, standard lattices)
- Extract seam tests → `tests/test_seam.py`, energy tests → `tests/test_energy.py`
- Add `pyproject.toml` so `pip install -e .` works
- **Test**: All existing tests pass via `python -m pytest tests/ -v`
- **Files**: `src/carving.py`, `tests/conftest.py`, `tests/test_seam.py`, `tests/test_energy.py`, `pyproject.toml`

### Phase 1: Algorithmic Faithfulness (Paper Pipeline)

**Task 1.1: Lattice smoothing (Section 3.4.2, Figure 9)** ✅ DONE
- `Lattice2D.smooth(max_iterations=50)` — iterative mean filter on origins
- Overlap detection, convergence stopping, shape preservation
- 7 tests in `tests/test_smoothing.py`

**Task 1.2: Energy normalization (paper page 10)** ✅ DONE
- `normalize_energy()` in `energy.py`, applied in all carving pipelines
- 4 tests in `tests/test_energy.py`

**Task 1.3: ROI-bounded carving** ✅ DONE
- `roi_bounds` parameter + validity masking in `carve_image_lattice_guided()`
- 3 tests in `tests/test_carving.py`

**Task 1.4: Forward energy (Rubinstein 2008)** ✅ DONE
- Three-cost DP (left/center/right transitions) in `energy.py`
- 4 tests in `tests/test_energy.py`

**Task 1.5: Validate seam pair mechanics (Section 3.6, Figure 10)** ✅ DONE
- 5 tests verifying boundary preservation, ROI changes, dimension preservation, shift cancellation
- Tests in `tests/test_carving.py`

### Phase 2: Visual Validation (Reproduce Paper Figures) ✅ DONE

**Task 2.1: Reproduce Figure 3 — traditional vs. lattice-guided** ✅ DONE
- `figure_arch_carving()` in `reproduce_figures.py`
- Shows arch preservation with lattice-guided vs. traditional distortion
- Output: `output/fig_arch_carving.png`

**Task 2.2: Reproduce Figure 6 — naive vs. correct (no-blur)** ✅ DONE
- `figure_no_blur_comparison()` in `reproduce_figures.py`
- Checkerboard with curved lattice, correct approach preserves sharpness
- Output: `output/fig_no_blur_comparison.png`

**Task 2.3: Reproduce Figure 10/22 — seam pairs** ✅ DONE (with known artifact issue)
- `figure_synthetic_bagel_seam_pairs()` and `figure_real_bagel_seam_pairs()`
- Synthetic bagel: shrink/grow ring body with cyclic lattice
- Real double-bagel: shrink/grow left bagel half
- **Finding**: Sawtooth artifacts from greedy seam wandering in flat energy. Multi-greedy (n_candidates=16) does not fix. DP seam finding needed.
- Output: `output/fig_synthetic_bagel_pairs.png`, `output/fig_real_bagel_*.png`

### Phase 3: Correctness Tests ✅ DONE

**Task 3.1: Write carving correctness tests** ✅ DONE
- 18 tests in `tests/test_carving.py`
- Covers: width reduction, seam avoidance, rectangular equivalence, no-blur vs naive,
  monotonicity, ROI bounds (exact equality outside, modification inside), seam pairs
  (boundary preservation, ROI changes, dimension preservation, shift cancellation)

**Task 3.2: Write smoothing and energy tests** ✅ DONE
- 7 smoothing tests, 14 energy tests
- All in `tests/test_smoothing.py` and `tests/test_energy.py`

### Phase 4: Browser Demo

**Task 4.1: Generate demo data (Python)**
- Script runs full carving pipeline, captures ALL intermediate state per iteration
- Output: JSON files with base64 images, seam coordinates, lattice geometry
- Configurations: bagel (seam pairs), arch (lattice-guided), river (comparison)
- **Files**: `examples/generate_visualization.py`, `demo_data/*.json`

**Task 4.2: Build browser demo**
- Static HTML/CSS/JS that loads pre-computed JSON, no server needed
- Layout: original + seam overlay | carved result, with seam count slider
- Controls: image selector, seam count slider, mode toggle (standard vs pairs)
- Toggleable lattice detail panel: unrolled lattice image, energy heatmap, seam overlay
- Canvas overlay for seam/lattice drawing on original image
- **Files**: `web/index.html`, `web/app.js`, `web/style.css`

**Task 4.3: Interactive lattice drawing**
- User clicks points to define a curve on the image
- Shows resulting lattice grid overlay in real-time
- Matches nearest pre-computed configuration to show carving result
- **Files**: `web/app.js`

### Phase 5: SAM Integration (Future)

**Task 5.1: SAM-based ROI definition**
- Use SAM to segment image regions
- User clicks a region → SAM generates mask → extract boundary → fit lattice
- Automatically determine ROI and pair windows from segmentation
- **Files**: New `src/sam_integration.py`

---

## Dependency Graph

```
Phase 0: Task 0.1 (fix imports/split tests) ✅
    │
    ├── Phase 1 (algorithmic faithfulness) ✅ ALL DONE
    │   ├── Task 1.1 (lattice smoothing) ✅
    │   ├── Task 1.2 (energy normalization) ✅
    │   ├── Task 1.3 (ROI-bounded carving) ✅
    │   ├── Task 1.4 (forward energy) ✅
    │   └── Task 1.5 (seam pair validation) ✅
    │
    ├── Phase 2 (visual validation) ✅ ALL DONE
    │   ├── Task 2.1 (Figure 3: arch) ✅
    │   ├── Task 2.2 (Figure 6: no-blur) ✅
    │   └── Task 2.3 (Figure 10/22: seam pairs) ✅ (sawtooth artifacts noted)
    │
    ├── Phase 3 (correctness tests) ✅ ALL DONE
    │   ├── Task 3.1 (carving tests: 18 tests) ✅
    │   └── Task 3.2 (smoothing/energy tests) ✅
    │
    ├── Phase 3.5 (quality improvement) ← CURRENT
    │   ├── Multi-greedy (n_candidates) ✅ Implemented, doesn't fix flat-energy wandering
    │   └── DP seam finding ❌ TODO — needed for artifact-free bagel results
    │
    └── Phase 4 (browser demo) ❌ NOT STARTED
        ├── Task 4.1 (generate data) → Task 4.2 (browser demo) → Task 4.3 (interactive)
```

---

## Paper Equation Verification Checklist

| Paper Reference | What | Status | Our Code |
|----------------|------|--------|----------|
| Eq. 6: E_I = \|∂I/∂x\| + \|∂I/∂y\| | L1 gradient magnitude | ✅ Done | `energy.py:gradient_magnitude_energy()` |
| Eq. 4-5: p_w* = g*(f(p_w)) | Carve the mapping | ✅ Done | `carving.py:carve_image_lattice_guided()` |
| Sec 3.3: Sample original once | No double interpolation | ✅ Done + validated | `carving.py` (fig_no_blur_comparison.png) |
| Sec 3.4.2: Mean filter smoothing | Lattice smoothing | ✅ Done | `lattice.py:smooth()` |
| Sec 3.5: Cyclic lattice | Connect last→first plane | ✅ Done + bug fixed | `lattice.py` (tangent penalty + frac wrap) |
| Sec 3.6: Seam pairs (+1/-1) | Local region resizing | ✅ Done + tested | `carving.py:carve_seam_pairs()` |
| Sec 4.0.1: Gaussian guide | Cyclic seam closure | ✅ Done | `seam.py:greedy_seam_cyclic()` |
| Sec 4.0.1: Multi-greedy | Multiple starting points | ✅ Done | `seam.py` (n_candidates param) |
| Sec 4.0.1: Graph-cut / DP | Optimal seam finding | ❌ TODO | Needed for flat-energy regions |
| Fig 9: Lattice from points | Arc-length resampling | ✅ Done | `lattice.py:from_curve_points()` |
| Page 10: Energy in [0,1] | Normalize energy | ✅ Done | `energy.py:normalize_energy()` |
| Forward energy (Rubinstein 2008) | Edge-aware seam cost | ✅ Done | `energy.py:forward_energy()` |
| ROI-bounded carving | Validity mask | ✅ Done | `carving.py:roi_bounds` |
| Figs 3,6,10,22: Visual results | Paper figure reproduction | ✅ Done | `examples/reproduce_figures.py` |

---

## Verification Strategy

1. **Unit tests**: `python -m pytest tests/ -v` — every component has targeted correctness tests
2. **Figure reproduction**: `python examples/reproduce_figures.py` — visual comparison with paper
3. **Browser demo**: Open `web/index.html` — interactive scrubbing through carving iterations
4. **Regression**: All tests run in CI-like fashion before each commit
