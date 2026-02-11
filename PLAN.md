# Plan: Faithful Generalized Fluid Carving Implementation

## Context

We're implementing "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation" (Flynn et al., 2021) in Python/JS. The Python implementation exists but progress has been slow because tasks weren't small enough and tests don't validate correctness well. The goal is: (1) a faithful Python implementation of the paper's pipeline, (2) solid tests for each component, (3) an interactive browser demo.

**Current state**: Lattice construction is solid (vectorized mappings, arc-length resampling, cyclic support). Carving algorithms exist but are **untested on real images** — no one has verified the output looks correct. ~40% of tests only check tensor shapes. Missing: lattice smoothing (Section 3.4.2), energy normalization, ROI-bounded carving, forward energy.

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

**Task 1.1: Lattice smoothing (Section 3.4.2, Figure 9)** — HIGH PRIORITY
- Add `Lattice2D.smooth(max_iterations=50)` method
- Algorithm: iterative mean filter on origins in the u/v directions (NOT w/normal direction), aligning each scanline with its neighbors
- Detect overlapping planes: check if adjacent scanlines' perpendicular ranges intersect in world space
- Stop when no overlaps or max iterations reached
- **Test**: Jagged zigzag curve → after smoothing, no overlapping scanlines. Smoothed curve still approximates original (max deviation bounded).
- **Files**: `src/lattice.py`, `tests/test_smoothing.py`

**Task 1.2: Energy normalization (paper page 10)** — HIGH PRIORITY
- Paper: "we also remap our energy functions to always be between 0 and 1"
- Add normalization step: `energy = (energy - min) / (max - min + eps)`
- Apply consistently in carving pipeline (in `carve_image_lattice_guided` and `carve_seam_pairs`)
- **Test**: After normalization, min≈0.0 and max≈1.0. Seam positions unchanged (monotonic transform). All existing tests pass.
- **Files**: `src/energy.py`, `src/carving.py`

**Task 1.3: ROI-bounded carving** — HIGH PRIORITY
- Currently warp applies to entire image. Paper figures show lattice covering only ROI.
- Add validity mask: only pixels whose forward mapping falls within lattice bounds get warped
- Pixels outside lattice region stay unchanged: `result = where(valid, warped, original)`
- **Test**: Create lattice covering center of image. After carving, border pixels identical to original. Interior pixels modified.
- **Files**: `src/carving.py`

**Task 1.4: Forward energy (Rubinstein 2008)** — MEDIUM PRIORITY
- Replace stub with actual implementation
- Forward energy considers cost of new edges introduced by seam removal
- Three transition costs (left/center/right) based on neighboring pixel differences
- **Test**: Uniform image → zero energy. Vertical edge → correct penalty for edge-crossing seams.
- **Files**: `src/energy.py`, `tests/test_energy.py`

**Task 1.5: Validate seam pair mechanics (Section 3.6, Figure 10)**
- Verify the +1/-1 shift logic is correct: ROI shrinks, pair expands
- Verify net displacement is zero outside both windows
- Verify pair window is in "positive direction" of ROI
- **Test**: After seam pairs on simple striped image, measure actual pixel displacement at boundaries.
- **Files**: `tests/test_carving.py`

### Phase 2: Visual Validation (Reproduce Paper Figures)

**Task 2.1: Reproduce Figure 3 — traditional vs. lattice-guided**
- Arch image: traditional carving distorts silhouette, lattice-guided preserves it
- Shows the core value proposition of the method
- **Test**: After lattice-guided carving, arch remains approximately semicircular (measure radial variance).
- **Files**: `examples/reproduce_figures.py`

**Task 2.2: Reproduce Figure 6 — naive vs. correct (no-blur)**
- Show naive double-interpolation causes blur; carving-the-mapping doesn't
- Use a detailed image with a curved lattice, carve 10+ seams
- **Test**: Correct approach has higher Laplacian variance (sharpness) than naive approach.
- **Files**: `examples/reproduce_figures.py`

**Task 2.3: Reproduce Figure 10/22 — seam pairs**
- Show seam pairs on bagel: hole shrinks, boundary unchanged
- This is the primary end-to-end validation
- **Test**: After seam pairs, image dimensions unchanged, boundary pixels identical, ROI visibly different.
- **Files**: `examples/reproduce_figures.py`

### Phase 3: Correctness Tests

**Task 3.1: Write carving correctness tests**
- **Circle preservation**: Ring image + circular lattice → ring stays circular after N seams
- **No-blur**: Laplacian variance of correct approach ≥ naive approach
- **Boundary fixedness**: Pixels far from both windows unchanged after seam pairs
- **Content shift direction**: After removing seam at column K, pixels at K+ shift left by 1
- **Cumulative shift values**: After N seams, shifts are integers in [0, N]
- **Files**: `tests/test_carving.py`

**Task 3.2: Write smoothing and energy tests**
- Smoothing convergence, overlap detection, shape preservation
- Forward energy penalizes edge-crossing more than gradient magnitude
- Energy normalization to [0, 1]
- **Files**: `tests/test_smoothing.py`, `tests/test_energy.py`

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
    ├── Phase 1 (all tasks can run in parallel):
    │   ├── Task 1.1 (lattice smoothing)
    │   ├── Task 1.2 (energy normalization)
    │   ├── Task 1.3 (ROI-bounded carving)
    │   ├── Task 1.4 (forward energy)
    │   └── Task 1.5 (seam pair validation)
    │
    ├── Phase 2 (after Phase 1):
    │   ├── Task 2.1 (Figure 3)
    │   ├── Task 2.2 (Figure 6)
    │   └── Task 2.3 (Figure 10/22)
    │
    ├── Phase 3 (after Phase 1):
    │   ├── Task 3.1 (carving tests)
    │   └── Task 3.2 (smoothing/energy tests)
    │
    └── Phase 4 (after Phase 2):
        ├── Task 4.1 (generate data) → Task 4.2 (browser demo) → Task 4.3 (interactive)

Phase 4.2 (HTML/CSS layout) can start in parallel with everything.
```

---

## Paper Equation Verification Checklist

| Paper Reference | What | Status | Our Code |
|----------------|------|--------|----------|
| Eq. 6: E_I = \|∂I/∂x\| + \|∂I/∂y\| | L1 gradient magnitude | ✅ Done | `energy.py:14` |
| Eq. 4-5: p_w* = g*(f(p_w)) | Carve the mapping | ✅ Done | `carving.py` |
| Sec 3.3: Sample original once | No double interpolation | ✅ Done | `carving.py:255` |
| Sec 3.4.2: Mean filter smoothing | Lattice smoothing | ✅ Done | `lattice.py:smooth()` |
| Sec 3.5: Cyclic lattice | Connect last→first plane | ✅ Done | `lattice.py` |
| Sec 3.6: Seam pairs (+1/-1) | Local region resizing | ✅ Done | `carving.py:263` |
| Sec 4.0.1: Gaussian guide | Cyclic seam closure | ✅ Done | `seam.py:82` |
| Fig 9: Lattice from points | Arc-length resampling | ✅ Done | `lattice.py:83` |
| Page 10: Energy in [0,1] | Normalize energy | ✅ Done | `energy.py:normalize_energy()` |
| Forward energy (Rubinstein 2008) | Edge-aware seam cost | ✅ Done | `energy.py:forward_energy()` |
| ROI-bounded carving | Validity mask | ✅ Done | `carving.py:roi_bounds` |
| Figs 3,6,10,22: Visual results | Paper figure reproduction | ❌ TODO | Phase 2 |

---

## Verification Strategy

1. **Unit tests**: `python -m pytest tests/ -v` — every component has targeted correctness tests
2. **Figure reproduction**: `python examples/reproduce_figures.py` — visual comparison with paper
3. **Browser demo**: Open `web/index.html` — interactive scrubbing through carving iterations
4. **Regression**: All tests run in CI-like fashion before each commit
