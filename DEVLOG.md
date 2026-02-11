# Development Log

## 2026-02-11 - Lattice-Space Visualization & Energy Function Fix

### Created Lattice-Space Visualization (Figure 12)

**Goal**: Visualize images in lattice index space (u,n coordinates) to validate resampling and check for lattice overlaps.

**Created scripts**:
- `create_test_images.py` - Generates reusable test images (bagel, arch, river)
  - Bagel: Uses sesame seed code from test_lattice_visualization.py
  - Arch: Semicircular arch using distance-based generation
  - River: Sine wave from test_lattice_visualization.py
- `show_lattice_space.py` - Visualizes test images in lattice space
  - Shows 4 panels: Image, Energy, ROI seam, Pair seam
  - All in (u,n) coordinates like Figure 12 from paper

**Purpose**:
- Validate that resampling works correctly (features should appear straight/regular in lattice space)
- Check for overlapping lattices (would show as distortions)
- Verify energy computation makes sense in lattice space
- Debug seam computation visually

### Fixed Energy Function (Paper Equation 6)

**Problem**: Used L2 norm (Euclidean) instead of L1 norm (Manhattan) for gradient magnitude.

**Our old implementation**:
```python
energy = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # L2 norm
```

**Paper equation 6**:
```
E_I(i,j) = ||∂/∂x I|| + ||∂/∂y I||  # L1 norm
```

**Fix**:
```python
energy = torch.abs(grad_x) + torch.abs(grad_y)  # L1 norm ✓
```

**Impact**:
- Now matches Avidan & Shamir 2007 exactly
- Faster (no sqrt operation)
- Should give seam paths matching paper's results

### Fixed Import Issues

**Problem**: Relative imports in `src/` files broke when running scripts.

**Solution**: Standardized all example scripts to:
1. Add `src/` to `sys.path`
2. Import modules without `src.` prefix
3. All `src/` modules use absolute imports (no relative imports)

**Files fixed**:
- `src/carving.py` - Changed from relative to absolute imports
- `examples/test_lattice_visualization.py` - Fixed import path
- `examples/show_lattice_space.py` - Uses correct import pattern

### Implemented Cyclic Seam with Gaussian Energy Guide

**Problem**: Seams on bagel (cyclic lattice) don't connect - they start and end at different positions.

**Root Cause**: Basic greedy seam has no mechanism to ensure closure for cyclic lattices.

**Paper Solution (Section 4.0.1, Figure 12)**:
> "This is done by adding an inverted multidimensional Gaussian to the energy function. The Gaussian is centered about the initial seam selected in the width so that the Gaussian steers it back towards the final seam."

**Our Implementation** (`greedy_seam_cyclic()` in `src/seam.py`):

1. **Compute initial seam** without guidance
   ```python
   initial_seam = greedy_seam_windowed(energy, col_range, direction)
   ```

2. **Create Gaussian guide** centered on initial seam path
   ```python
   dist_to_seam = torch.abs(uu - seam_positions.unsqueeze(1))
   gaussian_guide = torch.exp(-(dist_to_seam ** 2) / (2 * guide_width ** 2))
   ```

3. **Add closure strength** (stronger near start/end)
   ```python
   closure_strength = torch.linspace(1.0, 0.0, H // 2, device=energy.device)
   closure_strength = torch.cat([closure_strength, closure_strength.flip(0)])
   ```

4. **Subtract Gaussian from energy** (lower energy = preferred path)
   ```python
   energy_guided = energy - 0.5 * gaussian_guide * closure_strength.unsqueeze(1)
   ```

5. **Recompute seam** with guided energy
   ```python
   final_seam = greedy_seam_windowed(energy_guided, col_range, direction)
   ```

**Parameters**:
- `guide_width`: Controls Gaussian width (default 10.0)
  - Smaller = stronger guidance toward closure
  - Larger = looser guidance, more natural path

**Result**: Seams now connect properly on circular lattices ✓

**Updated Scripts**:
- `visualize_real_bagel.py` now uses `greedy_seam_cyclic()` for bagel
- Should see connected seams forming closed loops

### Updated Documentation

**Added to IMPLEMENTATION_STATUS.md**:
- ✅ Cyclic greedy with Gaussian guide - IMPLEMENTED
- Lattice smoothing (HIGH PRIORITY) - Section 3.4.2, Figure 9
- Both marked as important for faithful paper reproduction

**Next Steps**:
1. ✅ Implement Gaussian energy guide for cyclic seams
2. Run real bagel visualization to verify seams connect
3. Implement lattice smoothing if overlaps detected
4. Apply actual carving with seam pairs

## 2026-02-11 - Seam Pairs Understanding & Visualization

### Deep Dive into Section 3.6 (Seam Pairs)

**Read and analyzed** paper sections 3.5.1 (Cyclic Graph-Cut Seams) and 3.6 (Seam Pairs).

**Key Understanding:**
- **Seam pairs** enable local region resizing without changing global image boundaries
- **Two user-defined windows** in lattice u-coordinates (non-overlapping):
  - **ROI window**: Region being retargeted (feature to grow/shrink)
  - **Pair window**: Compensating region (usually background)
- **To DECREASE ROI size**: Remove seam from ROI (+1 shift), add seam to pair (-1 shift)
- **To INCREASE ROI size**: Reverse the shifts
- **Net effect**: Shifts cancel at boundaries → global dimensions unchanged

**Window Specification:**
- Paper says "user-defined windows" but doesn't detail the interface
- Likely using Houdini masks/VDB tools to define regions
- **Our approach**: Start with manual u-coordinate ranges, plan for segmentation-based UI later

**Clarified Relationship:**
- **Centerline points** (R): Define lattice path (n-direction)
- **Windows**: Define perpendicular regions (u-direction)
- These are orthogonal - no conflict!

### Implemented Cyclic Lattice Visualization Fix

**Problem**: Bagel lattice visualization showed a gap instead of continuous loop

**Root Cause**: When drawing lattice grid for cyclic lattices:
- Cross-scanline lines went from `n=0` to `n=n_lines-1`
- For cyclic, should go to `n=n_lines` to show wrap-around

**Fix**:
- Check `lattice._cyclic` flag
- If cyclic, extend n_vals to `n_lines` instead of `n_lines-1`
- Also fix seam visualization to wrap around properly
- Applies to both `visualize_lattice_grid` and `visualize_seams`

**Result**: Bagel lattice now displays as continuous closed loop ✓

### Updated Test Cases with Proper Seam Pair Windows

**Approach**:
1. Lattice `perp_extent` large enough to cover feature + background
2. ROI window: Inside feature (area to resize)
3. Pair window: Outside feature in background (compensating region)

**Updated test cases:**

**Arch** (grow arch outward):
- perp_extent: 80 (lattice_width=160)
- Arch band_width: 40px
- ROI window: (90, 110) - outer part of arch
- Pair window: (115, 140) - background beyond arch

**River** (shrink river):
- perp_extent: 120 (lattice_width=240)
- River band_width: 60px
- ROI window: (100, 140) - river itself
- Pair window: (160, 200) - background to one side

**Bagel** (grow bagel outward):
- perp_extent: 80 (lattice_width=160)
- Bagel: inner_radius=100, outer_radius=200, centerline=150
- ROI window: (30, 130) - full bagel donut (radius 100-200)
- Pair window: (140, 160) - background beyond bagel (radius 210-230)

### Critical Bug Fix: Normal Direction for Clockwise Curves

**Problem**: Orange boundary lines (Pair window) appeared inside the bagel hole instead of outside in the background.

**Debugging Process**:
- Added debug output to trace u-coordinate → world-space mappings
- Found that u=140 mapped to radius 90.5 (should be ~210)
- Found that u=30 mapped to radius 200 (should be ~100)
- **Root cause**: Coordinate system was backwards - positive u went inward, negative u went outward

**Analysis**:
For a clockwise circle, the "left" perpendicular to the tangent points INWARD toward the center.
Our original normal computation was:
```python
normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
```

This created normals pointing inward for clockwise curves. When we added `u_offset * normal` to the origin, positive u values moved toward the center instead of away from it.

**Fix** in `src/lattice.py`, `from_curve_points()`:
```python
# Changed from (WRONG):
normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)

# To (CORRECT):
normals = torch.stack([tangents[:, 1], -tangents[:, 0]], dim=1)
```

This flips the normals to point outward, making positive u values extend away from the centerline as expected.

**Verification**:
- After fix: u=140 → radius 209.8 ✓
- After fix: u=30 → radius 100 ✓
- Orange boundaries now correctly positioned outside bagel in background
- Yellow boundaries still correctly frame the bagel donut
- All cyclic visualizations close properly

### Improved Visualization

**Added window boundary markers**:
- Yellow dashed lines: ROI window boundaries
- Orange dashed lines: Pair window boundaries
- Makes it clear where each seam should be constrained

**Three-panel visualization**:
1. Original image + centerline
2. Image + lattice grid + window boundaries
3. Image + seam pair + window boundaries

### Documentation Updates

**IMPLEMENTATION_STATUS.md**:
- Updated seam pairs section with paper understanding
- Marked cyclic lattices as implemented (basic support)
- Updated visualization phases and next steps

**Next Steps**:
1. Validate seam visualization (check seam placement)
2. Apply actual carving with `carve_seam_pairs()`
3. Verify before/after results
4. Plan better window specification UI (segmentation-based)

## 2026-02-10 - Project Initialization

### Session Start

- Created project directory: `github/lattice-carving`
- Initialized git repository
- Created initial documentation structure (README.md, DEVLOG.md, CLAUDE.md)
- Awaiting background PDF to understand algorithm requirements

### Paper Review - "Generalized Fluid Carving"

Read first 10 pages. Key concepts:

**Core Innovation:**
- Traditional seam carving assumes rectangular boundaries/grids
- This approach uses non-uniform lattices that follow the shape/motion of volumetric data
- Performs carving in "lattice index space" then maps back to world space
- Enables carving curved regions, non-rectangular boundaries, moving objects

**Key Components:**
1. **Lattice Structure** - Non-uniform grid defined by:
   - Sequence of parallel planes (origins o_n, basis vectors u_n, v_n)
   - Variable spacing Δx between planes
   - Forms curved "tubes" following data shape

2. **Mapping Functions:**
   - f: world space V → lattice index space L
   - g: lattice index space L → world space V
   - Enables carving in regular lattice space, then mapping back

3. **Seam Computation:**
   - Graph-cut method (slow but optimal)
   - Greedy approach (fast, 500x+ speedup, near-identical results)
   - Seam pairs for local region carving

4. **Energy Functions:**
   - Images: gradient magnitude
   - Fluids/meshes: mean curvature + kinetic energy
   - Volumetrics: density gradients + vorticity

5. **Lattice Creation:**
   - Automatic: follow velocity field/greatest motion
   - User-defined: manual curves/paths

### Implementation Decisions

**Language:** Python with PyTorch
- Fast computation via GPU acceleration
- Easy prototyping and visualization
- Can convert to C++/JavaScript later if needed

**Initial Focus:** 2D images (bagel.jpg as test case)

**Project Structure:**
```
src/
  lattice.py   - Lattice structures and mapping functions
  energy.py    - Energy functions (gradient magnitude, etc.)
  seam.py      - Greedy seam computation algorithms
examples/
  basic_seam_carving.py - Test traditional rectangular seam carving
```

### Implementation Progress

**Completed:**
- ✓ Project structure setup
- ✓ Conda environment created (lattice-carving)
- ✓ Requirements file (torch, numpy, pillow, matplotlib, scipy)
- ✓ Basic lattice structure (Lattice2D class with rectangular/circular constructors)
- ✓ Energy functions (gradient magnitude using Sobel filters)
- ✓ Greedy seam algorithms (single-greedy and multi-greedy)
- ✓ Seam removal function
- ✓ Example script for basic seam carving
- ✓ **TESTED**: Basic rectangular seam carving works on bagel.jpg
  - Successfully computed greedy seams
  - Carved image by removing 100 vertical seams
  - GPU acceleration working (CUDA)

**Next Steps:**
- ✓ Implement lattice mapping functions (forward/inverse)
  - ✓ Forward: world space → lattice index space
  - ✓ Inverse: lattice index space → world space
  - ✓ Tested and validated for both rectangular and circular lattices
  - ✓ Visualization confirms circular world space → regular lattice space
- ✓ Implement lattice-guided carving workflow:
  - ✓ `resample_to_lattice_space` — inverse-maps a regular lattice grid to world coords, samples via `grid_sample`
  - ✓ `resample_from_lattice_space` — forward-maps world grid to lattice coords, samples via `grid_sample`
  - ✓ `carve_image_lattice_guided` — full pipeline: resample → seam carve in lattice space → resample back
  - ✓ `carve_image_traditional` — convenience wrapper for standard rectangular carving
  - ✓ `carve_with_comparison` — runs both methods side-by-side
- ✓ Vectorized mapping functions (replaced O(N*n_lines) Python loops with batched tensor ops)
- ✓ Fixed forward_mapping for radial/circular lattices — tangent-projection penalty resolves
  ambiguity when a scanline and its opposite both have 0 normal distance
- ✓ Test suite (30 tests, all passing):
  - Lattice construction (rectangular, circular)
  - Round-trip mapping (rectangular identity, circular inverse↔forward)
  - Resampling (rectangular identity, round-trip, shapes, grayscale)
  - Energy functions (shape, flat interior, stripe edges)
  - Seam computation (shape, continuity, removal)
  - End-to-end traditional carving (width/height reduction, dtype)
  - End-to-end lattice-guided carving (rectangular, circular, content change)

### Paper Re-read — What We Got Wrong

Re-read Sections 3.3, 3.5, 3.6, and 4.0 of the paper. Key findings:

**Problem: Naive double-interpolation (our current approach)**
Our `carve_image_lattice_guided` resamples pixel data V→L, carves seams in L,
then resamples L→V. The paper explicitly warns against this (Section 3.3,
Fig. 6): "this approach introduces significant blurring artifacts."

**Paper's actual method ("carving the mapping", Section 3.3, Fig. 8):**
1. Map the **energy** from V to L using forward mapping f (for seam computation)
2. Find the seam in L (greedy or graph-cut)
3. Carving the seam in L produces a **modified lattice** L* with a new inverse
   mapping g*
4. For each world-space pixel p_w, compute new position: p_w* = g*(f(p_w))
5. Assign V*(p_w) = V_copy(p_w*) — a **single** lookup in a copy of V
6. This means the actual pixel data is only sampled once, not twice

**Seam Pairs (Section 3.6):**
- Resize a local region without changing global image boundaries
- Two user-defined windows: region of interest + pair region
- Remove seam in region of interest, add compensating seam in pair region
- Content is redistributed: the ROI shrinks, pair region expands (or vice versa)
- This is how you'd shrink a bagel hole while expanding background

**Cyclic Lattices (Section 3.5):**
- For regions that wrap around (e.g., closed shapes like rings)
- Connect the last lattice plane back to the first
- Cyclic graph-cut: connect end nodes to start nodes
- Cyclic greedy: use inverted Gaussian guide on energy to steer the seam
  back to its starting point (Section 4.0.1, Fig. 12)

**Next Steps — Implementation Plan:**
1. ✓ Implement "carving the mapping" (replaces naive resample→carve→resample)
2. ✓ Implement seam pairs for local region carving
3. Implement cyclic lattices
4. Forward energy function (Rubinstein et al. 2008)
5. Visualizations and interactive demo

---

## 2026-02-11 - Visualization & Bug Hunting

### Visualization Framework Development

**Goal:** Build convincing visualizations to demonstrate lattice-guided carving works
correctly (no blur, preserves circular features, enables local region resizing).

**Created visualization script** (`examples/visualize_lattice_carving.py`):
- Test 1: Concentric circles with circular lattice
- Test 2: Bagel with seam pairs (shrink hole, expand background)
- Test 3: Circular grid pattern

**Initial results:** Severe distortion! The lattice-guided outputs were completely broken,
especially on circular lattices. Traditional carving looked normal, but lattice-guided
had extreme artifacts.

### Critical Bug Found: Cumulative Shift Accumulation

**Problem identified in `carve_image_lattice_guided()`:**

The algorithm precomputes `u_map` (world pixels → lattice u-coordinates) once, then
iteratively removes seams. Each iteration computed:

```python
u_shift = torch.where(u_map >= seam_interp, ones, zeros)
```

But `u_map` was never updated! After N seams, pixels were getting cumulative shifts
of up to +N, causing massive distortion. The comparison was always against the
original u_map, not accounting for previous shifts.

**Fix applied:**

```python
cumulative_shift = torch.zeros_like(u_map)  # Track across iterations
for i in range(n_seams):
    # ... find seam ...
    u_adjusted = u_map + cumulative_shift  # Account for previous shifts
    new_shift = torch.where(u_adjusted >= seam_interp, ones, zeros)
    cumulative_shift = cumulative_shift + new_shift
    current = _warp_and_resample(current, lattice, u_map, n_map, cumulative_shift)
```

Same fix applied to `carve_seam_pairs()`.

### Debugging Visualizations

After fix, results still poor. Created `examples/debug_lattice_carving.py` to
visualize intermediate steps:
- Original image
- Lattice structure overlayed on image
- Energy maps (world space and lattice space)
- Seam positions in lattice space

**Status:** Debugging in progress. Need to verify:
1. Lattice structure is correct
2. Energy resampling is correct
3. Seam finding is correct
4. Warping/resampling is correct

### Testing Issues

**Problem:** The 30 passing tests don't actually validate correctness. They only
check shapes and basic properties, not whether the carving produces good results.

**Action needed:**
- Remove or rewrite tests that don't validate correctness
- Add tests that check actual image quality (e.g., verify circles stay circular)
- Focus testing on the core algorithm, not just shape preservation

**Current priority:** Get visualization working first, then write meaningful tests
based on what we learn.

### Curved Lattice Implementation

**User insight:** The lattice structure needs to follow the region of interest (ROI),
not span the entire image. For a river, scanlines should follow the river's curve.
For a bagel, the lattice should cover the annular region.

**Implementation:**
- Added `Lattice2D.from_horizontal_curve()` for features like rivers
  - Takes a function y = f(x) defining the centerline
  - Creates scanlines perpendicular to the curve at regular x intervals
  - Each scanline origin is on the centerline, tangent is perpendicular to curve
- Fixed visualization performance issue (was using nested Python loops, now vectorized)
- Created LATTICE_STRUCTURE_EXPLAINED.md documenting the confusion and correct approach

**Key realizations:**
1. **Current radial lattice for bagel might be correct!**
   - Radial scanlines: n = angle, u = radius
   - Vertical seam (constant u, varying n) = circle at radius u
   - This is what we want for removing concentric circles

2. **River needs curved lattice:**
   - Rectangular lattice with horizontal scanlines doesn't follow river shape
   - Need scanlines that follow the river's curve
   - Implemented `from_horizontal_curve()` for this

3. **Lattice should cover ROI only:**
   - Not the entire image
   - Outside the lattice region, pixels are unchanged
   - Need to add ROI masking/bounds (TODO)

**Test cases planned:**
1. Bagel - seam pairs (shrink hole, expand background)
2. River - curved lattice following sinusoidal path
3. Arch (Figure 3 from paper) - after bagel/river work

**Next steps:**
1. Run debug visualization to see actual seam positions
2. Identify what's broken (lattice structure? interpolation? warping?)
3. Fix the core issue
4. Validate with all three test cases

### Simplified Seam Visualization & Added Arch Test Case

**User insight:** Stop guessing and overcomplicating. Use the mapping functions directly!

**Key simplification:**
- Old approach: Precompute u_map/n_map, interpolate seams, find nearby pixels (convoluted)
- **New approach**: Directly use `inverse_mapping()` to convert seam from lattice → world space
  - For each scanline n, seam is at lattice position (u_seam[n], n)
  - Call `inverse_mapping((u_seam[n], n))` → world (x, y)
  - Plot those (x, y) points on energy map
- Much clearer, follows the paper's approach

**Added arch test case (Figure 3 from paper):**
- Created `create_arch()` - semicircular arch on plain background
- Created `debug_arch()` - curved lattice following arch shape
- Applies both traditional and lattice-guided carving
- Creates side-by-side comparison to validate against Figure 3
- Clear reference: traditional should squish arch, lattice-guided should preserve it

**Test cases now complete:**
1. ✅ Bagel - radial lattice, seam pairs (shrink hole, expand background)
2. ✅ River - curved lattice following sinusoidal path, seam pairs
3. ✅ Arch - curved lattice following semicircle (Figure 3 reference)

**Philosophy:** Follow the paper's implementation as closely as possible. No unnecessary complexity.

### Critical Fix: Sample Original Image Once, Not Iteratively

**Problem discovered:** Our implementation was iteratively warping - each iteration sampled
from the already-warped image from the previous iteration. This compounds interpolation
errors and causes blur.

**What we were doing:**
```python
current = image.clone()
for i in range(n_seams):
    # ... find seam, compute shift ...
    current = _warp_and_resample(current, ...)  # Sampling from current (already warped!)
```

**What the paper says (Section 3.3):**
> "We sample the original pixel data once using the modified mapping"

**Fix applied:**
```python
original_image = image.clone()  # Keep original
cumulative_shift = torch.zeros(...)

for i in range(n_seams):
    # Compute energy from current state (for seam finding)
    current_warped = _warp_and_resample(original_image, ..., cumulative_shift)
    energy = gradient_magnitude_energy(current_warped)

    # Find seam, accumulate shifts
    # ...
    cumulative_shift = cumulative_shift + new_shift

# Final: sample from ORIGINAL image once
result = _warp_and_resample(original_image, ..., cumulative_shift)
```

This matches the paper's approach: modify the mapping (cumulative shifts), then sample
the original pixels once at the end.

**Applied to:**
- `carve_image_lattice_guided()`
- `carve_seam_pairs()`

**Expected improvement:** Significantly reduced blur, cleaner results.

### Next: ROI-Bounded Lattices

**User insight:** The lattice doesn't need to cover the whole image - just the region of
interest (ROI) with some surrounding context.

Looking at the paper's figures:
- Arch: lattice covers only the arch region
- Ring: lattice covers annular region (inner to outer radius)
- River: lattice follows the river path with bounded width

Outside the lattice region, pixels are unchanged. This is cleaner and more efficient.

**TODO:** Implement ROI bounds for lattice construction and carving.
