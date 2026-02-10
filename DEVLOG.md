# Development Log

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
1. Implement "carving the mapping" (replaces naive resample→carve→resample)
2. Implement seam pairs for local region carving
3. Implement cyclic lattices
4. Forward energy function (Rubinstein et al. 2008)
5. Visualizations and interactive demo
