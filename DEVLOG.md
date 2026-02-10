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

**Next Steps:**
- Test on circular carving (for bagel hole preservation)
- Handle cyclic lattices for closed shapes
- Forward energy function (Rubinstein et al. 2008)
