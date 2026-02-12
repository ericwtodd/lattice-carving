# Development Log

Research and implementation notes for the lattice carving project.

---

## 2026-02-11: Visual Validation & Seam Quality Discovery

### What was validated

Ran `reproduce_figures.py` on 5 test cases. Results:

1. **Traditional carving** — Works correctly. Seams avoid the bright stripe in a gradient image, removing low-energy columns as expected.

2. **Arch carving (Figure 3 analog)** — Lattice-guided carving preserves the arch silhouette while traditional carving squishes it. This demonstrates the core value of the paper's approach.

3. **No-blur comparison (Section 3.3)** — The "carving the mapping" approach (single interpolation from original) preserves checkerboard sharpness, while the naive double-interpolation approach visibly blurs. Confirmed by visual inspection.

4. **Synthetic bagel seam pairs** — Circular ring body can be shrunk/grown via cyclic lattice + seam pairs. The mechanism works (ring visibly changes size, background compensates) but produces **sawtooth artifacts** at the boundary where content shifts.

5. **Real double-bagel** — Left bagel half can be targeted independently. Same sawtooth artifacts as synthetic case.

### Bug fixes during validation

**Critical: Cyclic forward_mapping broken for closed curves**

When using `from_curve_points(cyclic=True)` (e.g., a circular lattice around a bagel), the forward mapping produced garbage for pixels inside the curve. The synthetic bagel's hole became a vertical rectangle instead of staying circular.

**Root cause**: The tangent projection penalty in `forward_mapping` adds 1e10 to scanlines where the query point's tangent projection is negative. For non-cyclic lattices, this correctly disambiguates which side of a scanline a point is on. But for cyclic lattices (closed curves), all scanlines have their tangent pointing "outward" — so pixels inside the curve have negative tangent projection to EVERY scanline, causing the argmin to pick arbitrarily.

**Fix**: Two changes in `lattice.py`:
1. Skip tangent projection penalty entirely for cyclic lattices (normal distance alone suffices since adjacent scanlines don't oppose each other)
2. Fractional n computation uses modular arithmetic `(best_n + 1) % n_lines` for cyclic lattices instead of clamping

### The sawtooth artifact problem

**Symptom**: After seam pair carving on the bagel, the boundary of the modified region has regular scalloped/sawtooth edges instead of smooth curves.

**Diagnosis via lattice-space view**: The seam in lattice space wanders wildly across the 30px-wide ROI window. In world space, this wandering seam creates an irregular shift boundary.

**Root cause**: The bagel body has approximately uniform color, so gradient energy is near-zero across the ring. The greedy seam has no preference in flat energy — it wanders randomly at each step, producing a jagged path.

**Multi-greedy doesn't help**: Added `n_candidates=16` parameter to try 16 different starting points and keep the lowest-energy seam. Result: no visible improvement. Multi-greedy helps when there are distinct energy valleys to find (structural variation), but in uniformly flat energy, all 16 candidates wander equally.

**Solution needed**: Dynamic programming (DP) seam finding. DP computes the globally minimum cumulative cost path, which in flat regions naturally produces a smooth/straight seam (all paths have equal cost but DP's argmin is deterministic and smooth). For 2D images, DP gives the same result as graph-cut (the paper's primary method).

### Seam quality hierarchy

From the paper and our experiments:

| Method | Complexity | Quality | When it helps |
|--------|-----------|---------|---------------|
| Greedy | O(H) per seam | Poor in flat energy | Fast prototyping |
| Multi-greedy (k starts) | O(H * k) | Slightly better | When energy has distinct valleys |
| DP (Avidan & Shamir 2007) | O(H * W) | Optimal for 2D | **Always** — this is what we need |
| Graph-cut | O(H * W * log(HW)) | Optimal | 3D volumes (equivalent to DP for 2D) |

The paper's primary method is graph-cut, with greedy as a "500x faster" alternative. For 2D images (our case), DP gives the same optimal result as graph-cut at O(H * W) cost.

### Color image handling

Energy computation: RGB is converted to grayscale using luma weights (0.299R + 0.587G + 0.114B), then Sobel gradients are computed on the grayscale. This matches Avidan & Shamir 2007. The seam found in grayscale energy is applied to all color channels during the shift/resample step.

---

## 2026-02-11: Cyclic Lattice Support

Added `cyclic=True` parameter to `Lattice2D.from_curve_points()`. For closed curves (circles, rings), the last scanline connects back to the first. Required changes:
- Scanline construction: don't close the curve (the last point IS the first point)
- `resample_to_lattice_space`: wrap-around indexing for cyclic grids
- Forward mapping: modular `n` coordinate, no tangent penalty (see bug fix above)

---

## 2026-02-11: Seam Pairs Clarification

The user's goal: "select the left bagel and then grow (or shrink) it without affecting the rest of the image." This is the seam pairs mechanism from Section 3.6.

- **Shrink bagel body**: ROI = ring body (center_u ± 15), Pair = outer background. Removing seams from ROI compresses the ring; inserting in pair expands background to compensate.
- **Grow bagel body**: Swap ROI and pair regions.
- Net shift cancels outside both windows, preserving image boundaries.

Initially implemented "shrink the hole" by mistake — user clarified they want to shrink/grow the bagel body itself.

---

## Architecture Decisions

- **PyTorch throughout**: All operations use torch tensors. `grid_sample` for bilinear interpolation. Enables future GPU acceleration.
- **Vectorized mappings**: Forward/inverse lattice mappings process all points in a single batch operation. No Python loops over pixels.
- **Single interpolation**: "Carving the mapping" (Section 3.3) samples pixel data only once from the original image. Energy can be interpolated freely (it's computed fresh each iteration anyway), but pixel data is never double-interpolated.
- **Cumulative shifts**: Instead of iteratively warping the image (which compounds interpolation blur), we accumulate u-coordinate shifts across all iterations and apply them in a single final warp.

---

## Next Steps

### Fix sawtooth artifacts (before demo)

Two issues compound to create the sawtooth artifacts on bagel seam pairs:

1. **DP seam finding** — The primary fix. Standard Avidan & Shamir 2007 algorithm: fill cumulative cost matrix top-to-bottom, backtrack to find globally optimal seam. In flat-energy regions (uniform bagel body), DP produces smooth/straight seams instead of random-walk wandering. For 2D images, DP = graph-cut quality at O(H * W) cost.

2. **Cyclic-aware seams in seam pairs** — Currently `carve_seam_pairs()` calls `greedy_seam_windowed()` which is NOT cyclic-aware. For cyclic lattices, scanline 0 and scanline N-1 are adjacent in world space (they wrap around), but the seam treats them as disconnected endpoints. This creates a visible discontinuity at the wrap point. Need to either: (a) use `greedy_seam_cyclic()` (Gaussian guide) in the seam pairs pipeline when the lattice is cyclic, or (b) implement DP with cyclic constraint (seam[0] == seam[N-1]).

Grid resolution (48-64 scanlines) is probably NOT a factor — artifacts are much larger than the angular step. Easy to verify by bumping to 128.

### Then: browser demo

Pre-compute carving iterations as JSON, build static HTML/JS viewer with seam count slider.
