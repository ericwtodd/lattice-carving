# Development Log

Research and implementation notes for the lattice carving project.

---

## 2026-02-12: Fix Cumulative Shift Bug — Switch to Iterative Warping

### Bug discovered: incorrect multi-seam shift composition

The "carving the mapping" implementation (Section 3.3) had a correctness bug. The
code accumulated u-coordinate shifts across all seam iterations and applied a single
final warp from the original image:

```python
# OLD (buggy): cumulative shift from original
cumulative_shift += where(u_map + cumulative_shift >= seam, 1, 0)
# ... after loop:
warp(original_image, cumulative_shift)
```

The problem: composing multiple g\* mappings is **not simply additive**. Each shift
interacts with previous shifts in a position-dependent way. The correct composition
from the paper (Eq. 4) is:

```
total_shift(u) = shift_n(u) + shift_{n-1}(u + shift_n(u)) + ...
```

**Concrete failure**: With seam1=50 and seam2=49, pixel at u=49 should sample from
original u=51 (shift=2). The old code produced shift=1 (off by one). These errors
compound with each additional seam, creating jagged boundaries at seam positions.

For single-seam carving (n_seams=1), the old code was correct. The bug only appeared
with multiple seams.

### Fix: iterative warping (paper's actual approach)

Replaced cumulative shifts with iterative warping — each iteration reads from the
*current* image state and applies a single-step shift:

```python
# NEW (correct): iterative warp from current state
for i in range(n_seams):
    shift = where(u_map >= seam, 1, 0)
    current_image = warp(current_image, shift)
```

This matches the paper's Section 3.3 / Fig. 8 exactly: "the values in V are updated
using a carved mapping into a copy of V." Each iteration introduces one bilinear
interpolation (N total for N seams), which the paper considers acceptable (Section
6.1).

Also fixed `_interpolate_seam()` for cyclic lattices — was clamping `n_ceil` to
`n_lines - 1` instead of wrapping with modular arithmetic.

### Test suite simplified

Removed tests that validated implementation details of the (now-removed) cumulative
shift approach. Kept behavioral tests: identity (zero seams), output sanity,
boundary preservation (seam pairs), dimension preservation.

### Visual results

All demo outputs regenerated. The sawtooth artifacts on the synthetic bagel persist
at 200px / 256 scanlines — this is a resolution/aliasing issue (see resolution sweep
analysis below), not a shift composition issue. The fix eliminates compounding
off-by-one errors that would worsen with more seams and non-adjacent seam positions.

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

## 2026-02-12: DP Seam Finding & Resolution Investigation

### DP seam implementation

Added three DP seam functions to `src/seam.py`:
- `dp_seam(energy, direction)` — Standard Avidan & Shamir 2007: fill cumulative cost matrix top→bottom, backtrack for globally optimal seam
- `dp_seam_windowed(energy, col_range)` — Masks columns outside window to infinity, then runs dp_seam
- `dp_seam_cyclic(energy, col_range)` — Tries each starting column, forces seam[0]==seam[-1], picks lowest-cost closed seam

Wired into `carve_image_lattice_guided()` and `carve_seam_pairs()` via `method='dp'` parameter (default). Cyclic lattices automatically use `dp_seam_cyclic`.

### Resolution sweep experiment

Tested 5 configurations: (200px/64), (200px/256), (200px/1024), (600px/256), (600px/1024). Both zero-seam roundtrip error and 5-seam-pair carving. See "Sawtooth artifacts investigation" in Next Steps section for full analysis.

Key finding: the artifacts are a resolution/aliasing problem, not a seam algorithm problem. Higher image res + denser lattice = smoother results.

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

## 2026-02-12: Widen Lattice Regions for Bagel Demo

### Problem

The synthetic bagel demo had a very tight ROI: 20 columns for 18 seams (90% compression). The pair region was only 21 columns. This caused later seams to operate in increasingly degenerate energy fields, contributing to sawtooth artifacts.

### Changes

Widened `perp_extent` from 90 to 120 in `setup_synthetic_bagel()`:

```
Before: perp=90, lattice_w=180
  ROI: (80, 100) — 20 cols for 18 seams
  Pair: (159, 180) — 21 cols

After: perp=120, lattice_w=240
  ROI: (45, 195) — 150 cols for 20 seams (full ring body)
  Pair: (195, 240) — 45 cols (outer background)

Layout:
  u=0          u=45         u=120         u=195       u=240
  |--hole bgnd--|--------- ring body ---------|--outer bgnd--|
     45 cols    |<-------- ROI (150) -------->|<-pair (45)->|
```

Also improved the lattice overlay visualization:
- Increased default `n_scanlines` from 16 to 48 for denser grid
- Added ROI boundary curves (cyan dashed) and pair boundary curves (magenta dashed) in world-space overlays

### Rationale

- Full ring body as ROI gives seams 150 columns to work with (7.5x headroom for 20 seams)
- Pair region in outer background has 45 columns (2.25x headroom for 20 seams)
- Wider lattice coverage means more background is available for compensation
- Denser overlay + boundary visualization makes it easier to inspect lattice alignment

---

## 2026-02-12: Lattice-Space Carving — Attempted & Reverted

### Attempted approach

Tried replacing the "carving the mapping" approach with direct lattice-space pixel operations: `remove_seam()` / `insert_seam()` on the regular lattice grid, then `resample_from_lattice_space()` for world-space display.

### Why it failed

The paper explicitly warns against this in Section 3.3 (the "naive approach"):

> "The most straightforward but naive approach would be to simply map data values in V to L, compute and carve seams from these values in L, and then remap them back to V. However, due to the non-uniformity of the mapping, the values do not map perfectly between V and L, and this approach introduces significant blurring artifacts... Performing multiple carving operations compounds these artifacts."

Results confirmed: visible lines/banding through the bagel body, grass artifacts on river, background artifacts on arch — all from the compounding V→L→V interpolation.

### What was kept

- `insert_seam()` in `src/seam.py` — still useful, mirrors `remove_seam()`
- `greedy_seam_cyclic` improvements (funnel-shaped Gaussian guide, `return_guided_energy` parameter)
- Setup function improvements (sesame seeds on bagel, wider ROI/pair ranges, buffer between windows)

### Lesson learned

The paper's "carving the mapping" approach (cumulative shifts + single warp from original) is fundamentally correct. The lattice mapping is non-uniform, so round-tripping through it compounds errors. Pixel data should only be sampled ONCE from the original image.

---

## Next Steps

### Sawtooth artifacts investigation — RESOLVED

**Summary**: The sawtooth/scalloping artifacts on bagel seam pairs were caused by insufficient resolution, not seam quality.

**What we tried (didn't fix it):**
1. Multi-greedy (n_candidates=16) — All candidates wander equally in flat energy
2. DP seam finding — Globally optimal, but "optimal" in flat energy still wanders to chase noise minima
3. Cyclic-aware DP seams — Ensures seam[0]==seam[-1] but doesn't prevent wandering in between
4. Narrow ROI (±5 instead of ±15) + more scanlines (256 instead of 64) — Helped somewhat

**Root cause discovery:**
- Zero-seam roundtrip (forward_mapping → inverse_mapping with NO carving) already shows max error of 0.530 on the synthetic bagel
- Error is concentrated at sharp ring boundaries (uniform color → sharp edge → sub-pixel displacement creates large intensity delta)
- The scalloping pattern matches the lattice scanline angular spacing — each scanline introduces a tiny positional error that becomes visible at sharp edges

**Resolution sweep results (output/fig_resolution_sweep.png):**

| Config | Roundtrip error | 5-seam carving quality |
|--------|----------------|----------------------|
| 200px / 64 lines | Heavy radial pattern | Worst scalloping |
| 200px / 256 lines | Moderate | Better but still visible |
| 200px / 1024 lines | Low | Significantly smoother |
| 600px / 256 lines | Moderate | Better (higher image res helps) |
| 600px / 1024 lines | Minimal | **Best result** — smooth ring |

**Conclusion**: Both image resolution and lattice resolution matter. The sawtooth is a sampling/aliasing artifact from the discrete lattice grid. The paper's demos use high-res images with proportionally dense lattices. At 600px/1024 scanlines, results look close to the paper.

**Key insight from paper (Section 6.0.7)**: "retarget the region between the seam pairs rather than the entire image" — validity masking (only warping pixels within the lattice region) would further reduce artifacts by not roundtripping pixels that don't need to move.

### Next: browser demo

Pre-compute carving iterations as JSON, build static HTML/JS viewer with seam count slider.

---

## 2026-02-12: Recent Changes, Bug Fixes, and Sawtooth Analysis

### Changes made (explored by user)

1. **Grow/shrink mode for `carve_seam_pairs`**: Added `mode` parameter ('shrink' or 'grow') to both `carve_seam_pairs()` in `src/carving.py` and `generate_seam_pair_demo()`. Uses `roi_sign` (+1 for shrink, -1 for grow) to flip which region compresses vs expands. This enables the side-by-side shrink/original/grow comparison GIFs.

2. **Extracted `_compute_valid_mask()` helper**: The roundtrip validity test (forward → inverse → check position error) was duplicated in `generate_demo_data.py` and implicitly needed in `carving.py`. Now a shared function in `carving.py` with a configurable `threshold` parameter (default 3.0 pixels).

3. **Valid mask applied in carving pipelines**: Both `carve_image_lattice_guided()` and `carve_seam_pairs()` now zero out shifts for invalid pixels and composite warped results over the original. This prevents nonsensical shifts on pixels far from the lattice.

4. **Wider ROI/pair ranges for bagel**: Changed from (center±5, last 11) to (center±10, last 21). Still tight — see analysis below.

5. **Figure sizing for overlays**: Switched `save_world_seam_overlay()` and `save_lattice_overlay()` from aspect-ratio-based sizing with `bbox_inches='tight'` to pixel-exact DPI-based sizing with `fig.add_axes([0, 0, 1, 1])`.

6. **GIF generators**: Two new scripts in `web/`:
   - `generate_seam_carving_gif.py` — single-demo animated GIF from step images
   - `generate_gif_comparison.py` — side-by-side shrink/original/grow comparison

7. **Comment cleanup**: Stripped verbose docstrings and inline comments from `src/carving.py` to reduce noise.

### Bug fix: white line at bottom of overlay images

**Symptom**: `image_seams.png` and `image_lattice.png` had an all-white (255) bottom pixel row, visible in the web viewer when toggling off seams/lattice overlays.

**Root cause**: `ax.set_ylim(H, 0)` in the matplotlib overlay functions. Matplotlib `imshow` places pixel centers at integer coordinates, so the image extends from y=-0.5 to y=H-0.5. Setting ylim to H leaves a 0.5-pixel gap of white figure background at the bottom.

**Fix**: Changed to `ax.set_ylim(H - 0.5, -0.5)` and `ax.set_xlim(-0.5, W - 0.5)` in both `save_world_seam_overlay()` and `save_lattice_overlay()`.

### DP seam quality analysis with real data

Analyzed seam metadata from the synthetic bagel shrink demo (18 steps, 1024 scanlines, ROI [80,100], pair [159,180]):

- **max_jump = 1** on all seams (correct — DP guarantees 8-connectivity)
- **mean_jump ≈ 0.63–0.68** — seams change column ~65% of the time
- ROI seams cluster toward the upper end of the window (near u=100)
- Pair seams are more consistent (range [166–178])

The seams are *individually* smooth (no jumps > 1 pixel) but *collectively* zigzaggy because the flat-energy ring body has many equally-optimal paths. The cumulative effect of 18 independent zigzaggy seams creates a noisy shift field, producing the sawtooth pattern in the warped image.

### Region tightness problem

Current configuration: **ROI = 20 columns wide, 18 seams = only 2 columns of headroom**.

After 18 shrink operations, the ROI content has been compressed into just 2 effective lattice columns. This means:
- Later seams are finding paths through increasingly compressed (duplicated) content
- The energy field in the ROI becomes nearly uniform, making seam paths arbitrary
- The pair region (21 columns for 18 expansions) is similarly tight — content gets stretched significantly

**The regions do NOT need explicit repositioning** after each step because the energy is resampled from the warped image each iteration, which already reflects all prior shifts. The seam finder sees the current state of the content, not the original. The roi_range/pair_range stay fixed in lattice coordinates, which is correct.

**However, the regions need to be wider** to avoid exhausting the available space. Recommendation: ROI should be at least 2–3x the number of seams (40–60 columns for 18 seams).

### Root causes of sawtooth (updated understanding)

Three factors combine:

1. **Flat energy in ring body**: Gradient magnitude ≈ 0 across the uniform-color ring. DP picks through noise, producing zigzaggy seams even though they're technically optimal.

2. **Tight ROI**: 20 columns for 18 seams leaves no room. Later seams are forced into a shrinking effective space with increasingly degenerate energy.

3. **Independent seam paths**: Each iteration finds a new seam independently. Since the energy is flat, successive seams don't align, creating an irregular cumulative shift field.

### Potential improvements to try (ordered by expected impact)

1. **Wider ROI + pair regions** — Simple config change, reduces crowding. Try (center±25, last 51) for 18 seams.

2. **Energy smoothing / blur before DP** — Gaussian blur on the lattice-space energy map before seam finding. Smooths out noise, encouraging straighter seams in flat regions. The paper mentions this in Section 4.

3. **Forward energy instead of gradient magnitude** — `forward_energy()` penalizes the cost of *creating new edges* when a pixel is removed, not just the current gradient. This gives non-zero costs even in flat regions (removing a pixel in a flat area still creates a visible seam). Already implemented in `src/energy.py` but not wired into the carving pipeline.

4. **Seam guide / straightness prior** — Add a penalty for lateral movement in the DP cost. Something like `M[i] = energy[i] + min(neighbors) + lambda * |col_shift|`. This biases toward straight vertical seams in flat energy.

5. **Ordered seam removal** — Instead of finding one seam per iteration, find multiple seams in a single DP pass (like Avidan & Shamir's k-seam method) to ensure they don't overlap and are well-distributed.
