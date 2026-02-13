# Lattice Carving

Implementation of ["Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"](https://doi.org/10.1145/3478513.3480544) (Flynn et al., ACM TOG 2021).

Traditional seam carving only works on rectangular images — seams must flow monotonically from one edge to the other. This paper introduces **lattice-guided seam carving**: define a non-uniform lattice that follows the shape of the region you want to carve, map the energy into "lattice index space" (a rectified grid), find seams there, then map the result back. This lets you carve along curves, around holes, and through arbitrary regions while preserving shape silhouettes.

<p align="left">
<img src="web/assets/river_comparison.gif" style="width:100%;"/>
</p>

## Quick Start

```bash
# Install
conda create -n lattice-carving python=3.11
conda activate lattice-carving
pip install -e .

# Run tests (69 tests)
python -m pytest tests/ -v
```

## Carve Your Own Image

The typical workflow for carving along a curve in your own image:

### 1. Define a centerline curve

**Option A — Draw it in the browser (recommended):**

Open `web/curve_drawer.html` in a browser. Load your image, draw a freehand
curve along the feature you want to carve (e.g. a river, road, arch), adjust
the number of control points and smoothing, then click **Copy JSON**.

Save the JSON to a file:

```bash
# Paste your copied JSON into this file
cat > output/my_centerline.json << 'EOF'
[[x1,y1], [x2,y2], ...]
EOF
```

**Option B — SAM auto-segmentation:**

If your feature is visually distinct (dark water, etc.), SAM can extract a
mask and derive the centerline automatically:

```bash
# Install SAM dependencies (one-time)
pip install segment-anything scikit-image opencv-python-headless

# Download SAM checkpoint (375MB, one-time)
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Run SAM segmentation demo on river.jpg
python examples/segment_river_demo.py
# → saves output/river_centerline.json + diagnostic figure
```

### 2. Build a lattice and carve

```python
import json, torch
from PIL import Image
import numpy as np
from src.lattice import Lattice2D
from src.carving import carve_seam_pairs

# Load image
img = Image.open("river.jpg").convert("RGB")
image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

# Load centerline (from curve drawer or SAM)
with open("output/river_centerline.json") as f:
    pts = torch.tensor(json.load(f), dtype=torch.float32)

# Build lattice along the curve
perp = 80  # how far the lattice extends perpendicular to the curve
lattice = Lattice2D.from_curve_points(pts, n_lines=512, perp_extent=perp)
lattice.smooth(max_iterations=100)

# Define ROI (what to shrink) and pair region (where to compensate)
lattice_w = int(2 * perp)  # 160
center_u = perp             # 80
river_half = 25             # half-width of the feature in lattice u-coords
roi_range = (int(center_u - river_half), int(center_u + river_half))
pair_range = (int(center_u + river_half + 5), lattice_w)

# Carve!
result = carve_seam_pairs(
    image, lattice, n_seams=12,
    roi_range=roi_range, pair_range=pair_range,
    lattice_width=lattice_w, mode='shrink',  # or 'grow'
)
```

### 3. Generate pipeline figures

The pipeline figure script picks up `output/river_centerline.json` automatically:

```bash
python examples/generate_pipeline_figure.py
# → output/pipeline_real_river.png (5-panel: original, ROI, lattice, energy+seams, result)
```

## What's Implemented

| Paper Section | Feature | Module |
|---|---|---|
| 3.1 | Lattice construction (rectangular, circular, curve-based, cyclic) | `src/lattice.py` |
| 3.2 | Forward/inverse mapping (vectorized) | `src/lattice.py` |
| 3.3 | "Carving the mapping" (single-interpolation, no blur) | `src/carving.py` |
| 3.4.2 | Lattice smoothing (iterative mean filter) | `src/lattice.py` |
| 3.5 | Cyclic lattices for closed curves | `src/lattice.py` |
| 3.6 | Seam pairs (local shrink/expand without changing boundaries) | `src/carving.py` |
| 4.0.1 | DP + greedy + multi-greedy seam computation | `src/seam.py` |
| Eq. 6 | Gradient magnitude energy (L1 norm) | `src/energy.py` |
| Rubinstein 2008 | Forward energy (edge-aware DP) | `src/energy.py` |
| — | SAM segmentation + mask-to-centerline extraction | `src/roi_extraction.py` |

## Key Concepts

**Lattice types:**
- `Lattice2D.rectangular(H, W)` — standard grid (equivalent to traditional seam carving)
- `Lattice2D.circular(center, radius, n_lines)` — radial scanlines for circular regions
- `Lattice2D.from_curve_points(points, n_lines, perp_extent)` — scanlines perpendicular to any user-defined curve (Figure 9)

**Carving approaches:**
- `carve_image_traditional()` — standard seam carving (reduces image dimensions)
- `carve_image_lattice_guided()` — carves through the lattice mapping, preserving image dimensions. Supports `roi_bounds` to only warp pixels within the lattice region.
- `carve_seam_pairs()` — removes a seam in the ROI and adds one in a pair region, keeping global boundaries unchanged (Section 3.6)

**Centerline → lattice coordinate system:**
- `perp_extent` controls how far the lattice extends on each side of the curve (in pixels)
- `lattice_w = 2 * perp_extent` is the total width of lattice space
- `u = perp_extent` is the center of the lattice (the curve itself)
- `roi_range` defines which u-coordinates to shrink/grow
- `pair_range` defines where to compensate (expand/shrink to preserve boundaries)

## Project Structure

```
src/
  lattice.py         # Lattice2D: construction, mapping, resampling, smoothing
  energy.py          # gradient_magnitude_energy, forward_energy, normalize_energy
  seam.py            # dp_seam, greedy_seam, windowed, cyclic, multi-greedy, remove_seam
  carving.py         # High-level carving orchestration
  roi_extraction.py  # SAM segmentation + mask-to-centerline extraction
tests/               # 69 tests across 5 modules
examples/
  reproduce_figures.py          # Paper figure comparisons
  generate_pipeline_figure.py   # 5-panel pipeline figure (synthetic + real river)
  generate_demo_data.py         # Step-by-step demo data for browser viewer
  segment_river_demo.py         # SAM segmentation demo
web/
  curve_drawer.html  # Browser-based curve drawing tool
output/              # Generated figures and cached data (gitignored)
models/              # SAM checkpoints (gitignored)
```

## Tools

### Browser Curve Drawer (`web/curve_drawer.html`)

Open directly in a browser (no server needed). Load an image, draw a curve,
adjust smoothing and point count, then copy the JSON. The output format
`[[x1,y1], [x2,y2], ...]` is ready for `Lattice2D.from_curve_points()` or
can be saved as `output/river_centerline.json` for the pipeline scripts.

### SAM Segmentation (`src/roi_extraction.py`)

Automatic or point-prompted segmentation using Meta's Segment Anything Model.
Extracts a binary mask, skeletonizes it, finds the longest path through the
skeleton graph, and resamples to evenly-spaced control points.

```python
from src.roi_extraction import segment_river
centerline = segment_river("river.jpg", n_control_points=40)
```

## Visual Validation

```bash
# Paper figure comparisons (synthetic demos)
python examples/reproduce_figures.py

# Pipeline figure with real river
python examples/generate_pipeline_figure.py

# Step-by-step demo data (for browser viewer)
python examples/generate_demo_data.py
```

## References

- Flynn, S., Hart, D., Morse, B., Holladay, S., & Egbert, P. (2021). Generalized Fluid Carving with Fast Lattice-Guided Seam Computation. *ACM Trans. Graph.*, 40(6), Article 255.
- Avidan, S., & Shamir, A. (2007). Seam Carving for Content-Aware Image Resizing. *ACM Trans. Graph.*, 26(3).
- Rubinstein, M., Shamir, A., & Avidan, S. (2008). Improved Seam Carving for Video Retargeting. *ACM Trans. Graph.*, 27(3).
