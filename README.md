# Lattice Carving

Implementation of ["Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"](https://doi.org/10.1145/3478513.3480544) (Flynn et al., ACM TOG 2021).

Traditional seam carving only works on rectangular images — seams must flow monotonically from one edge to the other. This paper introduces **lattice-guided seam carving**: define a non-uniform lattice that follows the shape of the region you want to carve, map the energy into "lattice index space" (a rectified grid), find seams there, then map the result back. This lets you carve along curves, around holes, and through arbitrary regions while preserving shape silhouettes.

## What's Implemented

| Paper Section | Feature | Module |
|---|---|---|
| 3.1 | Lattice construction (rectangular, circular, curve-based, cyclic) | `src/lattice.py` |
| 3.2 | Forward/inverse mapping (vectorized) | `src/lattice.py` |
| 3.3 | "Carving the mapping" (single-interpolation, no blur) | `src/carving.py` |
| 3.4.2 | Lattice smoothing (iterative mean filter) | `src/lattice.py` |
| 3.5 | Cyclic lattices for closed curves | `src/lattice.py` |
| 3.6 | Seam pairs (local shrink/expand without changing boundaries) | `src/carving.py` |
| 4.0.1 | Greedy + multi-greedy seam computation, Gaussian guide for cyclic closure | `src/seam.py` |
| Eq. 6 | Gradient magnitude energy (L1 norm) | `src/energy.py` |
| Rubinstein 2008 | Forward energy (edge-aware DP) | `src/energy.py` |

## Quick Start

```bash
# Install
conda create -n lattice-carving python=3.11
conda activate lattice-carving
pip install -e .

# Run tests
python -m pytest tests/ -v

# Basic usage
python -c "
import torch
from src import Lattice2D, carve_image_lattice_guided

image = torch.rand(3, 256, 256)
lattice = Lattice2D.rectangular(256, 256)
result = carve_image_lattice_guided(image, lattice, n_seams=10)
print(result.shape)  # (3, 256, 256) — same dims, content shifted
"
```

## Key Concepts

**Lattice types:**
- `Lattice2D.rectangular(H, W)` — standard grid (equivalent to traditional seam carving)
- `Lattice2D.circular(center, radius, n_lines)` — radial scanlines for circular regions
- `Lattice2D.from_curve_points(points, n_lines, perp_extent)` — scanlines perpendicular to any user-defined curve (Figure 9)

**Carving approaches:**
- `carve_image_traditional()` — standard seam carving (reduces image dimensions)
- `carve_image_lattice_guided()` — carves through the lattice mapping, preserving image dimensions. Supports `roi_bounds` to only warp pixels within the lattice region.
- `carve_seam_pairs()` — removes a seam in the ROI and adds one in a pair region, keeping global boundaries unchanged (Section 3.6)

## Project Structure

```
src/
  lattice.py    # Lattice2D: construction, mapping, resampling, smoothing
  energy.py     # gradient_magnitude_energy, forward_energy, normalize_energy
  seam.py       # greedy_seam, windowed, cyclic, multi-greedy, remove_seam
  carving.py    # High-level carving orchestration
tests/          # 68 tests across 5 modules
examples/       # Visualization and test image generation scripts
```

## Visual Validation

Run `python examples/reproduce_figures.py` to generate comparison images in `output/`:

- **Traditional carving** — seams avoid high-energy edges
- **Arch carving** (Figure 3) — lattice-guided preserves arch shape vs. traditional distortion
- **No-blur comparison** (Section 3.3) — carving-the-mapping preserves sharpness vs. naive double-interpolation
- **Synthetic bagel seam pairs** (Figure 10) — shrink/grow ring body with cyclic lattice
- **Real double-bagel seam pairs** — shrink/grow left bagel half independently

## References

- Flynn, S., Hart, D., Morse, B., Holladay, S., & Egbert, P. (2021). Generalized Fluid Carving with Fast Lattice-Guided Seam Computation. *ACM Trans. Graph.*, 40(6), Article 255.
- Avidan, S., & Shamir, A. (2007). Seam Carving for Content-Aware Image Resizing. *ACM Trans. Graph.*, 26(3).
- Rubinstein, M., Shamir, A., & Avidan, S. (2008). Improved Seam Carving for Video Retargeting. *ACM Trans. Graph.*, 27(3).
