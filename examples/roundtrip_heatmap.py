"""
Roundtrip diff heatmap: forward_mapping -> inverse_mapping -> pixel error.
Shows accuracy of forward_mapping for rectangular, circular, and curve lattices.

Run:
    conda run -n lattice-carving python examples/roundtrip_heatmap.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.lattice import Lattice2D

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def roundtrip_error(lattice, H, W):
    """Per-pixel error: distance(forward(inverse(p)), p) for grid of world points."""
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (N, 2) as (x, y)

    lattice_pts = lattice.forward_mapping(pts)
    recovered = lattice.inverse_mapping(lattice_pts)

    err = torch.sqrt(((pts - recovered) ** 2).sum(dim=1)).reshape(H, W)
    return err.numpy()


fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# --- Case 1: Rectangular (should be ~0 everywhere) ---
H, W = 80, 120
lat = Lattice2D.rectangular(H, W)
err = roundtrip_error(lat, H, W)
im = axes[0, 0].imshow(err, cmap='hot', vmin=0, vmax=0.01)
axes[0, 0].set_title(f"Rectangular\nmax={err.max():.5f}, mean={err.mean():.5f}")
plt.colorbar(im, ax=axes[0, 0])

# --- Case 2: Circular ---
H, W = 120, 120
cx, cy = 60.0, 60.0
radius = 45.0
lat = Lattice2D.circular((cx, cy), radius, n_lines=72)
err = roundtrip_error(lat, H, W)
im = axes[0, 1].imshow(err, cmap='hot', vmin=0, vmax=2)
axes[0, 1].set_title(f"Circular (r=45, n_lines=72)\nmax={err.max():.4f}")
plt.colorbar(im, ax=axes[0, 1])
theta = np.linspace(0, 2 * np.pi, 200)
axes[0, 1].plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta),
                'w--', linewidth=1.5, label='extent')
axes[0, 1].legend(fontsize=7)

# Circular — inside-circle only
yy_np, xx_np = np.mgrid[0:H, 0:W]
inside = (xx_np - cx) ** 2 + (yy_np - cy) ** 2 <= radius ** 2
err_inside = np.where(inside, err, np.nan)
im = axes[1, 0].imshow(err_inside, cmap='hot', vmin=0, vmax=0.5)
axes[1, 0].set_title(
    f"Circular (inside only)\nmax={np.nanmax(err_inside):.4f}, "
    f"mean={np.nanmean(err_inside):.4f}"
)
plt.colorbar(im, ax=axes[1, 0])

# --- Case 3: Sinusoidal from_curve_points ---
H, W = 160, 200
t = torch.linspace(0, 3 * math.pi, 40)
cx_curve = torch.linspace(20, 180, 40)
cy_curve = 80 + 40 * torch.sin(t)
control_pts = torch.stack([cx_curve, cy_curve], dim=1)
lat = Lattice2D.from_curve_points(control_pts, n_lines=40, perp_extent=50.0)
err = roundtrip_error(lat, H, W)
im = axes[0, 2].imshow(err, cmap='hot', vmin=0, vmax=3)
axes[0, 2].set_title(f"Sinusoidal curve (n_lines=40, extent=50)\nmax(near-curve)={err[err < 5].max():.4f}")
plt.colorbar(im, ax=axes[0, 2])
axes[0, 2].plot(cx_curve.numpy(), cy_curve.numpy(), 'w-', linewidth=2, label='centerline')
axes[0, 2].legend(fontsize=7)

# Sinusoidal — near-curve only (err < 5px = valid lattice region)
err_near = np.where(err < 5, err, np.nan)
im = axes[1, 1].imshow(err_near, cmap='hot', vmin=0, vmax=1)
axes[1, 1].set_title(
    f"Sinusoidal (near-curve only)\nmax={np.nanmax(err_near):.4f}, "
    f"mean={np.nanmean(err_near):.4f}"
)
plt.colorbar(im, ax=axes[1, 1])
axes[1, 1].plot(cx_curve.numpy(), cy_curve.numpy(), 'w-', linewidth=2)

# Error histogram
axes[1, 2].hist(err_near[~np.isnan(err_near)].ravel(), bins=50,
                color='steelblue', edgecolor='k')
axes[1, 2].set_xlabel("Roundtrip pixel error")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_title("Error distribution (sinusoidal, near-curve)")
axes[1, 2].axvline(0.1, color='r', linestyle='--', label='0.1 px')
axes[1, 2].legend()

fig.suptitle(
    "forward_mapping Roundtrip Error: forward(world) → inverse → distance from original\n"
    "Near-zero = mapping is accurate",
    fontsize=12
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_roundtrip_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
