"""
Seam overlay diagnostic: show where seams are placed before carving.

For each test case: original image | seams highlighted in red | carved result.

Run:
    conda run -n lattice-carving python examples/seam_overlay_diagnostic.py
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
from src.carving import (
    _precompute_forward_mapping, _interpolate_seam,
    carve_image_traditional, carve_image_lattice_guided, carve_seam_pairs,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import dp_seam

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_seams(image_np, seam_masks, color=(1.0, 0.0, 0.0)):
    """Overlay seam pixels in red on an RGB numpy image (H, W, 3)."""
    out = image_np.copy()
    combined = np.zeros(image_np.shape[:2], dtype=bool)
    for m in seam_masks:
        combined |= m
    out[combined] = color
    return out


def get_seam_world_mask(lattice, H, W, n_seams=1, method='dp'):
    """Return a list of boolean masks (H, W) showing seam pixels in world space."""
    device = 'cpu'
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    world_pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    lat_pts = lattice.forward_mapping(world_pts)
    u_map = lat_pts[:, 0].reshape(H, W)
    n_map = lat_pts[:, 1].reshape(H, W)

    # Placeholder energy: uniform (seam goes wherever DP decides)
    # We'll use actual image energy per call below
    return u_map, n_map


def seam_mask_from_energy(lattice, image, lattice_width, n_seams=3):
    """Find N seams on an image, return list of world-space seam masks."""
    H, W = image.shape[-2], image.shape[-1]
    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic
    u_map, n_map = get_seam_world_mask(lattice, H, W)

    masks = []
    img = image.clone()
    if img.dim() == 2:
        img = img.unsqueeze(0)

    for _ in range(n_seams):
        energy = normalize_energy(gradient_magnitude_energy(img))
        if energy.dim() == 2:
            energy = energy.unsqueeze(0)
        lat_energy = lattice.resample_to_lattice_space(energy, lattice_width)
        if lat_energy.dim() == 3:
            lat_energy = lat_energy.squeeze(0)
        lat_energy = normalize_energy(lat_energy)

        seam = dp_seam(lat_energy, direction='vertical')
        seam_interp = _interpolate_seam(seam, n_map, cyclic=is_cyclic)

        # Pixels on the seam: u_map within 0.5 of the seam position
        mask = (u_map - seam_interp).abs() < 0.6
        masks.append(mask.numpy())

        # Advance image (apply warp so next seam is on updated image)
        from src.carving import _compute_valid_mask, _warp_and_resample
        valid = _compute_valid_mask(lattice, u_map, n_map, H, W, 'cpu',
                                     lattice_width=lattice_width)
        shift = torch.where(u_map >= seam_interp,
                            torch.ones_like(u_map), torch.zeros_like(u_map))
        shift = torch.where(valid, shift, torch.zeros_like(shift))
        img = _warp_and_resample(img, lattice, u_map, n_map, shift)

    return masks


def t2np(t):
    if t.dim() == 3:
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return t.clamp(0, 1).cpu().numpy()


# ===========================================================================
# Case 1: L-shape with rectangular lattice
# ===========================================================================
def case_l_shape():
    H, W = 100, 120
    image = torch.zeros(3, H, W)
    image[:, :, :30] = 0.8        # vertical bar of L
    image[:, 70:, :80] = 0.6     # horizontal bar of L
    image += torch.rand(3, H, W) * 0.05

    lat = Lattice2D.rectangular(H, W)
    masks = seam_mask_from_energy(lat, image, W, n_seams=5)
    carved = carve_image_traditional(image, n_seams=5, direction='vertical')

    overlay = draw_seams(t2np(image), masks)
    return t2np(image), overlay, t2np(carved), "L-shape + rectangular lattice"


# ===========================================================================
# Case 2: Arch with arch-following lattice (existing test case)
# ===========================================================================
def case_arch():
    torch.manual_seed(42)
    H, W = 200, 300
    cy, cx = H - 20, W // 2
    outer_r, inner_r = 80, 55
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    image = torch.full((3, H, W), 0.15)
    arch_mask = (dist >= inner_r) & (dist <= outer_r) & (yy < cy)
    image[0][arch_mask] = 0.85
    image[1][arch_mask] = 0.65
    image[2][arch_mask] = 0.35
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    angles = torch.linspace(math.pi, 0, 60)
    mid_r = (inner_r + outer_r) / 2
    arch_x = cx + mid_r * torch.cos(angles)
    arch_y = cy + mid_r * torch.sin(angles)
    curve_pts = torch.stack([arch_x, arch_y], dim=1)
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=H / 2)

    masks = seam_mask_from_energy(lat, image, W, n_seams=5)
    carved = carve_image_lattice_guided(image, lat, n_seams=20, lattice_width=W)

    overlay = draw_seams(t2np(image), masks)
    return t2np(image), overlay, t2np(carved), "Arch + arch-following lattice"


# ===========================================================================
# Case 3: Striped image + sinusoidal lattice
# ===========================================================================
def case_stripes_curve():
    H, W = 160, 200
    # Vertical stripes
    x = torch.arange(W, dtype=torch.float32)
    stripe = (torch.sin(x * 2 * math.pi / 20) > 0).float()
    image = stripe.unsqueeze(0).unsqueeze(0).expand(3, H, W).clone()
    image += torch.rand(3, H, W) * 0.05
    image = image.clamp(0, 1)

    # Sinusoidal centerline
    t = torch.linspace(0, 2 * math.pi, 40)
    cx_curve = torch.linspace(10, 190, 40)
    cy_curve = H / 2 + 30 * torch.sin(t)
    control_pts = torch.stack([cx_curve, cy_curve], dim=1)
    lat = Lattice2D.from_curve_points(control_pts, n_lines=40, perp_extent=60.0)

    masks = seam_mask_from_energy(lat, image, W, n_seams=5)
    carved = carve_image_lattice_guided(image, lat, n_seams=10, lattice_width=W)

    # Also draw the centerline
    overlay = draw_seams(t2np(image), masks)
    # Draw centerline in blue
    cy_int = cy_curve.long().clamp(0, H - 1)
    cx_int = cx_curve.long().clamp(0, W - 1)
    for i in range(len(cx_int)):
        r, c = cy_int[i].item(), cx_int[i].item()
        overlay[max(0,r-1):r+2, max(0,c-1):c+2] = [0.0, 0.4, 1.0]

    return t2np(image), overlay, t2np(carved), "Stripes + sinusoidal lattice"


# ===========================================================================
# Plot everything
# ===========================================================================
cases = [case_l_shape, case_arch, case_stripes_curve]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, fn in enumerate(cases):
    orig, overlay, carved, title = fn()
    axes[row, 0].imshow(orig, aspect='auto')
    axes[row, 0].set_title(f"{title}\nOriginal", fontsize=9)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(overlay, aspect='auto')
    axes[row, 1].set_title("First 5 seams (red)\n← should follow lattice geometry", fontsize=9)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(carved, aspect='auto')
    axes[row, 2].set_title("After carving", fontsize=9)
    axes[row, 2].axis('off')

fig.suptitle(
    "Seam Placement Diagnostic — do the red seams follow the expected geometry?",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_seam_overlay_diagnostic.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
