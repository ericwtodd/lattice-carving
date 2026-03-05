"""
Animate seam carving in semi-real-time.

Each animation is produced in two variants:
  <name>.gif         — clean (no seam overlay)
  <name>_overlay.gif — with seam pair paths drawn on each frame before carving

Animations:
  animation_traditional            — synthetic gradient+edge, traditional SC
  animation_energy_seam            — image || energy map side-by-side
  animation_bagel_seam_pairs       — synthetic seeded bagel ring shrink/grow
  animation_street_traditional     — street.jpg traditional SC
  animation_real_bagel_pairs       — bagel.jpg ring shrink/grow
  animation_real_bagel_double      — bagel_double.jpg left bagel shrink/grow
  animation_real_river_pairs       — river.jpg river narrowing/widening

Run:
    conda run -n lattice-carving python examples/animate_carving.py

Output goes to output/ directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from src.lattice import Lattice2D
from src.carving import (
    _precompute_forward_mapping,
    _compute_valid_mask,
    _interpolate_seam,
    _shift_to_warp_grid,
    _sample_src_map,
    _src_map_to_image,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import dp_seam, dp_seam_cyclic, dp_seam_windowed, remove_seam

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) or (H,W) tensor to HxWxC uint8 numpy array."""
    if t.dim() == 3:
        arr = t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    else:
        arr = t.unsqueeze(2).expand(-1, -1, 3).clamp(0, 1).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def frames_to_gif(frames: list, path: Path, duration_ms: int = 50, loop: int = 0):
    """Save a list of HxWxC uint8 numpy arrays as an animated GIF."""
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        loop=loop,
        duration=duration_ms,
        optimize=False,
    )
    print(f"  Saved: {path}  ({len(frames)} frames @ {duration_ms}ms)")


def load_image(path: Path) -> torch.Tensor:
    """Load image file as (C, H, W) float32 tensor in [0, 1]."""
    pil_img = PILImage.open(path).convert('RGB')
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).contiguous()


def _draw_seam_overlay(frame_np: np.ndarray, lat: Lattice2D,
                       roi_seam: torch.Tensor, pair_seam: torch.Tensor,
                       thickness: int = 2) -> np.ndarray:
    """Paint seam paths onto a HxWxC uint8 numpy array.

    ROI seam  → cyan    (0, 220, 220)
    Pair seam → magenta (220, 0, 220)
    """
    H, W = frame_np.shape[:2]
    n_lines = lat.n_lines
    n_vals = torch.arange(n_lines, dtype=torch.float32)

    for seam, color in [(roi_seam, [0, 220, 220]), (pair_seam, [220, 0, 220])]:
        pts_lat = torch.stack([seam.float(), n_vals], dim=1)
        world = lat.inverse_mapping(pts_lat).detach().cpu().numpy()
        xs = np.round(world[:, 0]).astype(int)
        ys = np.round(world[:, 1]).astype(int)
        for dr in range(-thickness, thickness + 1):
            for dc in range(-thickness, thickness + 1):
                if dr*dr + dc*dc > thickness*thickness:
                    continue
                ny, nx = ys + dr, xs + dc
                valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                frame_np[ny[valid], nx[valid]] = color

    return frame_np


def _seam_pair_step(image, lat, u_map, n_map, valid_mask, x_grid, y_grid,
                    src_map, lattice_w, roi_range, pair_range,
                    roi_sign=1.0, cyclic=True):
    """One seam-pair iteration. Returns (new_src_map, roi_seam, pair_seam)."""
    H, W = u_map.shape
    current_image = _src_map_to_image(image, src_map, H, W)
    energy = gradient_magnitude_energy(current_image)
    energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
    lattice_energy = lat.resample_to_lattice_space(energy_3d, lattice_w)
    if lattice_energy.dim() == 3:
        lattice_energy = lattice_energy.squeeze(0)
    lattice_energy = normalize_energy(lattice_energy)

    if cyclic:
        roi_seam = dp_seam_cyclic(lattice_energy, roi_range)
        pair_seam = dp_seam_cyclic(lattice_energy, pair_range)
    else:
        roi_seam = dp_seam_windowed(lattice_energy, roi_range)
        pair_seam = dp_seam_windowed(lattice_energy, pair_range)

    roi_seam_interp = _interpolate_seam(roi_seam, n_map, cyclic=cyclic)
    pair_seam_interp = _interpolate_seam(pair_seam, n_map, cyclic=cyclic)

    combined_shift = torch.zeros_like(u_map)
    combined_shift = combined_shift + torch.where(
        u_map >= roi_seam_interp, torch.full_like(u_map, roi_sign), torch.zeros_like(u_map))
    combined_shift = combined_shift + torch.where(
        u_map > pair_seam_interp, torch.full_like(u_map, -roi_sign), torch.zeros_like(u_map))
    combined_shift = torch.where(valid_mask, combined_shift, torch.zeros_like(combined_shift))

    grid = _shift_to_warp_grid(lat, u_map, n_map, combined_shift, H, W)
    src_map = _sample_src_map(src_map, grid, valid_mask, x_grid, y_grid)
    return src_map, roi_seam, pair_seam


def _run_seam_pairs_animation(image, lat, lattice_w, roi_range, pair_range,
                               n_seams, cyclic, roi_sign,
                               valid_mask, u_map, n_map, x_grid, y_grid,
                               hold_frames=1):
    """Run one phase of seam pair carving.

    Returns:
        clean_frames:   list of uint8 arrays — result only, `hold_frames` per step
        overlay_frames: list of uint8 arrays — seam overlay frame then `hold_frames`
                        result frames per step (so the seam flashes before each carved state)
    """
    C, H, W = image.shape
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)
    src_map = torch.stack([x_grid, y_grid], dim=0)
    clean_frames, overlay_frames = [], []

    for _ in range(n_seams):
        src_map, roi_seam, pair_seam = _seam_pair_step(
            image, lat, u_map, n_map, valid_mask, x_grid, y_grid,
            src_map, lattice_w, roi_range, pair_range,
            roi_sign=roi_sign, cyclic=cyclic,
        )
        result = _src_map_to_image(image, src_map, H, W)
        result = torch.where(valid_mask_3d, result, image)
        result_np = tensor_to_uint8(result)

        for _ in range(hold_frames):
            clean_frames.append(result_np)

        overlay_np = _draw_seam_overlay(result_np.copy(), lat, roi_seam, pair_seam)
        overlay_frames.append(overlay_np)          # seam flash
        for _ in range(hold_frames):
            overlay_frames.append(result_np)       # then carved result

    return clean_frames, overlay_frames


def _save_pair_animation(stem: str, phases: list, orig_np: np.ndarray,
                          duration_ms: int = 50, hold_end: int = 8):
    """Assemble and save clean + overlay GIFs from a list of phase frame-pairs.

    Args:
        stem: output filename stem (e.g. "animation_bagel_seam_pairs")
        phases: list of (clean_frames, overlay_frames) tuples, one per phase
        orig_np: original image frame to bookend the animation
        duration_ms: ms per frame
        hold_end: how many times to repeat the last frame of each phase
    """
    intro = [orig_np] * 5
    clean_all = list(intro)
    overlay_all = list(intro)

    for clean_phase, overlay_phase in phases:
        clean_all.extend(clean_phase)
        overlay_all.extend(overlay_phase)
        # Hold at end of phase
        for _ in range(hold_end):
            clean_all.append(clean_all[-1])
            overlay_all.append(overlay_all[-1])

    # Return to original
    clean_all.extend([orig_np] * 5)
    overlay_all.extend([orig_np] * 5)

    frames_to_gif(clean_all,   OUTPUT_DIR / f"{stem}.gif",         duration_ms)
    frames_to_gif(overlay_all, OUTPUT_DIR / f"{stem}_overlay.gif", duration_ms)


# ---------------------------------------------------------------------------
# Animation: Traditional seam carving
# ---------------------------------------------------------------------------

def make_gradient_edge_image(H: int = 120, W: int = 180) -> torch.Tensor:
    image = torch.zeros(3, H, W)
    grad = torch.linspace(0.1, 0.45, W).unsqueeze(0).expand(H, W)
    image[0] = grad
    image[1] = grad * 0.85
    image[2] = grad * 0.6
    image[:, :, 80:92] = torch.tensor([0.9, 0.85, 0.5]).view(3, 1, 1)
    image += (torch.rand(3, H, W) - 0.5) * 0.04
    return image.clamp(0, 1)


def _animate_traditional(image: torch.Tensor, n_seams: int, label: str,
                          stem: str, duration_ms: int = 40):
    """Traditional seam carving animation — clean and overlay (red seam flash) variants."""
    print(f"\n--- Animation: {label} ---")
    C, H, W = image.shape
    carved = image.clone()

    intro = [tensor_to_uint8(image)] * 5
    clean_frames = list(intro)
    overlay_frames = list(intro)

    for _ in range(n_seams):
        current_W = carved.shape[2]
        energy = normalize_energy(gradient_magnitude_energy(carved))
        seam = dp_seam(energy, direction='vertical')
        pad = W - current_W

        # Overlay: red seam highlight
        highlighted = carved.clone()
        row_idx = torch.arange(H, device=carved.device)
        highlighted[:, row_idx, seam] = torch.tensor([1.0, 0.15, 0.15],
                                                      device=carved.device).view(3, 1)
        overlay_frames.append(tensor_to_uint8(F.pad(highlighted, (0, pad), value=0.0)))

        carved = remove_seam(carved, seam, direction='vertical')
        new_pad = W - carved.shape[2]
        result_np = tensor_to_uint8(F.pad(carved, (0, new_pad), value=0.0))
        clean_frames.append(result_np)
        overlay_frames.append(result_np)

    for _ in range(8):
        clean_frames.append(clean_frames[-1])
        overlay_frames.append(overlay_frames[-1])

    frames_to_gif(clean_frames,   OUTPUT_DIR / f"{stem}.gif",         duration_ms)
    frames_to_gif(overlay_frames, OUTPUT_DIR / f"{stem}_overlay.gif", duration_ms)


def animate_traditional_carving(n_seams: int = 35):
    torch.manual_seed(42)
    _animate_traditional(make_gradient_edge_image(), n_seams,
                         "Traditional Seam Carving (synthetic)",
                         "animation_traditional")


# ---------------------------------------------------------------------------
# Animation: Energy map + seam overlay (synthetic only)
# ---------------------------------------------------------------------------

def animate_energy_seam(n_seams: int = 25):
    print("\n--- Animation: Energy + Seam Overlay ---")
    torch.manual_seed(42)
    H, W = 120, 180
    image = make_gradient_edge_image(H, W)
    carved = image.clone()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    def render_frame(img, energy, seam):
        current_W = img.shape[2]
        pad = W - current_W
        img_np = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        if pad > 0:
            img_np = np.pad(img_np, ((0, 0), (0, pad), (0, 0)))
        seam_img = img_np.copy()
        for row in range(img.shape[1]):
            col = seam[row].item()
            if 0 <= col < img_np.shape[1]:
                seam_img[row, max(0, col-1):col+2] = [1.0, 0.1, 0.1]
        energy_np = energy.clamp(0, 1).cpu().numpy()
        if pad > 0:
            energy_np = np.pad(energy_np, ((0, 0), (0, pad)))
        axes[0].clear()
        axes[0].imshow(seam_img, aspect='auto')
        axes[0].set_title("Image + seam", fontsize=9)
        axes[0].axis('off')
        axes[1].clear()
        axes[1].imshow(energy_np, cmap='hot', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title("Energy (normalized)", fontsize=9)
        axes[1].axis('off')
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return buf[:, :, :3].copy()

    frames = []
    energy = normalize_energy(gradient_magnitude_energy(carved))
    seam = dp_seam(energy, direction='vertical')
    for _ in range(4):
        frames.append(render_frame(carved, energy, seam))
    for _ in range(n_seams):
        energy = normalize_energy(gradient_magnitude_energy(carved))
        seam = dp_seam(energy, direction='vertical')
        frames.append(render_frame(carved, energy, seam))
        carved = remove_seam(carved, seam, direction='vertical')
    energy = normalize_energy(gradient_magnitude_energy(carved))
    for _ in range(6):
        frames.append(render_frame(carved, energy, torch.zeros(carved.shape[1], dtype=torch.long)))

    plt.close(fig)
    frames_to_gif(frames, OUTPUT_DIR / "animation_energy_seam.gif", duration_ms=100)


# ---------------------------------------------------------------------------
# Animation: Synthetic seeded bagel
# ---------------------------------------------------------------------------

def make_seeded_bagel(H=200, W=200, cx=100.0, cy=100.0,
                      inner_r=30.0, outer_r=80.0, seed=7):
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = torch.full((3, H, W), 0.15)
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    image[0][ring_mask] = 0.85
    image[1][ring_mask] = 0.70
    image[2][ring_mask] = 0.40
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    rng = torch.Generator()
    rng.manual_seed(seed)
    seed_color = torch.tensor([0.96, 0.88, 0.68])
    for _ in range(60):
        angle = torch.rand(1, generator=rng).item() * 2 * np.pi
        r = inner_r + (outer_r - inner_r) * torch.rand(1, generator=rng).item()
        sx, sy = cx + r * np.cos(angle), cy + r * np.sin(angle)
        sa_ang = torch.rand(1, generator=rng).item() * np.pi
        ca, sa = np.cos(sa_ang), np.sin(sa_ang)
        for di in range(-3, 4):
            for dj in range(-1, 2):
                px = int(round(sx + di * ca - dj * sa))
                py = int(round(sy + di * sa + dj * ca))
                if 0 <= px < W and 0 <= py < H:
                    if inner_r <= np.sqrt((px-cx)**2 + (py-cy)**2) <= outer_r:
                        image[:, py, px] = seed_color
    return image.clamp(0, 1)


def animate_bagel_seam_pairs(n_seams: int = 15, hold_frames: int = 1):
    print("\n--- Animation: Bagel Seam Pairs ---")
    torch.manual_seed(42)
    H, W = 200, 200
    cx, cy, inner_r, outer_r = 100.0, 100.0, 30.0, 80.0
    image = make_seeded_bagel(H, W, cx, cy, inner_r, outer_r)

    theta = torch.linspace(0, 2 * np.pi, 80)
    mid_r = (inner_r + outer_r) / 2
    curve_pts = torch.stack([cx + mid_r * torch.cos(theta),
                              cy + mid_r * torch.sin(theta)], dim=1)
    perp = (outer_r - inner_r) / 2 + 15    # = 40
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=256, perp_extent=perp, cyclic=True)
    lattice_w = int(2 * perp)              # = 80
    center_u = int(perp)                   # = 40

    shrink_roi  = (center_u - 5,  center_u + 5)   # ring body, narrow
    shrink_pair = (lattice_w - 11, lattice_w)       # outer background

    # Grow: place pair seam at INNER ring edge so the full ring body (u > inner_edge)
    # gets the -1 shift. inner_edge ≈ center_u - ring_half = 40 - 25 = 15
    inner_ring_edge = center_u - int((outer_r - inner_r) / 2)  # ≈ 15
    grow_roi  = (lattice_w - 11, lattice_w)
    grow_pair = (max(2, inner_ring_edge - 8), inner_ring_edge + 8)  # tight band at inner edge

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=True, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_bagel_seam_pairs", phases,
                          tensor_to_uint8(image), duration_ms=50)


# ---------------------------------------------------------------------------
# Real image animations
# ---------------------------------------------------------------------------

def animate_street_traditional(n_seams: int = 80):
    path = ASSETS_DIR / "street.jpg"
    if not path.exists():
        print(f"  SKIPPED: {path} not found"); return
    image = load_image(path)
    print(f"  Street: {image.shape[2]}x{image.shape[1]}")
    _animate_traditional(image, n_seams,
                         "Street — Traditional Seam Carving",
                         "animation_street_traditional", duration_ms=40)


def animate_real_bagel_pairs(n_seams: int = 20, hold_frames: int = 1):
    path = ASSETS_DIR / "bagel.jpg"
    if not path.exists():
        print(f"  SKIPPED: {path} not found"); return
    print("\n--- Animation: Real Bagel Seam Pairs ---")
    image = load_image(path)
    C, H, W = image.shape
    print(f"  Bagel: {W}x{H}")

    cx, cy = W / 2.0, H / 2.0
    inner_r, outer_r = 28.0, 92.0
    mid_r = (inner_r + outer_r) / 2
    ring_half = (outer_r - inner_r) / 2   # 32

    theta = torch.linspace(0, 2 * np.pi, 200)
    curve_pts = torch.stack([cx + mid_r * torch.cos(theta),
                              cy + mid_r * torch.sin(theta)], dim=1)
    perp = ring_half + 55                  # = 87
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=1024, perp_extent=perp, cyclic=True)
    lattice_w = int(2 * perp)             # = 174
    center_u = int(perp)                  # = 87

    shrink_roi  = (center_u - 15, center_u + 15)                  # (72, 102)
    shrink_pair = (int(center_u + ring_half + 8), lattice_w)       # (127, 174)

    # Grow: pair seam at inner ring edge ≈ center_u - ring_half = 87 - 32 = 55
    inner_ring_edge = int(center_u - ring_half)                    # ≈ 55
    grow_roi  = shrink_pair                                         # (127, 174)
    grow_pair = (max(2, inner_ring_edge - 8), inner_ring_edge + 8) # tight band at inner edge

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=True, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_real_bagel_pairs", phases,
                          tensor_to_uint8(image), duration_ms=50)


def animate_real_bagel_double(n_seams: int = 15, hold_frames: int = 1):
    path = ASSETS_DIR / "bagel_double.jpg"
    if not path.exists():
        print(f"  SKIPPED: {path} not found"); return
    print("\n--- Animation: Double Bagel Seam Pairs ---")
    image = load_image(path)
    C, H, W = image.shape
    print(f"  Double bagel: {W}x{H}")

    left_cx = W * 0.27
    left_cy = H * 0.50
    left_radius = H * 0.30
    perp = left_radius * 0.6

    theta = torch.linspace(0, 2 * np.pi, 100)
    curve_pts = torch.stack([left_cx + left_radius * torch.cos(theta),
                              left_cy + left_radius * torch.sin(theta)], dim=1)
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=256, perp_extent=perp, cyclic=True)
    lattice_w = int(2 * perp)
    center_u = int(perp)

    shrink_roi  = (center_u - 5,  center_u + 5)    # narrow ring body
    shrink_pair = (lattice_w - 11, lattice_w)        # outer background

    # Grow: pair seam at inner ring edge (not certain for real image, ~u=15–25 range)
    grow_roi  = shrink_pair
    grow_pair = (2, 25)                              # force seam to inner ring boundary

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=True, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_real_bagel_double", phases,
                          tensor_to_uint8(image), duration_ms=50)


def animate_real_river(n_seams: int = 12, hold_frames: int = 1):
    import json
    path = ASSETS_DIR / "river.jpg"
    if not path.exists():
        print(f"  SKIPPED: {path} not found"); return
    print("\n--- Animation: River Seam Pairs ---")
    image = load_image(path)
    C, H, W = image.shape
    print(f"  River: {W}x{H}")

    cached_path = OUTPUT_DIR / "river_centerline.json"
    if cached_path.exists():
        with open(cached_path) as f:
            control_pts = torch.tensor(json.load(f), dtype=torch.float32)
        print("  Centerline: from cache")
    else:
        control_pts = torch.tensor([
            [30, 310], [70, 275], [120, 220], [155, 175], [155, 140],
            [140, 110], [165, 85], [215, 75], [265, 105], [310, 145],
            [355, 175], [395, 155], [440, 110], [490, 65], [540, 55],
            [590, 65], [640, 95], [670, 120],
        ], dtype=torch.float32)
        print("  Centerline: hardcoded")

    perp = 50
    lat = Lattice2D.from_curve_points(control_pts, n_lines=2048, perp_extent=perp)
    print("  Smoothing lattice...")
    lat.smooth(max_iterations=500)

    lattice_w = int(2 * perp)   # 100
    center_u = int(perp)         # 50
    river_half = 20

    shrink_roi  = (center_u - river_half, center_u + river_half)   # (30, 70)
    shrink_pair = (center_u + river_half + 3, lattice_w)            # (73, 100)

    # Grow: pair seam at inner river edge ≈ center_u - river_half = 30
    # so the full river body (u > 30) gets -1 shift
    grow_roi    = shrink_pair
    grow_pair   = (max(2, center_u - river_half - 8), center_u - river_half + 8)  # (22, 38)

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=False, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_real_river_pairs", phases,
                          tensor_to_uint8(image), duration_ms=80)


# ---------------------------------------------------------------------------
# Synthetic cookie batch
# ---------------------------------------------------------------------------

def make_cookie_batch(H: int = 300, W: int = 400, seed: int = 42) -> torch.Tensor:
    """Baking sheet with 6 chocolate chip cookies in a 2x3 grid."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Parchment paper background
    image = torch.full((3, H, W), 0.92)
    image[0] += (torch.rand(3, H, W, generator=rng)[0] - 0.5) * 0.04
    image[1] += (torch.rand(3, H, W, generator=rng)[0] - 0.5) * 0.03
    image[2] += (torch.rand(3, H, W, generator=rng)[0] - 0.5) * 0.02

    # Cookie centers in a 2x3 grid
    cookie_r = 45
    centers = []
    for row in range(2):
        for col in range(3):
            cx = int(W * (col + 1) / 4)
            cy = int(H * (row + 1) / 3)
            centers.append((cx, cy))

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    for cx, cy in centers:
        dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
        cookie_mask = dist <= cookie_r

        # Cookie base: golden-brown
        image[0][cookie_mask] = 0.82
        image[1][cookie_mask] = 0.62
        image[2][cookie_mask] = 0.30
        # Edge slightly darker (baked rim)
        rim_mask = (dist > cookie_r * 0.80) & (dist <= cookie_r)
        image[0][rim_mask] = 0.70
        image[1][rim_mask] = 0.50
        image[2][rim_mask] = 0.22
        # Subtle noise texture
        noise = (torch.rand(3, H, W, generator=rng) - 0.5) * 0.06
        for c in range(3):
            image[c][cookie_mask] = (image[c][cookie_mask] + noise[c][cookie_mask]).clamp(0, 1)

        # Chocolate chips: ~8 per cookie
        n_chips = 8
        for _ in range(n_chips):
            r_chip = cookie_r * 0.75 * torch.rand(1, generator=rng).item()
            a_chip = torch.rand(1, generator=rng).item() * 2 * np.pi
            sx = cx + r_chip * np.cos(a_chip)
            sy = cy + r_chip * np.sin(a_chip)
            chip_angle = torch.rand(1, generator=rng).item() * np.pi
            ca, sa = np.cos(chip_angle), np.sin(chip_angle)
            for di in range(-4, 5):
                for dj in range(-2, 3):
                    if di*di/16 + dj*dj/4 > 1.0:
                        continue
                    px = int(round(sx + di * ca - dj * sa))
                    py = int(round(sy + di * sa + dj * ca))
                    if 0 <= px < W and 0 <= py < H:
                        rd = np.sqrt((px - cx)**2 + (py - cy)**2)
                        if rd <= cookie_r * 0.88:
                            image[:, py, px] = torch.tensor([0.22, 0.14, 0.08])

    return image.clamp(0, 1)


def animate_cookie_batch(n_seams: int = 18, hold_frames: int = 1):
    """Animate shrink/grow on one cookie in a synthetic batch."""
    print("\n--- Animation: Cookie Batch Seam Pairs ---")
    torch.manual_seed(42)
    H, W = 300, 400
    cookie_r = 45

    image = make_cookie_batch(H, W)

    # Target the top-left cookie (first in 2x3 grid)
    cx = int(W * 1 / 4)
    cy = int(H * 1 / 3)

    theta = torch.linspace(0, 2 * np.pi, 160)
    # Lattice centered on cookie edge (r = cookie_r), scanlines perpendicular to ring
    curve_pts = torch.stack([cx + cookie_r * torch.cos(theta),
                              cy + cookie_r * torch.sin(theta)], dim=1)

    # perp_extent covers from outside the cookie inward through the body
    perp = cookie_r + 20          # extends 20px into background + full cookie body
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=512, perp_extent=perp, cyclic=True)
    lattice_w = int(2 * perp)     # scanline goes from -perp (inner) to +perp (outer)
    center_u = int(perp)          # u=center_u is on the cookie edge circle

    # Cookie body is at u < center_u (inward from edge), background at u > center_u
    # Shrink: ROI seam inside cookie body, pair seam in background
    body_band = 12
    shrink_roi  = (center_u - body_band, center_u - 2)    # cookie body near edge
    shrink_pair = (center_u + 5, lattice_w - 2)            # background

    # Grow: pair seam at the cookie center (u=0 side = deepest inside cookie)
    grow_roi    = shrink_pair
    grow_pair   = (2, body_band)                           # deep inside cookie body

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=True, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_cookie_batch", phases,
                          tensor_to_uint8(image), duration_ms=60)


# ---------------------------------------------------------------------------
# Synthetic river and arch
# ---------------------------------------------------------------------------

def animate_synthetic_river(n_seams: int = 15, hold_frames: int = 1):
    """Animate seam pairs on a synthetic sinusoidal river."""
    print("\n--- Animation: Synthetic River Seam Pairs ---")
    torch.manual_seed(42)
    H, W = 400, 600
    band_width = 60

    # Build river image via vectorized distance-to-curve
    x_vals = torch.linspace(0, W, 200, dtype=torch.float32)
    y_vals = H / 2 + 60 * torch.sin(2 * np.pi * x_vals / W)
    curve_pts = torch.stack([x_vals, y_vals], dim=1)

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)
    # Vectorized min distance: (H*W, 1) vs (1, N_curve)
    px = xx.reshape(-1, 1) - curve_pts[:, 0].unsqueeze(0)  # (H*W, N)
    py = yy.reshape(-1, 1) - curve_pts[:, 1].unsqueeze(0)
    min_dist = torch.sqrt(px**2 + py**2).min(dim=1).values.reshape(H, W)

    river_mask = min_dist < band_width
    image = torch.zeros(3, H, W)
    image[0] = 0.25; image[1] = 0.50; image[2] = 0.20
    image[0][river_mask] = 0.15
    image[1][river_mask] = 0.35
    image[2][river_mask] = 0.75
    image += (torch.rand(3, H, W) - 0.5) * 0.06
    image = image.clamp(0, 1)

    perp = band_width + 40        # = 100
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=1024, perp_extent=perp)
    lattice_w = int(2 * perp)     # = 200
    center_u = int(perp)          # = 100

    shrink_roi  = (center_u - band_width, center_u + band_width)  # (40, 160)
    shrink_pair = (center_u + band_width + 5, lattice_w)           # (165, 200)

    # Grow: pair seam at inner river edge = center_u - band_width = 40
    inner_edge = center_u - band_width
    grow_roi    = shrink_pair
    grow_pair   = (max(2, inner_edge - 8), inner_edge + 8)         # (32, 48)

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=False, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_synthetic_river", phases,
                          tensor_to_uint8(image), duration_ms=60)


def animate_synthetic_arch(n_seams: int = 12, hold_frames: int = 1):
    """Animate seam pairs on a synthetic semicircular arch."""
    print("\n--- Animation: Synthetic Arch Seam Pairs ---")
    torch.manual_seed(42)
    H, W = 200, 300
    cy, cx = H - 20, W // 2
    outer_r, inner_r = 80, 55

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    image = torch.full((3, H, W), 0.15)
    arch_mask = (dist >= inner_r) & (dist <= outer_r) & (yy < cy)
    image[0][arch_mask] = 0.85
    image[1][arch_mask] = 0.65
    image[2][arch_mask] = 0.35
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    angles = torch.linspace(np.pi, 0, 80)
    mid_r = (inner_r + outer_r) / 2
    curve_pts = torch.stack([cx + mid_r * torch.cos(angles),
                              cy - mid_r * torch.sin(angles)], dim=1)

    ring_half = (outer_r - inner_r) / 2   # = 12.5
    perp = ring_half + 20                  # = 32.5
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=perp)
    lattice_w = int(2 * perp)             # = 65
    center_u = int(perp)                  # = 32

    shrink_roi  = (center_u - 8, center_u + 8)   # arch body
    shrink_pair = (lattice_w - 15, lattice_w)      # outer background

    # Grow: pair seam at inner arch edge ≈ center_u - ring_half ≈ 19
    inner_edge = int(center_u - ring_half)
    grow_roi    = shrink_pair
    grow_pair   = (max(2, inner_edge - 6), inner_edge + 6)

    device = image.device
    u_map, n_map = _precompute_forward_mapping(lat, H, W, device)
    valid_mask = _compute_valid_mask(lat, u_map, n_map, H, W, device, lattice_width=lattice_w)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    phases = []
    for phase, roi, pair in [("Shrink", shrink_roi, shrink_pair),
                               ("Grow",   grow_roi,   grow_pair)]:
        print(f"  {phase}...")
        phases.append(_run_seam_pairs_animation(
            image, lat, lattice_w, roi, pair,
            n_seams=n_seams, cyclic=False, roi_sign=1.0,
            valid_mask=valid_mask, u_map=u_map, n_map=n_map,
            x_grid=x_grid, y_grid=y_grid, hold_frames=hold_frames,
        ))

    _save_pair_animation("animation_synthetic_arch", phases,
                          tensor_to_uint8(image), duration_ms=60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    animate_traditional_carving(n_seams=35)
    animate_energy_seam(n_seams=25)
    animate_bagel_seam_pairs(n_seams=15)
    animate_cookie_batch(n_seams=18)
    animate_synthetic_river(n_seams=15)
    animate_synthetic_arch(n_seams=12)
    animate_street_traditional(n_seams=80)
    animate_real_bagel_pairs(n_seams=20)
    animate_real_bagel_double(n_seams=15)
    animate_real_river(n_seams=12)

    print(f"\nAll animations saved to {OUTPUT_DIR}/")
