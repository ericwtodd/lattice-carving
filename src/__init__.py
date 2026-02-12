"""
Lattice-guided seam carving implementation.

Based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
by Flynn et al., 2021.
"""

__version__ = "0.1.0"

from .lattice import Lattice2D
from .energy import gradient_magnitude_energy, normalize_energy, forward_energy
from .seam import (greedy_seam, greedy_seam_windowed, multi_greedy_seam, remove_seam,
                    dp_seam, dp_seam_windowed, dp_seam_cyclic)
from .carving import (
    carve_image_traditional,
    carve_image_lattice_guided,
    carve_seam_pairs,
    carve_with_comparison,
)

__all__ = [
    'Lattice2D',
    'gradient_magnitude_energy',
    'normalize_energy',
    'forward_energy',
    'greedy_seam',
    'greedy_seam_windowed',
    'multi_greedy_seam',
    'remove_seam',
    'dp_seam',
    'dp_seam_windowed',
    'dp_seam_cyclic',
    'carve_image_traditional',
    'carve_image_lattice_guided',
    'carve_seam_pairs',
    'carve_with_comparison',
]
