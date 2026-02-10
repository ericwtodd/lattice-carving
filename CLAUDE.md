# Claude Code Instructions

## Project Context

Research/implementation project for generalized lattice-guided seam carving,
based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation"
(Flynn et al., 2021). The paper PDF is at `generalized-fluid-carving.pdf`.

## Coding Preferences

- Python with PyTorch (GPU-accelerated tensor ops)
- Vectorized operations over Python loops wherever possible
- Conda environment: `lattice-carving`
- Run commands with: `conda run -n lattice-carving ...`

## Testing

- pytest test suite in `tests/test_lattice.py`
- Run with: `conda run -n lattice-carving python -m pytest tests/ -v`
- Tests should verify algorithmic correctness, not just shapes
- Use deterministic seeds (`torch.manual_seed`) for reproducibility

## Documentation

- Keep DEVLOG.md updated with progress, decisions, and problems encountered
- Document algorithmic choices and their rationale
- Include references to the background paper
- Commit often with descriptive messages

## Architecture

- `src/lattice.py` — Lattice2D class: construction, forward/inverse mapping, resampling
- `src/energy.py` — Energy functions (gradient magnitude, forward energy)
- `src/seam.py` — Greedy seam computation, seam removal
- `src/carving.py` — High-level carving orchestration
- `examples/` — Visual test scripts (not automated)
- `tests/` — pytest suite

## Key Algorithmic Notes

- Current carving uses the "naive" approach (resample V→L, carve, resample L→V).
  This causes blurring from double interpolation. Need to switch to the paper's
  "carving the mapping" approach (Section 3.3) which only resamples once.
- Greedy seam is sensitive to first-row energy (border artifacts from Sobel padding).
  Multi-greedy helps by trying multiple starting points.
- Circular lattice forward_mapping needs tangent-projection penalty to disambiguate
  between a radial scanline and its 180° opposite.
