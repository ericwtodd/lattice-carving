# Lattice Structure: What We Got Wrong

## The Confusion

**Our current implementation:**
- For circular objects: Using RADIAL scanlines (lines from center outward)
- For rivers: Using HORIZONTAL scanlines across entire image

**This is wrong!** Here's why:

## What the Paper Actually Says

### Lattice Definition (Section 3.1)

A lattice is a sequence of **parallel or nearly-parallel planes** (in 2D: lines/curves).

**Key insight:** The scanlines define a COORDINATE SYSTEM that maps curved regions to a regular rectangular grid.

### For a Bagel/Ring (Figure 8)

The lattice should use **CONCENTRIC CIRCLES** as scanlines:
- Each scanline is a circle at radius r_n
- **u coordinate**: angle θ around the circle (0 to 2π)
- **n coordinate**: which circle (inner to outer radii)

**In lattice space:**
- Horizontal axis (u) = angle around circle
- Vertical axis (n) = radius
- A VERTICAL seam (constant u across all n) = a radial wedge from inner to outer
- A HORIZONTAL seam (constant n across all u) = removing an entire circle

**Seam pairs for bagel:**
- ROI (inner hole): seam at small n (inner circles) → shrinks hole
- Pair (outer background): seam at large n (outer circles) → expands background
- Net effect: hole smaller, bagel boundary unchanged

### For a River (Figure 9)

The lattice should use **CURVED scanlines that FOLLOW THE RIVER**:
- Each scanline runs parallel to the river's flow direction
- **u coordinate**: position along the river's length
- **n coordinate**: perpendicular distance from river centerline

**Key:** The lattice only covers the REGION OF INTEREST (river + surrounding area), NOT the entire image!

**In lattice space:**
- The river appears as a horizontal band in the middle
- Vertical seams remove strips along the river's length
- The lattice straightens out the curved river into a rectangular region

## What We Need to Fix

### 1. Implement Concentric Circle Lattice

For circular objects (bagel, ring):

```python
class Lattice2D:
    @classmethod
    def concentric_circles(cls, center, inner_radius, outer_radius, n_circles):
        """
        Create concentric circle lattice.

        Scanlines are circles at different radii.
        - u: angle around circle (0 to 2π)
        - n: which circle (inner to outer)
        """
        # Each origin is at the center
        # Tangent vectors point tangent to the circle (perpendicular to radius)
        # ...
```

### 2. Implement Curved Scanline Lattice

For rivers, roads, etc.:

```python
class Lattice2D:
    @classmethod
    def from_curve(cls, centerline_points, width, n_scanlines):
        """
        Create lattice following a curved path.

        Args:
            centerline_points: Points defining the center of the curved region
            width: How far perpendicular to extend from centerline
            n_scanlines: Number of parallel scanlines
        """
        # Scanlines parallel to the curve
        # ...
```

### 3. Lattice Should Cover ROI Only

The lattice doesn't need to cover the entire image. It should cover:
- The region being carved (ROI)
- Plus surrounding context for smooth transitions

Outside the lattice region, pixels are unchanged.

## Next Steps

1. **Test current radial lattice:** Maybe it's correct but we're visualizing it wrong?
   - Draw what a vertical seam in lattice space actually removes in world space
   - Verify if this makes sense for the bagel case

2. **Implement concentric circle lattice:** If radial is wrong, implement the correct one

3. **Implement curved scanline lattice:** For the river case

4. **Add ROI masking:** Lattice only affects a local region, not entire image
