# Lattice Carving

A generalized implementation of seam carving algorithms for volumetric data.

## Overview

Based on "Generalized Fluid Carving with Fast Lattice-Guided Seam Computation" (Flynn et al., 2021), this project implements a lattice-guided approach to seam carving that works with:
- Non-rectangular boundaries and motion
- Volumetric data (fluids, smoke, fire, meshes, particles)
- VDB grid representations
- Custom lattice structures that follow the shape/motion of the data

The key innovation is using non-uniform lattices in "lattice index space" rather than rectangular world space, allowing seam carving operations on complex, curved regions while preserving shape silhouettes.

## Goals

- Implement lattice structures with configurable geometry
- Support lattice mapping functions (world space ↔ lattice index space)
- Implement greedy seam computation algorithm for fast carving
- Provide flexible energy function definitions for various data types
- Support seam pairs for local region carving without modifying global boundaries
- Handle cyclic lattices for closed-loop scenarios

## Status

Project initialization - setting up repository and gathering requirements.

## Installation

TBD

## Usage

TBD
