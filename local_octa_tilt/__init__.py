"""
LocalOctaTilt - Python implementation of phenomenological model for local octahedral tilting
in lead halide perovskites.

This package simulates S(q) arising from local I4/mcm (P4/mbm) nanodomains in cubic perovskite
structures, as described in the publication:
"Dynamic nanodomains dictate macroscopic properties in lead halide perovskites"
"""

from .simulation import LocalSymmetrizedSimulation
from .io import import_structure_factors, load_md_data
from .slicing import two_d_slice, slice_cube
from .plotting import colormap_plot, isosurface_plot
from .transforms import transform_hkl_no_inv, create_matrix_from_top_right
from .matrix import Matrix

__version__ = "1.0.0"
__author__ = "Based on MATLAB code by dubajicmilos"

__all__ = [
    "LocalSymmetrizedSimulation",
    "import_structure_factors",
    "load_md_data",
    "two_d_slice",
    "slice_cube",
    "colormap_plot",
    "isosurface_plot",
    "transform_hkl_no_inv",
    "create_matrix_from_top_right",
    "Matrix",
]
