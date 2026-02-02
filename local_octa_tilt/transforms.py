"""
Coordinate transformation functions for reciprocal lattice vectors.
"""

import numpy as np
import pandas as pd
from typing import Union


def transform_hkl_no_inv(
    G: pd.DataFrame,
    M_sg: np.ndarray,
    M_rot: np.ndarray,
    hkl_range: float = 5.5
) -> np.ndarray:
    """
    Transform hkl indices using space group and rotation matrices.

    Parameters
    ----------
    G : pd.DataFrame
        Structure factors DataFrame with columns h, k, l, F
    M_sg : np.ndarray
        Space group transformation matrix (3x3)
    M_rot : np.ndarray
        Rotation matrix (3x3)
    hkl_range : float, optional
        Maximum absolute value for hkl indices (default: 5.5)

    Returns
    -------
    np.ndarray
        Transformed array with columns [h, k, l, F]
    """
    # Create reflection list with original indexation
    GM = np.column_stack([G['h'].values, G['k'].values, G['l'].values, G['F'].values])

    # Transform the hkl coordinates
    hkl_original = GM[:, :3]
    F1 = hkl_original @ M_sg @ M_rot

    # Create the transformed reflection list
    GMT = np.column_stack([F1, G['F'].values])

    # Remove all points that are outside the index_range
    mask = (
        (GMT[:, 0] <= hkl_range) & (GMT[:, 0] >= -hkl_range) &
        (GMT[:, 1] <= hkl_range) & (GMT[:, 1] >= -hkl_range) &
        (GMT[:, 2] <= hkl_range) & (GMT[:, 2] >= -hkl_range)
    )

    return GMT[mask]


def create_matrix_from_top_right(top_right_quadrant: np.ndarray) -> np.ndarray:
    """
    Create a full symmetric matrix from the top-right quadrant using symmetry operations.

    Parameters
    ----------
    top_right_quadrant : np.ndarray
        The top-right quadrant of the matrix

    Returns
    -------
    np.ndarray
        The full symmetric matrix
    """
    # Q3: Bottom Left (flip both up-down and left-right)
    Q3 = np.flipud(np.fliplr(top_right_quadrant))
    Q3 = Q3[:-1, :]  # Remove last row

    # Q4: Bottom Right (flip up-down only)
    Q4 = np.flipud(top_right_quadrant)
    Q4 = Q4[:-1, 1:]  # Remove last row and first column

    # Q2: Top Left (flip left-right only)
    Q2 = np.fliplr(top_right_quadrant)

    # Top Right (remove first column)
    TR = top_right_quadrant[:, 1:]

    # Combine quadrants
    bottom = np.hstack([Q3, Q4])
    top = np.hstack([Q2, TR])
    Tot = np.vstack([bottom, top])

    return Tot
