"""
Functions for extracting 2D slices from 3D reciprocal space data.
"""

import numpy as np
from typing import Tuple, Optional, Union, List


def two_d_slice(
    plane: str,
    S: np.ndarray,
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    log_mode: str = 'lin',
    plot_mode: str = 'off'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2D slice from 3D reciprocal space data.

    Parameters
    ----------
    plane : str
        Plane specification, e.g., 'HK1.5', 'H0.5L', '1.5KL'
    S : np.ndarray
        3D intensity array
    H, K, L : np.ndarray
        1D coordinate arrays
    log_mode : str, optional
        'log' for logarithmic scale, 'lin' for linear (default: 'lin')
    plot_mode : str, optional
        'on' to display plot, 'off' to suppress (default: 'off')

    Returns
    -------
    tuple
        (X, Y, Z) - Coordinate arrays and intensity slice
    """
    tol = 0.001

    if plane.startswith('HK'):
        # HK plane at constant L
        L_slice = float(plane[2:])
        indices = np.where(np.abs(L - L_slice) < tol)[0]
        if len(indices) == 0:
            raise ValueError(f"No slice found at L={L_slice}")
        idx = indices[0]
        Slice = S[:, idx, :]
        Slice1 = Slice.reshape(len(K), len(H))
        X, Y, Z = K, H, Slice1

    elif plane.startswith('H') and not plane.startswith('HK'):
        # H*L plane at constant K (e.g., 'H0.5L')
        K_slice = float(plane[1:-1])
        indices = np.where(np.abs(K - K_slice) < tol)[0]
        if len(indices) == 0:
            raise ValueError(f"No slice found at K={K_slice}")
        idx = indices[0]
        Slice = S[idx, :, :]
        Slice1 = Slice.reshape(len(L), len(H))
        X, Y, Z = L, H, Slice1

    else:
        # *KL plane at constant H (e.g., '1.5KL')
        H_slice = float(plane[:-2])
        indices = np.where(np.abs(H - H_slice) < tol)[0]
        if len(indices) == 0:
            raise ValueError(f"No slice found at H={H_slice}")
        idx = indices[0]
        Slice1 = S[:, :, idx]
        X, Y, Z = K, L, Slice1

    if log_mode == 'log':
        Z = np.log(np.maximum(Z, 1e-10))  # Avoid log(0)

    if plot_mode == 'on':
        from .plotting import colormap_plot
        colormap_plot(X, Y, Z.T)

    return X, Y, Z


def slice_cube(
    data: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: Union[List[float], np.ndarray],
    dy: Union[List[float], np.ndarray],
    dz: Union[List[float], np.ndarray],
    bin_axis: Optional[Union[List[int], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and optionally integrate a slice from a 3D data cube.

    Parameters
    ----------
    data : np.ndarray
        3D intensity array
    vx, vy, vz : np.ndarray
        1D coordinate arrays for each axis
    dx, dy, dz : list or array
        Range specification for each axis. If 2 elements [min, max],
        select that range. Otherwise, use full range.
    bin_axis : list or array, optional
        Axis to integrate over, specified as [1,0,0], [0,1,0], or [0,0,1]

    Returns
    -------
    tuple
        (datacut, xa, ya, za) - Sliced data and coordinate arrays
    """
    # Process dx range
    if len(dx) == 2:
        i1 = np.searchsorted(vx, dx[0])
        i2 = np.searchsorted(vx, dx[1])
        if i1 > 0:
            i1 -= 1
        idxx = slice(i1, i2 + 1)
    else:
        idxx = slice(None)

    # Process dy range
    if len(dy) == 2:
        i1 = np.searchsorted(vy, dy[0])
        i2 = np.searchsorted(vy, dy[1])
        if i1 > 0:
            i1 -= 1
        idxy = slice(i1, i2 + 1)
    else:
        idxy = slice(None)

    # Process dz range
    if len(dz) == 2:
        i1 = np.searchsorted(vz, dz[0])
        i2 = np.searchsorted(vz, dz[1])
        if i1 > 0:
            i1 -= 1
        idxz = slice(i1, i2 + 1)
    else:
        idxz = slice(None)

    xa = vx[idxx]
    ya = vy[idxy]
    za = vz[idxz]
    datacut = data[idxx, idxy, idxz]

    # Handle binning/integration
    if bin_axis is not None and len(bin_axis) == 3:
        dt = datacut != 0
        db = np.where(np.array(bin_axis) == 1)[0]
        if len(db) > 0:
            axis = db[0]
            if axis == 0:
                datacut = np.nansum(datacut, axis=0) / np.sum(dt, axis=0)
                xa = ya
                ya = za
                za = np.mean(dx)
            elif axis == 1:
                datacut = np.nansum(datacut, axis=1) / np.sum(dt, axis=1)
                ya = za
                za = np.mean(dy)
            elif axis == 2:
                datacut = np.nansum(datacut, axis=2) / np.sum(dt, axis=2)
                za = np.mean(dz)
            datacut = np.squeeze(datacut)

    datacut = np.nan_to_num(datacut, nan=0.0)

    return datacut, xa, ya, za
