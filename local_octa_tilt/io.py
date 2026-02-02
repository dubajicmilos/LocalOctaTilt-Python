"""
I/O functions for loading structure factor files and MD data.
"""

import numpy as np
import pandas as pd
import re
import h5py
from pathlib import Path
from typing import Tuple, Optional


def import_structure_factors(filename: str, start_row: int = 7) -> pd.DataFrame:
    """
    Import structure factors from a SingleCrystal-generated text file.

    Parameters
    ----------
    filename : str
        Path to the structure factor file (.txt)
    start_row : int, optional
        Line number where data starts (default: 7, 0-indexed)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Number, h, k, l, d, FRe, FIm, Phase, F
    """
    data = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines[start_row - 1:]:
        line = line.strip()
        if not line or line.startswith('*'):
            continue

        # Parse line with format: [   1]   -8  -3  -3   0.9488   1.02334e+01   0.00000e+00    0.000   1.02334e+01
        # Remove brackets and split
        match = re.match(r'\[\s*(\d+)\]\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)', line)

        if match:
            values = [float(match.group(i)) for i in range(1, 10)]
            data.append(values)

    df = pd.DataFrame(data, columns=['Number', 'h', 'k', 'l', 'd', 'FRe', 'FIm', 'Phase', 'F'])
    return df


def load_md_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Molecular Dynamics data from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file

    Returns
    -------
    tuple
        (SQ_QS, SQ_full, H, K, Energy, array3D) - The loaded datasets

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    with h5py.File(filepath, 'r') as f:
        SQ_QS = f['/SQ_QS'][:]
        SQ_full = f['/SQ_full'][:]
        H = f['/H'][:]
        K = f['/K'][:]
        Energy = f['/E'][:]
        array3D = f['/SQ_E'][:]

    return SQ_QS, SQ_full, H, K, Energy, array3D
