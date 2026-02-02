"""
Core simulation functions for local octahedral tilting in perovskites.
"""

import numpy as np
from scipy.ndimage import rotate
from typing import Tuple, Optional
import warnings

from .io import import_structure_factors
from .transforms import transform_hkl_no_inv


class LocalSymmetrizedSimulation:
    """
    Simulate S(q) arising from local I4/mcm (P4/mbm) nanodomains in cubic perovskite structures.

    Parameters
    ----------
    structure_factor_file : str
        Path to structure factor file (.txt)
    phase : str
        Crystal phase, either 'I4/mcm' or 'P4/mbm'
    dq : float, optional
        Q spacing in r.l.u. (default: 0.05)
    Q_size : float, optional
        Maximum |Q| of the grid (default: 5.5)
    symmetry : str, optional
        Point group symmetry, '-1' or '2/m' (default: '2/m')
    use_gpu : bool, optional
        Whether to use GPU acceleration with CuPy (default: False)
    """

    # Transformation matrices
    TM_I = np.array([
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 0.5]
    ])  # From I4/mcm to Pm-3m

    Rz = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    def __init__(
        self,
        structure_factor_file: str,
        phase: str,
        dq: float = 0.05,
        Q_size: float = 5.5,
        symmetry: str = '2/m',
        use_gpu: bool = False
    ):
        self.structure_factor_file = structure_factor_file
        self.phase = phase
        self.dq = dq
        self.Q_size = Q_size
        self.symmetry = symmetry
        self.use_gpu = use_gpu

        # Compute transformation matrix for P4/mbm
        self.TM_abc = np.eye(3) * 0.5 @ self.Rz

        # Resolution of Bragg peaks
        self.sigma = 0.0125 / np.sqrt(2)

        # Load structure factors
        self.G = import_structure_factors(structure_factor_file)

        # Try to import CuPy if GPU is requested
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                warnings.warn("CuPy not available, falling back to NumPy")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np

    def simulate(
        self,
        delta1: float,
        delta2: float,
        C: float,
        bgr: float,
        deltag: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation with given parameters.

        Parameters
        ----------
        delta1 : float
            Out-of-plane broadening (eq. S4 in SI)
        delta2 : float
            In-plane broadening (eq. S4 in SI)
        C : float
            Coefficient B in eq. S4
        bgr : float
            Background coefficient C in eq. S4
        deltag : float
            sigma_bgr in eq. S4

        Returns
        -------
        tuple
            (S, H, K, L) - Simulated intensity and coordinate arrays
        """
        xp = self.xp

        # Transform hkl based on phase
        if self.phase == 'I4/mcm':
            hkl_org = transform_hkl_no_inv(self.G, self.TM_I, np.eye(3))
        elif self.phase == 'P4/mbm':
            hkl_org = transform_hkl_no_inv(self.G, self.TM_abc, np.eye(3))
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        # Remove entries with zero structure factor
        hkl_org = hkl_org[hkl_org[:, 3] != 0]

        # Create coordinate grids
        H = np.arange(-self.Q_size, self.Q_size + self.dq, self.dq)
        K = np.arange(-self.Q_size, self.Q_size + self.dq, self.dq)
        L = np.arange(-self.Q_size, self.Q_size + self.dq, self.dq)

        # Create 3D meshgrid
        K_3D, H_3D, L_3D = np.meshgrid(K, H, L)

        # Move to GPU if available
        if self.use_gpu:
            H_3D = xp.asarray(H_3D)
            K_3D = xp.asarray(K_3D)
            L_3D = xp.asarray(L_3D)
            hkl_org = xp.asarray(hkl_org)

        # Initialize accumulator
        M = xp.zeros_like(H_3D)

        # Set sigma values
        sigmaL = sigmaK = sigmaH = self.sigma
        deltaH = deltaK = delta2
        deltaL = delta1

        # Process each reflection
        for i in range(len(hkl_org)):
            h, k, l, F = hkl_org[i]

            if self.symmetry == '-1':
                # Check if in specific octants
                if ((h >= 0 and k >= 0 and l >= 0) or
                    (h <= 0 and k >= 0 and l >= 0) or
                    (h >= 0 and k <= 0 and l >= 0) or
                    (h <= 0 and k <= 0 and l >= 0)):

                    M1 = self._compute_term(
                        H_3D, K_3D, L_3D, h, k, l, F,
                        sigmaH, sigmaK, sigmaL, deltaH, deltaK, deltaL
                    )
                    M1_inv = M1[::-1, ::-1, ::-1]
                    M = M + M1 + M1_inv

            elif self.symmetry == '2/m':
                # Check if in selected 2 octants
                if ((h >= 0 and k >= 0 and l >= 0) or
                    (h <= 0 and k >= 0 and l >= 0)):

                    M1 = self._compute_term(
                        H_3D, K_3D, L_3D, h, k, l, F,
                        sigmaH, sigmaK, sigmaL, deltaH, deltaK, deltaL
                    )

                    # Apply two-fold rotation (180 degrees) along z-axis
                    M1_rotated = xp.flip(xp.flip(M1, 0), 1)

                    # Apply mirror symmetry across y-plane (flip along z)
                    M1_mirror_z = M1[:, :, ::-1]

                    # Apply mirror to rotated
                    M1_rotated_mirror_z = M1_rotated[:, :, ::-1]

                    # Accumulate
                    M = M + M1 + M1_rotated + M1_mirror_z + M1_rotated_mirror_z

        # Transfer back from GPU if needed
        if self.use_gpu:
            M = xp.asnumpy(M)
            H_3D = xp.asnumpy(H_3D)
            K_3D = xp.asnumpy(K_3D)
            L_3D = xp.asnumpy(L_3D)

        # Apply rotations to simulate three twin components
        # MATLAB: Mr = imrotate3(M, 180, [1 0 -1], 'nearest', 'crop')
        # This is 180° rotation around axis [1,0,-1] (diagonal in H-L plane)
        # The rotation matrix for 180° around normalized [1/√2, 0, -1/√2] is:
        # [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
        # This swaps H↔L and negates K: (H,K,L) -> (L,-K,H) but for 180° it's (L,K,H) with sign changes
        # In array terms: swap axis 0 and 2, then flip axis 1
        Mr = np.swapaxes(M, 0, 2)[:, ::-1, :]

        # MATLAB: Mr1 = imrotate3(M, 180, [0 1 -1], 'nearest', 'crop')
        # This is 180° rotation around axis [0,1,-1] (diagonal in K-L plane)
        # The rotation matrix for 180° around normalized [0, 1/√2, -1/√2] is:
        # [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
        # This swaps K↔L and negates H
        # In array terms: swap axis 1 and 2, then flip axis 0
        Mr1 = np.swapaxes(M, 1, 2)[::-1, :, :]

        # Sum all three twin local structure components
        S = M + Mr + Mr1

        # Add background
        S = C * S + bgr * np.exp(
            -((L_3D - 0)**2 / (2 * deltag**2) +
              (K_3D - 0)**2 / (2 * deltag**2) +
              (H_3D - 0)**2 / (2 * deltag**2))
        )

        return S, H, K, L

    def _compute_term(
        self,
        H_3D: np.ndarray,
        K_3D: np.ndarray,
        L_3D: np.ndarray,
        h: float, k: float, l: float, F: float,
        sigmaH: float, sigmaK: float, sigmaL: float,
        deltaH: float, deltaK: float, deltaL: float
    ) -> np.ndarray:
        """
        Compute the Gaussian term for a single reflection.
        """
        xp = self.xp

        # Check if this is a Bragg peak (integer indices)
        is_bragg = (h % 1 == 0 and k % 1 == 0 and l % 1 == 0)

        if is_bragg:
            # Use sigma for Bragg peaks
            result = F**2 * xp.exp(
                -((L_3D - l)**2 / (2 * sigmaL**2) +
                  (K_3D - k)**2 / (2 * sigmaK**2) +
                  (H_3D - h)**2 / (2 * sigmaH**2))
            )
        else:
            # Use delta for diffuse scattering
            result = F**2 * xp.exp(
                -((L_3D - l)**2 / (2 * deltaL**2) +
                  (K_3D - k)**2 / (2 * deltaK**2) +
                  (H_3D - h)**2 / (2 * deltaH**2))
            )

        return result


def local_symmetrized_fun(
    name: str,
    phase: str,
    par: list,
    dq: float = 0.05,
    Q_size: float = 5.5,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function matching MATLAB interface.

    Parameters
    ----------
    name : str
        Path to structure factor file
    phase : str
        Crystal phase ('I4/mcm' or 'P4/mbm')
    par : list
        Parameters [delta1, delta2, C, bgr, deltag]
    dq : float, optional
        Q spacing (default: 0.05)
    Q_size : float, optional
        Maximum Q (default: 5.5)
    use_gpu : bool, optional
        Use GPU acceleration (default: False)

    Returns
    -------
    tuple
        (S, H, K, L)
    """
    sim = LocalSymmetrizedSimulation(
        name, phase, dq=dq, Q_size=Q_size, use_gpu=use_gpu
    )
    return sim.simulate(par[0], par[1], par[2], par[3], par[4])
