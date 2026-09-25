"""
Regression test for two_d_slice: each plane specification must fix the axis it names.
"""

import numpy as np
import pytest

from local_octa_tilt.slicing import two_d_slice

# Unequal grids, so that slicing the wrong axis cannot pass by accident
H = np.arange(-2.0, 2.01, 0.5)  # 9 points
K = np.arange(-1.5, 1.51, 0.5)  # 7 points
L = np.arange(-1.0, 2.51, 0.5)  # 8 points

# S[iH, iK, iL], the layout returned by LocalSymmetrizedSimulation.simulate.
# The values encode the coordinates of each voxel: S = 100*H + 10*K + L.
S = 100 * H[:, None, None] + 10 * K[None, :, None] + L[None, None, :]


@pytest.mark.parametrize("plane, X_expected, Y_expected, Z_expected", [
    # HK plane at L = 1.5: rows follow H, columns follow K
    ('HK1.5', K, H, 100 * H[:, None] + 10 * K[None, :] + 1.5),
    # HL plane at K = 0.5: rows follow H, columns follow L
    ('H0.5L', L, H, 100 * H[:, None] + 10 * 0.5 + L[None, :]),
    # KL plane at H = 1.5: rows follow L, columns follow K
    ('1.5KL', K, L, 100 * 1.5 + 10 * K[None, :] + L[:, None]),
])
def test_two_d_slice_fixes_named_axis(plane, X_expected, Y_expected, Z_expected):
    X, Y, Z = two_d_slice(plane, S, H, K, L)
    np.testing.assert_array_equal(X, X_expected)
    np.testing.assert_array_equal(Y, Y_expected)
    assert Z.shape == (len(Y), len(X))
    np.testing.assert_array_equal(Z, Z_expected)
