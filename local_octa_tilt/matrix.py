"""
Matrix class for handling 2D data with axes.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d


class Matrix:
    """
    A class for handling 2D matrix data with X and Y axis coordinates.

    Parameters
    ----------
    x : np.ndarray
        X-axis values
    y : np.ndarray
        Y-axis values
    M : np.ndarray
        2D data matrix

    Attributes
    ----------
    X : np.ndarray
        X-axis coordinates
    Y : np.ndarray
        Y-axis coordinates
    M : np.ndarray
        Data matrix
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, M: np.ndarray):
        self.X = np.asarray(x)
        self.Y = np.asarray(y)
        self.M = np.asarray(M)

    def cut_matrix(
        self,
        x_border: Tuple[float, float],
        y_border: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a sub-region of the matrix.

        Parameters
        ----------
        x_border : tuple
            (x_min, x_max) range to extract
        y_border : tuple
            (y_min, y_max) range to extract

        Returns
        -------
        tuple
            (X, Y, M) - Cropped coordinate arrays and matrix
        """
        # Find indices closest to borders
        itm = np.argmin(np.abs(self.X - x_border[0]))
        ith = np.argmin(np.abs(self.X - x_border[1]))
        ilm = np.argmin(np.abs(self.Y - y_border[0]))
        i1h = np.argmin(np.abs(self.Y - y_border[1]))

        X = self.X[itm:ith + 1]
        Y = self.Y[ilm:i1h + 1]
        M = self.M[itm:ith + 1, ilm:i1h + 1]

        return X, Y, M

    def fourier(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Fourier transform along the Y-axis for each row.

        Returns
        -------
        tuple
            (frequencies, fourier_amplitudes)
        """
        Mat = self.M

        # Get time step from Y values
        T = self.Y[6] - self.Y[5] if len(self.Y) > 6 else self.Y[1] - self.Y[0]

        # Handle duplicate Y values
        _, unique_idx = np.unique(self.Y, return_index=True)
        Y_unique = self.Y[unique_idx]

        tN = np.arange(self.Y[0], self.Y[-1] + T, T)
        Fur = []

        for i in range(Mat.shape[0]):
            Yc = Mat[i, unique_idx]

            # Interpolate to regular grid
            interp_func = interp1d(Y_unique, Yc, kind='linear', fill_value='extrapolate')
            Yn = interp_func(tN)

            L = len(Yn)
            P2 = np.abs(np.fft.fft(Yn / L))
            P1 = P2[:L // 2 + 1]

            f = (1 / T) * np.arange(0, L // 2 + 1) / L * 1000
            Fur.append(P1)

        return f, np.array(Fur).T

    def plot_2d(self, **kwargs):
        """
        Create a 2D colormap plot of the matrix.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to colormap_plot

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import colormap_plot
        return colormap_plot(self.X, self.Y, self.M, **kwargs)
