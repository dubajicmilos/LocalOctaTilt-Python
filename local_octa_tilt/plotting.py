"""
Plotting functions for visualization of diffuse scattering data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from typing import Optional, Tuple, Dict, Any


def colormap_plot(
    X: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    labels: Optional[Dict[str, str]] = None,
    font_size: int = 14,
    cmap: str = 'magma',
    clim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True
) -> plt.Axes:
    """
    Create a 2D colormap plot.

    Parameters
    ----------
    X : np.ndarray
        X-axis values
    Y : np.ndarray
        Y-axis values
    M : np.ndarray
        2D intensity matrix
    labels : dict, optional
        Dictionary with 'xlabel', 'ylabel', 'title' keys
    font_size : int, optional
        Font size (default: 14)
    cmap : str, optional
        Colormap name (default: 'magma')
    clim : tuple, optional
        Color limits (min, max)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show_colorbar : bool, optional
        Whether to show colorbar (default: True)

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create meshgrid
    X_grid, Y_grid = np.meshgrid(Y, X)

    # Plot using pcolormesh
    pcm = ax.pcolormesh(Y_grid, X_grid, M, cmap=cmap, shading='gouraud')

    if show_colorbar:
        plt.colorbar(pcm, ax=ax)

    if clim is not None:
        pcm.set_clim(clim)

    ax.set_xlim(min(X), max(X))
    ax.set_ylim(min(Y), max(Y))
    ax.tick_params(labelsize=font_size, direction='out')

    if labels is not None:
        if 'xlabel' in labels:
            ax.set_xlabel(labels['xlabel'], fontsize=font_size)
        if 'ylabel' in labels:
            ax.set_ylabel(labels['ylabel'], fontsize=font_size)
        if 'title' in labels:
            ax.set_title(labels['title'], fontsize=font_size)

    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')

    return ax


def isosurface_plot(
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    isovalue: float,
    c_val: float = 1.75,
    cmap: str = 'magma',
    font_size: int = 16,
    ax: Optional[plt.Axes] = None,
    smooth: bool = True
) -> plt.Axes:
    """
    Create a 3D isosurface plot.

    Parameters
    ----------
    H, K, L : np.ndarray
        1D coordinate arrays
    M : np.ndarray
        3D intensity array
    isovalue : float
        Isosurface level
    c_val : float, optional
        Cutoff value for display range (default: 1.75)
    cmap : str, optional
        Colormap name (default: 'magma')
    font_size : int, optional
        Font size (default: 16)
    ax : matplotlib.axes.Axes, optional
        3D axes to plot on
    smooth : bool, optional
        Whether to smooth the data before plotting (default: True)

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axes object
    """
    from scipy.ndimage import uniform_filter

    # Create 3D meshgrid
    K_3D, H_3D, L_3D = np.meshgrid(K, H, L)

    # Crop to c_val range
    mask_H = (H >= -c_val) & (H <= c_val)
    mask_K = (K >= -c_val) & (K <= c_val)
    mask_L = (L >= -c_val) & (L <= c_val)

    H_3D_c = H_3D[np.ix_(mask_H, mask_K, mask_L)]
    K_3D_c = K_3D[np.ix_(mask_H, mask_K, mask_L)]
    L_3D_c = L_3D[np.ix_(mask_H, mask_K, mask_L)]
    Ms_c = M[np.ix_(mask_H, mask_K, mask_L)]

    # Smooth the data if requested
    if smooth:
        Ms = uniform_filter(Ms_c, size=3)
    else:
        Ms = Ms_c

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    try:
        # Use marching cubes to extract isosurface
        verts, faces, normals, values = measure.marching_cubes(
            Ms, isovalue, spacing=(
                H[1] - H[0] if len(H) > 1 else 1,
                K[1] - K[0] if len(K) > 1 else 1,
                L[1] - L[0] if len(L) > 1 else 1
            )
        )

        # Offset vertices to correct coordinates
        H_min = H[mask_H].min()
        K_min = K[mask_K].min()
        L_min = L[mask_L].min()

        verts[:, 0] += H_min
        verts[:, 1] += K_min
        verts[:, 2] += L_min

        # Create mesh
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor([1, 0.75, 0.65, 0.8])
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

        # Set limits
        ax.set_xlim(-c_val, c_val)
        ax.set_ylim(-c_val, c_val)
        ax.set_zlim(-c_val, c_val)

    except ValueError as e:
        print(f"Warning: Could not generate isosurface. {e}")

    # Set labels and appearance
    ax.set_xlabel('H [r.l.u.]', fontsize=font_size)
    ax.set_ylabel('K [r.l.u.]', fontsize=font_size)
    ax.set_zlabel('L [r.l.u.]', fontsize=font_size)
    ax.tick_params(labelsize=font_size - 2)
    ax.view_init(elev=21, azim=37)
    ax.set_box_aspect([1, 1, 1])
    ax.figure.set_facecolor('white')

    return ax


def plot_a_cation_orientations(
    mol_data: np.ndarray,
    resolution: int = 80,
    octahedron_radius: float = 0.3,
    cmap: str = 'magma'
) -> plt.Axes:
    """
    Plot A-cation orientations on a sphere with corner octahedra.

    Parameters
    ----------
    mol_data : np.ndarray
        Molecular orientation data (Nx3 array of unit vectors)
    resolution : int, optional
        Sphere mesh resolution (default: 80)
    octahedron_radius : float, optional
        Size of corner octahedra (default: 0.3)
    cmap : str, optional
        Colormap name (default: 'magma')

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axes object
    """
    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 2 * resolution)
    v = np.linspace(0, np.pi, resolution)
    xmesh = np.outer(np.cos(u), np.sin(v))
    ymesh = np.outer(np.sin(u), np.sin(v))
    zmesh = np.outer(np.ones_like(u), np.cos(v))

    # Prepare points
    points = np.stack([xmesh, ymesh, zmesh], axis=-1)
    counting = np.zeros((2 * resolution, resolution))

    # Reshape molecular data
    mol_data = mol_data.reshape(-1, 3)

    # Count orientations
    for i in range(len(mol_data)):
        dots = np.sum(points * mol_data[i], axis=-1)
        counting[dots > 0.995] += 1

    # Normalize
    heatmap = counting / counting.max() if counting.max() > 0 else counting

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere
    ax.plot_surface(xmesh, ymesh, zmesh, facecolors=plt.get_cmap(cmap)(heatmap),
                    rstride=1, cstride=1, shade=False)

    # Define corner positions
    corner_positions = np.array([
        [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1],
        [1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]
    ])

    # Define octahedron vertices
    octa_verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ]) * octahedron_radius

    # Define octahedron faces
    octa_faces = [
        [0, 2, 4], [0, 3, 4], [1, 2, 4], [1, 3, 4],
        [0, 2, 5], [0, 3, 5], [1, 2, 5], [1, 3, 5]
    ]

    # Plot octahedra at corners
    for pos in corner_positions:
        shifted_verts = octa_verts + pos
        for face in octa_faces:
            tri = Poly3DCollection([shifted_verts[face]])
            tri.set_facecolor('white')
            tri.set_edgecolor('black')
            tri.set_linewidth(0.5)
            ax.add_collection3d(tri)

    # Set view and limits
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.view_init(elev=21, azim=37)
    ax.set_box_aspect([1, 1, 1])

    # Clean up appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    return ax
