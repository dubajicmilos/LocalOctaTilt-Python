# LocalOctaTilt - Python Implementation

A Python implementation of the phenomenological model for simulating diffuse scattering arising from local octahedral tilting in lead halide perovskites.

This code reproduces results from the publication:
> **"Dynamic nanodomains dictate macroscopic properties in lead halide perovskites"**

## Overview

This package simulates S(q) arising from local I4/mcm (P4/mbm) nanodomains in cubic perovskite structures. The model accounts for:

- Local symmetry breaking from average cubic Pm-3m to tetragonal I4/mcm or P4/mbm
- Anisotropic broadening of superstructure reflections
- Multiple twin domain orientations
- Background scattering contributions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/dubajicmilos/LocalOctaTilt-Python.git
cd LocalOctaTilt-Python
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.20.0 | Numerical computing |
| scipy | >= 1.7.0 | Scientific computing, 3D rotations |
| pandas | >= 1.3.0 | Data handling for structure factors |
| matplotlib | >= 3.4.0 | Visualization |
| h5py | >= 3.0.0 | HDF5 file support for MD data |
| scikit-image | >= 0.18.0 | Isosurface extraction |

### Optional GPU Acceleration

For GPU-accelerated simulations (NVIDIA CUDA):

```bash
pip install cupy>=10.0.0
```

## Quick Start

### Basic Simulation

```python
from local_octa_tilt import LocalSymmetrizedSimulation, two_d_slice, colormap_plot
import matplotlib.pyplot as plt

# Load structure factors and create simulation
sim = LocalSymmetrizedSimulation(
    'Structure_Factor_Files/MAPbBr3_I4_mcm.txt',
    phase='I4/mcm'
)

# Parameters: [delta1, delta2, C, bgr, deltag]
# delta1: Out-of-plane broadening
# delta2: In-plane broadening
# C: Intensity scaling factor
# bgr: Background level
# deltag: Background width
params = [0.256, 0.078, 0.021, 15.2, 1.02]

# Run simulation
S, H, K, L = sim.simulate(
    delta1=params[0],
    delta2=params[1],
    C=params[2],
    bgr=params[3],
    deltag=params[4]
)

# Extract 2D slice at L=1.5
X, Y, Z = two_d_slice('1.5KL', S, H, K, L)

# Plot
fig, ax = plt.subplots()
Z_norm = Z / Z.max()
colormap_plot(X, Y, Z_norm.T, ax=ax,
              labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]'})
plt.show()
```

### Isosurface Visualization

```python
from local_octa_tilt import isosurface_plot
import matplotlib.pyplot as plt

# Simulate without background for cleaner isosurface
params_iso = [0.256, 0.078, 0.021, 0, 1.02]
S, H, K, L = sim.simulate(params_iso[0], params_iso[1], params_iso[2],
                          params_iso[3], params_iso[4])

# Create 3D isosurface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
isosurface_plot(H, K, L, S, isovalue=6, c_val=1.75, ax=ax)
plt.show()
```

## Structure Factor Files

Structure factor files are generated using [SingleCrystal](https://crystalmaker.com/singlecrystal/) software. The files contain:

- Miller indices (h, k, l)
- d-spacing
- Structure factor components (F_Re, F_Im)
- Phase angle
- Structure factor magnitude |F|

### Included Files

| File | Phase | Material |
|------|-------|----------|
| `MAPbBr3_I4_mcm.txt` | I4/mcm | MAPbBr3 |
| `FAPbBr3_P4_mbm_pseudocubic.txt` | P4/mbm | FAPbBr3 |

### Creating Custom Structure Factor Files

1. Open SingleCrystal software
2. Load or create your crystal structure
3. Calculate structure factors for desired hkl range
4. Export as text file with the standard format

## API Reference

### LocalSymmetrizedSimulation

Main simulation class.

```python
LocalSymmetrizedSimulation(
    structure_factor_file: str,
    phase: str,              # 'I4/mcm' or 'P4/mbm'
    dq: float = 0.05,        # Q spacing in r.l.u.
    Q_size: float = 5.5,     # Max |Q|
    symmetry: str = '2/m',   # Point group: '-1' or '2/m'
    use_gpu: bool = False    # Enable GPU acceleration
)
```

### two_d_slice

Extract 2D slices from 3D reciprocal space data.

```python
two_d_slice(
    plane: str,       # e.g., 'HK1.5', 'H0.5L', '1.5KL'
    S: np.ndarray,    # 3D intensity array
    H, K, L: np.ndarray,
    log_mode: str = 'lin',  # 'lin' or 'log'
    plot_mode: str = 'off'  # 'on' or 'off'
) -> (X, Y, Z)
```

### colormap_plot

Create 2D colormap visualizations.

```python
colormap_plot(
    X, Y: np.ndarray,
    M: np.ndarray,
    labels: dict = None,    # {'xlabel', 'ylabel', 'title'}
    cmap: str = 'magma',
    clim: tuple = None
)
```

### isosurface_plot

Create 3D isosurface visualizations.

```python
isosurface_plot(
    H, K, L: np.ndarray,
    M: np.ndarray,
    isovalue: float,
    c_val: float = 1.75,    # Display range cutoff
    smooth: bool = True
)
```

## Model Parameters

The simulation uses 5 parameters described in equation S4 of the supplementary information:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| delta1 | δ₁ | Out-of-plane peak broadening |
| delta2 | δ₂ | In-plane peak broadening |
| C | B | Intensity scaling coefficient |
| bgr | C | Background level |
| deltag | σ_bgr | Background Gaussian width |

### Fitted Parameters for Reference Materials

**MAPbBr3 (DESY, 300K):**
```python
params = [0.256412516, 0.078330761, 0.021337638, 15.22830278, 1.024027567]
```

**FAPbBr3:**
```python
params = [0.11428078, 0.075591433, 0.000539108, 1.135693401, 100.0685359]
```

## Examples

See the `examples/` directory for complete example scripts:

- `simulation_qeds.py` - Full simulation workflow for MAPbBr3 and FAPbBr3

## Performance

### GPU vs CPU Benchmarks (GTX 1070)

| Grid Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 21³ | 0.2s | 2.2s | 0.1x |
| 61³ | 10s | 1.5s | **7x** |
| 81³ | 26s | 2.1s | **12x** |
| 201³ | 438s | 21s | **21x** |

*Note: For small grids, CPU is faster due to GPU memory transfer overhead.*

### Recommended Settings

For testing:
```python
sim = LocalSymmetrizedSimulation(file, phase, dq=0.1, Q_size=3.0)  # Fast
```

For publication-quality:
```python
sim = LocalSymmetrizedSimulation(file, phase, dq=0.05, Q_size=5.5, use_gpu=True)
```

## Comparison with MATLAB

This Python implementation produces results equivalent to the original MATLAB code. Key differences:

| Feature | MATLAB | Python |
|---------|--------|--------|
| GPU support | Native CUDA | CuPy (optional) |
| 3D rotation | imrotate3 | scipy.ndimage.rotate |
| Isosurface | isosurface/patch | scikit-image marching_cubes |
| Colormap | magma | matplotlib magma |

## Citation

If you use this code in your research, please cite:

**Full citation:** Dubajic, M. et al. Dynamic nanodomains dictate macroscopic properties in lead halide perovskites. *Nat. Nanotechnol.* **20**, 755–763 (2025). https://doi.org/10.1038/s41565-025-01917-0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original MATLAB implementation by [dubajicmilos](https://github.com/dubajicmilos)
- Structure factor calculations using [SingleCrystal](https://crystalmaker.com/singlecrystal/)
