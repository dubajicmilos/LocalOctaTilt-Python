"""
Example script: Simulation of QEDS diffuse scattering patterns.

This script demonstrates how to simulate S(q) arising from local I4/mcm (P4/mbm)
nanodomains in cubic perovskite structures, reproducing results from:
"Dynamic nanodomains dictate macroscopic properties in lead halide perovskites"

Usage:
    python simulation_qeds.py
"""

import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_octa_tilt import (
    LocalSymmetrizedSimulation,
    two_d_slice,
    colormap_plot,
    isosurface_plot,
)


def main():
    # Path to structure factor files
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sf_path = os.path.join(base_path, 'LocalOctaTilt', 'Structure_Factor_Files')

    # =========================================================================
    # MAPbBr3 Simulation
    # =========================================================================
    print("Simulating MAPbBr3 (I4/mcm phase)...")

    filename_ma = os.path.join(sf_path, 'MAPbBr3_I4_mcm.txt')
    phase_ma = 'I4/mcm'

    # Parameters determined from experimental data (MAPbBr3 DESY 300K)
    # [delta1, delta2, C, bgr, deltag]
    par_ma = [0.256412516, 0.078330761, 0.021337638, 15.22830278, 1.024027567]

    # Create simulation
    sim_ma = LocalSymmetrizedSimulation(filename_ma, phase_ma)

    # Run simulation
    S_ma, H_ma, K_ma, L_ma = sim_ma.simulate(
        delta1=par_ma[0],
        delta2=par_ma[1],
        C=par_ma[2],
        bgr=par_ma[3],
        deltag=par_ma[4]
    )

    # Extract 2D slice at L=1.5
    plane = '1.5KL'
    X_ma, Y_ma, Z_ma = two_d_slice(plane, S_ma, H_ma, K_ma, L_ma, log_mode='lin')

    # Normalize and plot
    Z_ma_norm = Z_ma / Z_ma.max()

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colormap_plot(X_ma, Y_ma, Z_ma_norm.T, ax=ax1,
                  labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
                         'title': 'MAPbBr3 - 1.5KL plane'},
                  clim=(0, 1))
    plt.tight_layout()

    # Isosurface plot (without background for cleaner visualization)
    print("Creating MAPbBr3 isosurface...")
    par_ma_iso = [0.256412516, 0.078330761, 0.021337638, 0, 1.024027567]
    S_ma_iso, H_ma_iso, K_ma_iso, L_ma_iso = sim_ma.simulate(
        delta1=par_ma_iso[0],
        delta2=par_ma_iso[1],
        C=par_ma_iso[2],
        bgr=par_ma_iso[3],
        deltag=par_ma_iso[4]
    )

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    isosurface_plot(H_ma_iso, K_ma_iso, L_ma_iso, S_ma_iso, isovalue=6, c_val=1.75, ax=ax2)
    ax2.set_title('MAPbBr3 Isosurface')

    # =========================================================================
    # FAPbBr3 Simulation
    # =========================================================================
    print("Simulating FAPbBr3 (P4/mbm phase)...")

    filename_fa = os.path.join(sf_path, 'FAPbBr3_P4_mbm_pseudocubic.txt')
    phase_fa = 'P4/mbm'

    # Parameters for FAPbBr3
    par_fa = [0.11428078, 0.075591433, 0.000539108, 1.135693401, 100.0685359]

    # Create simulation
    sim_fa = LocalSymmetrizedSimulation(filename_fa, phase_fa)

    # Run simulation
    S_fa, H_fa, K_fa, L_fa = sim_fa.simulate(
        delta1=par_fa[0],
        delta2=par_fa[1],
        C=par_fa[2],
        bgr=par_fa[3],
        deltag=par_fa[4]
    )

    # Extract 2D slice
    X_fa, Y_fa, Z_fa = two_d_slice(plane, S_fa, H_fa, K_fa, L_fa, log_mode='lin')
    Z_fa_norm = Z_fa / Z_fa.max()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    colormap_plot(X_fa, Y_fa, Z_fa_norm.T, ax=ax3,
                  labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
                         'title': 'FAPbBr3 - 1.5KL plane'},
                  clim=(0, 1))
    plt.tight_layout()

    # Isosurface for FAPbBr3
    print("Creating FAPbBr3 isosurface...")
    par_fa_iso = [0.11428078, 0.075591433, 0.000539108, 0, 100.0685359]
    S_fa_iso, H_fa_iso, K_fa_iso, L_fa_iso = sim_fa.simulate(
        delta1=par_fa_iso[0],
        delta2=par_fa_iso[1],
        C=par_fa_iso[2],
        bgr=par_fa_iso[3],
        deltag=par_fa_iso[4]
    )

    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    isosurface_plot(H_fa_iso, K_fa_iso, L_fa_iso, S_fa_iso, isovalue=1.5, c_val=1.75, ax=ax4)
    ax4.set_title('FAPbBr3 Isosurface')

    print("Simulation complete. Displaying plots...")
    plt.show()


if __name__ == '__main__':
    main()
