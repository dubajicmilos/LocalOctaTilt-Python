"""
GPU-accelerated simulation of QEDS diffuse scattering patterns.
Generates HK1.5 cross-section plots for MAPbBr3 and FAPbBr3.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_octa_tilt import (
    LocalSymmetrizedSimulation,
    two_d_slice,
    colormap_plot,
)


def main():
    print("=" * 60)
    print("LocalOctaTilt GPU Simulation")
    print("=" * 60)

    # Check GPU availability
    try:
        import cupy as cp
        cp.zeros(1)
        print(f"GPU: Enabled (CuPy {cp.__version__})")
        use_gpu = True
    except:
        print("GPU: Not available, using CPU")
        use_gpu = False

    # Path to structure factor files
    base_path = os.path.dirname(os.path.abspath(__file__))
    sf_path = os.path.join(base_path, 'LocalOctaTilt', 'Structure_Factor_Files')
    output_path = base_path

    # =========================================================================
    # MAPbBr3 Simulation
    # =========================================================================
    print("\n" + "-" * 60)
    print("Simulating MAPbBr3 (I4/mcm phase)...")
    print("-" * 60)

    filename_ma = os.path.join(sf_path, 'MAPbBr3_I4_mcm.txt')

    # MAPbBr3 DESY 300K parameters
    par_ma = [0.256412516, 0.078330761, 0.021337638, 15.22830278, 1.024027567]

    # Create simulation with GPU
    sim_ma = LocalSymmetrizedSimulation(
        filename_ma,
        phase='I4/mcm',
        dq=0.05,      # Full resolution
        Q_size=5.5,   # Full Q range
        use_gpu=use_gpu
    )

    # Run simulation
    start_time = time.perf_counter()
    S_ma, H_ma, K_ma, L_ma = sim_ma.simulate(
        delta1=par_ma[0],
        delta2=par_ma[1],
        C=par_ma[2],
        bgr=par_ma[3],
        deltag=par_ma[4]
    )
    elapsed = time.perf_counter() - start_time
    print(f"  Simulation completed in {elapsed:.2f}s")
    print(f"  Grid size: {S_ma.shape}")

    # Extract HK1.5 slice (actually 1.5KL in the notation)
    plane = '1.5KL'
    X_ma, Y_ma, Z_ma = two_d_slice(plane, S_ma, H_ma, K_ma, L_ma, log_mode='lin')

    # Normalize
    Z_ma_norm = Z_ma / Z_ma.max()

    # Plot and save
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    colormap_plot(X_ma, Y_ma, Z_ma_norm.T, ax=ax1,
                  labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
                         'title': 'MAPbBr3 - HK1.5 plane'},
                  clim=(0, 1))
    ax1.set_aspect('equal')
    plt.tight_layout()

    output_file_ma = os.path.join(output_path, 'MAPbBr3_HK1.5.png')
    fig1.savefig(output_file_ma, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file_ma}")

    # =========================================================================
    # FAPbBr3 Simulation
    # =========================================================================
    # print("\n" + "-" * 60)
    # print("Simulating FAPbBr3 (P4/mbm phase)...")
    # print("-" * 60)

    # filename_fa = os.path.join(sf_path, 'FAPbBr3_P4_mbm_pseudocubic.txt')

    # # FAPbBr3 parameters
    # par_fa = [0.11428078, 0.075591433, 0.000539108, 1.135693401, 100.0685359]

    # # Create simulation with GPU
    # sim_fa = LocalSymmetrizedSimulation(
    #     filename_fa,
    #     phase='P4/mbm',
    #     dq=0.05,
    #     Q_size=5.5,
    #     use_gpu=use_gpu
    # )

    # # Run simulation
    # start_time = time.perf_counter()
    # S_fa, H_fa, K_fa, L_fa = sim_fa.simulate(
    #     delta1=par_fa[0],
    #     delta2=par_fa[1],
    #     C=par_fa[2],
    #     bgr=par_fa[3],
    #     deltag=par_fa[4]
    # )
    # elapsed = time.perf_counter() - start_time
    # print(f"  Simulation completed in {elapsed:.2f}s")
    # print(f"  Grid size: {S_fa.shape}")

    # # Extract HK1.5 slice
    # X_fa, Y_fa, Z_fa = two_d_slice(plane, S_fa, H_fa, K_fa, L_fa, log_mode='lin')
    # Z_fa_norm = Z_fa / Z_fa.max()

    # # Plot and save
    # fig2, ax2 = plt.subplots(figsize=(8, 8))
    # colormap_plot(X_fa, Y_fa, Z_fa_norm.T, ax=ax2,
    #               labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
    #                      'title': 'FAPbBr3 - HK1.5 plane'},
    #               clim=(0, 1))
    # ax2.set_aspect('equal')
    # plt.tight_layout()

    # output_file_fa = os.path.join(output_path, 'FAPbBr3_HK1.5.png')
    # fig2.savefig(output_file_fa, dpi=150, bbox_inches='tight')
    # print(f"  Saved: {output_file_fa}")

    # # =========================================================================
    # # Combined comparison plot
    # # =========================================================================
    # print("\n" + "-" * 60)
    # print("Creating comparison plot...")
    # print("-" * 60)

    # fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

    # colormap_plot(X_ma, Y_ma, Z_ma_norm.T, ax=axes[0],
    #               labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
    #                      'title': 'MAPbBr3 (I4/mcm)'},
    #               clim=(0, 1), show_colorbar=False)
    # axes[0].set_aspect('equal')

    # colormap_plot(X_fa, Y_fa, Z_fa_norm.T, ax=axes[1],
    #               labels={'xlabel': 'K [r.l.u.]', 'ylabel': 'L [r.l.u.]',
    #                      'title': 'FAPbBr3 (P4/mbm)'},
    #               clim=(0, 1))
    # axes[1].set_aspect('equal')

    # plt.suptitle('HK1.5 Cross-sections - Diffuse Scattering from Local Octahedral Tilting',
    #              fontsize=14, y=1.02)
    # plt.tight_layout()

    # output_file_comp = os.path.join(output_path, 'comparison_HK1.5.png')
    # fig3.savefig(output_file_comp, dpi=150, bbox_inches='tight')
    # print(f"  Saved: {output_file_comp}")

    # print("\n" + "=" * 60)
    # print("Simulation complete!")
    # print("=" * 60)
    # print(f"\nOutput files:")
    # print(f"  - {output_file_ma}")
    # print(f"  - {output_file_fa}")
    # print(f"  - {output_file_comp}")

    # # Close all figures
    # plt.close('all')


if __name__ == '__main__':
    main()
