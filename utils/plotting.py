# utils/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt

import config
from reconstruction.common import apply_constraints

def save_plots_and_data(P_estimated, iteration, algorithm_name, voxel_size):
    """
    Saves 2D slice plots and raw data for a given reconstructed volume.

    Args:
        P_estimated (np.ndarray): The 1D reconstructed density vector.
        iteration (int): The current iteration number.
        algorithm_name (str): The name of the algorithm (e.g., "SART").
        voxel_size (float): The size of each voxel.
    """
    algo_output_dir = os.path.join(config.OUTPUT_ROOT_DIR, algorithm_name, f"voxel_{voxel_size}")
    if not os.path.exists(algo_output_dir):
        os.makedirs(algo_output_dir)

    x_voxels = int((config.ROI_X[1] - config.ROI_X[0]) / voxel_size) + 1
    y_voxels = int((config.ROI_Y[1] - config.ROI_Y[0]) / voxel_size) + 1
    z_voxels = int((config.ROI_Z[1] - config.ROI_Z[0]) / voxel_size) + 1
    
    # Ensure constraints are applied before saving
    P_constrained = apply_constraints(P_estimated, voxel_size)

    # Define x-positions of the slices to plot
    x_positions_to_plot = [0, 10]
    
    for x_pos in x_positions_to_plot:
        x_index = round((x_pos - config.ROI_X[0]) / voxel_size)
        
        if 0 <= x_index < x_voxels:
            sub_dir = os.path.join(algo_output_dir, f'x_{x_pos}')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            
            # Reshape the 1D vector into a 3D volume and extract the slice
            slice_data = P_constrained.reshape((x_voxels, y_voxels, z_voxels))[x_index, :, :].T
            
            # --- Save Plot ---
            plt.figure(figsize=(8, 10))
            plt.imshow(slice_data, cmap='jet', origin='lower', extent=[config.ROI_Y[0], config.ROI_Y[1], config.ROI_Z[0], config.ROI_Z[1]])
            plt.colorbar(label="Estimated Density")
            plt.title(f'Density Slice at x={x_pos}m\n(Iter: {iteration}, Alg: {algorithm_name}, Voxel: {voxel_size}m)')
            plt.xlabel('Y-axis (m)')
            plt.ylabel('Z-axis (m)')
            
            plot_filename = os.path.join(sub_dir, f'density_slice_iter_{iteration}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # --- Save Raw Data ---
            data_filename = os.path.join(sub_dir, f'density_slice_iter_{iteration}.txt')
            np.savetxt(data_filename, slice_data)