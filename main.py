# main.py

import numpy as np
from scipy.sparse import load_npz

# Import configurations and algorithms from other modules
import config
from reconstruction import sart, em, optimizers

def run_all_reconstructions():
    """
    Main function to orchestrate the loading of data and execution of
    reconstruction algorithms for various voxel sizes.
    """
    # Load the measurement data (D vector)
    opacity_vector = config.get_opacity_vector()
    # Tile the opacity vector for all 8 detectors to form the full D vector
    D_vector = np.tile(opacity_vector, 8)

    # Loop through all specified voxel sizes to run reconstructions
    for voxel_size, matrix_file in config.VOXEL_SIZES_AND_FILES.items():
        print("-" * 50)
        print(f"Starting reconstructions for voxel size: {voxel_size} m")

        # Load the pre-calculated sparse path-length matrix (L)
        L_matrix = load_npz(matrix_file)

        # Calculate the total number of voxels (N) based on the current resolution
        x_voxels = int((config.ROI_X[1] - config.ROI_X[0]) / voxel_size) + 1
        y_voxels = int((config.ROI_Y[1] - config.ROI_Y[0]) / voxel_size) + 1
        z_voxels = int((config.ROI_Z[1] - config.ROI_Z[0]) / voxel_size) + 1
        N = x_voxels * y_voxels * z_voxels

        # Define the initial guess for the density vector P (a zero vector)
        P_initial = np.zeros(N)

        # --- Run the selected algorithms (uncomment the ones you want to test) ---
        
        print("\n--- Running SART ---")
        sart.run(
            L=L_matrix,
            D=D_vector,
            P_initial=P_initial,
            voxel_size=voxel_size
        )
        
        print("\n--- Running EM ---")
        em.run(
            L=L_matrix,
            D=D_vector,
            voxel_size=voxel_size
        )
        
        print("\n--- Running L-BFGS ---")
        optimizers.run_lbfgs(
            L=L_matrix,
            D=D_vector,
            voxel_size=voxel_size
        )
        
        print("\n--- Running Trust-Region ---")
        optimizers.run_tr(
            L=L_matrix,
            D=D_vector,
            voxel_size=voxel_size
        )

    print("-" * 50)
    print("All reconstructions complete.")
    print(f"Results have been saved in the '{config.OUTPUT_ROOT_DIR}' directory.")


if __name__ == "__main__":
    run_all_reconstructions()