# reconstruction/em.py

import time
import os
import numpy as np
from scipy.ndimage import median_filter

import config
from .common import apply_constraints
from utils import plotting

def _tv_regularization(P, shape, weight):
    """
    Applies a smoothing (Total Variation-like) regularization using a median filter.
    This promotes smoother, less noisy reconstructions.
    """
    P_reshaped = P.reshape(shape)
    smoothed_P = median_filter(P_reshaped, size=3)
    return P + weight * (smoothed_P.flatten() - P)

def _em_update(P, L, D, voxel_size):
    """The core Expectation-Maximization (EM) update process."""
    # Calculate voxel dimensions for reshaping
    x_voxels = int((config.ROI_X[1] - config.ROI_X[0]) / voxel_size) + 1
    y_voxels = int((config.ROI_Y[1] - config.ROI_Y[0]) / voxel_size) + 1
    z_voxels = int((config.ROI_Z[1] - config.ROI_Z[0]) / voxel_size) + 1
    shape = (x_voxels, y_voxels, z_voxels)
    
    P_em = P.copy()
    
    algo_output_dir = os.path.join(config.OUTPUT_ROOT_DIR, "EM", f"voxel_{voxel_size}")
    if not os.path.exists(algo_output_dir):
        os.makedirs(algo_output_dir)

    with open(os.path.join(algo_output_dir, "RunTime.txt"), "w") as time_file:
        for iteration in range(config.EM_ITERATIONS):
            iter_num = iteration + 1
            start_time_iter = time.time()
            
            # E-step: Calculate expected projection from the current estimate
            model_D = L.dot(P_em)
            
            # M-step: Update the density distribution
            residual = D / (model_D + 1e-9)
            weighted_residual = L.T.dot(residual)
            P_new = P_em * weighted_residual
            
            em_time = time.time() - start_time_iter
            
            # Apply hard regularization
            start_time_constraints = time.time()
            P_new = apply_constraints(P_new, voxel_size)
            constraints_time = time.time() - start_time_constraints
            
            # Apply soft regularization (smoothing)
            P_new = _tv_regularization(P_new, shape, weight=config.EM_TV_WEIGHT)
            
            # Limit the update step size to stabilize convergence
            P_new = P_em + config.EM_UPDATE_STEP * (P_new - P_em)
            
            P_em = P_new
            
            time_file.write(f"Iteration {iter_num}:\n")
            time_file.write(f"  EM Time: {em_time:.6f} seconds\n")
            time_file.write(f"  Constraints Time: {constraints_time:.6f} seconds\n")
            time_file.flush()

            if iter_num % 2 == 0:
                print(f"EM: Iteration {iter_num}: Saving plots...")
                plotting.save_plots_and_data(P_em, iter_num, "EM", voxel_size)

    plotting.save_plots_and_data(P_em, config.EM_ITERATIONS, "EM", voxel_size)
    return P_em

def run(L, D, voxel_size):
    """
    Executes the Expectation-Maximization (EM) algorithm.

    Args:
        L (scipy.sparse.csr_matrix): The path-length matrix.
        D (np.ndarray): The measurement vector.
        voxel_size (float): The size of each voxel.
    
    Returns:
        np.ndarray: The final reconstructed density vector.
    """
    start_time_total = time.time()
    
    x_voxels = int((config.ROI_X[1] - config.ROI_X[0]) / voxel_size) + 1
    y_voxels = int((config.ROI_Y[1] - config.ROI_Y[0]) / voxel_size) + 1
    z_voxels = int((config.ROI_Z[1] - config.ROI_Z[0]) / voxel_size) + 1
    N = x_voxels * y_voxels * z_voxels
    
    # Initialize density with a random distribution for the EM algorithm
    P_initial = np.random.uniform(low=0, high=7, size=N)
    P_em = _em_update(P_initial, L, D, voxel_size=voxel_size)
    
    total_duration = time.time() - start_time_total
    print(f"EM Total Time: {total_duration:.2f} seconds for Voxel Size {voxel_size}")
    return P_em