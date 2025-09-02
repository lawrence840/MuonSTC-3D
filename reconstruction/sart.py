# reconstruction/sart.py

import time
import os
import numpy as np
from numba import njit

import config
from .common import apply_constraints
from utils import plotting

@njit
def _sart_iteration(L_data, L_indices, L_indptr, D, P, relaxation_factor):
    """A single iteration of the SART algorithm, JIT-compiled with Numba for performance."""
    for i in range(len(L_indptr) - 1):
        start, end = L_indptr[i], L_indptr[i+1]
        Li = L_data[start:end]
        indices = L_indices[start:end]
        
        numerator = D[i] - np.sum(Li * P[indices])
        denominator = np.sum(Li ** 2)
        
        if denominator > 1e-9:
            update = relaxation_factor * (numerator / denominator) * Li
            P[indices] += update
    return P

def run(L, D, P_initial, voxel_size):
    """
    Executes the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Args:
        L (scipy.sparse.csr_matrix): The path-length matrix.
        D (np.ndarray): The measurement vector.
        P_initial (np.ndarray): The initial guess for the density vector.
        voxel_size (float): The size of each voxel.
    
    Returns:
        np.ndarray: The final reconstructed density vector.
    """
    start_time_total = time.time()
    P_sart = P_initial.copy()
    
    # Ensure L is in CSR format for efficient row slicing and get its components
    L_csr = L.tocsr()
    L_data, L_indices, L_indptr = L_csr.data, L_csr.indices, L_csr.indptr

    # Create algorithm-specific output directory
    algo_output_dir = os.path.join(config.OUTPUT_ROOT_DIR, "SART", f"voxel_{voxel_size}")
    if not os.path.exists(algo_output_dir):
        os.makedirs(algo_output_dir)

    # Log runtime for each iteration
    with open(os.path.join(algo_output_dir, "RunTime.txt"), "w") as time_file:
        for iteration in range(config.SART_ITERATIONS):
            iter_num = iteration + 1
            start_time_iter = time.time()
            
            P_sart = _sart_iteration(L_data, L_indices, L_indptr, D, P_sart, config.SART_RELAXATION)
            
            sart_time = time.time() - start_time_iter

            # Apply regularization
            start_time_constraints = time.time()
            P_sart = apply_constraints(P_sart, voxel_size)
            constraints_time = time.time() - start_time_constraints
            
            # Log times
            time_file.write(f"Iteration {iter_num}:\n")
            time_file.write(f"  SART Time: {sart_time:.6f} seconds\n")
            time_file.write(f"  Constraints Time: {constraints_time:.6f} seconds\n")
            time_file.flush()

            # Save intermediate results at specified intervals
            if iter_num % config.SART_SAVE_INTERVAL == 0:
                print(f"SART: Iteration {iter_num}: Saving plots...")
                plotting.save_plots_and_data(P_sart, iter_num, "SART", voxel_size)
    
    # Save the final result
    plotting.save_plots_and_data(P_sart, config.SART_ITERATIONS, "SART", voxel_size)
    
    total_duration = time.time() - start_time_total
    print(f"SART Total Time: {total_duration:.2f} seconds for Voxel Size {voxel_size}")
    return P_sart