# reconstruction/common.py

import numpy as np
import config

def apply_constraints(P, voxel_size):
    """
    Applies regularization by enforcing physical and geometric constraints.
    This is a form of hard regularization, forcing the solution to exist
    within a known, valid space.
    
    Args:
        P (np.ndarray): The 1D density vector.
        voxel_size (float): The size of each voxel.

    Returns:
        np.ndarray: The constrained 1D density vector.
    """
    # 1. Density Range Constraint: Enforces plausible physical density values.
    P = np.clip(P, 1E-9, 7)
    
    # Calculate the number of voxels for reshaping
    x_voxels = int((config.ROI_X[1] - config.ROI_X[0]) / voxel_size) + 1
    y_voxels = int((config.ROI_Y[1] - config.ROI_Y[0]) / voxel_size) + 1
    z_voxels = int((config.ROI_Z[1] - config.ROI_Z[0]) / voxel_size) + 1
    
    P_reshaped = P.reshape((x_voxels, y_voxels, z_voxels))
    
    # 2. Geometric Constraint: Sets density to zero for voxels outside a known boundary.
    for x in range(x_voxels):
        for y in range(y_voxels):
            for z in range(z_voxels):
                # Convert voxel indices back to physical coordinates
                x_coord = config.ROI_X[0] + x * voxel_size
                y_coord = config.ROI_Y[0] + y * voxel_size
                z_coord = config.ROI_Z[0] + z * voxel_size
                
                # If a voxel is outside the known cylindrical volume, set its density to zero
                if z_coord > 42 or np.sqrt(x_coord**2 + y_coord**2) > 8:
                    P_reshaped[x, y, z] = 0
                    
    return P_reshaped.flatten()