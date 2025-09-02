# scripts/calculate_paths.py

import numpy as np
from scipy.sparse import lil_matrix, save_npz
from math import pi, sin, cos, sqrt

# --- Configuration ---
# NOTE: This script is standalone. For a more integrated project,
# these values would be imported from a central config file.

# Define the Region of Interest (ROI) in meters
x_begin, x_end = -10, 10
y_begin, y_end = -10, 10
z_begin, z_end = 0, 50
voxel_size = 1.0  # Change this to generate matrices for different resolutions

# --- Voxel Grid Setup ---

# Calculate the number of voxels along each axis
x_voxels = int((x_end - x_begin) / voxel_size)
y_voxels = int((y_end - y_begin) / voxel_size)
z_voxels = int((z_end - z_begin) / voxel_size)
N = x_voxels * y_voxels * z_voxels
print(f"Voxel Size: {voxel_size} m")
print(f"Grid Dimensions: {x_voxels} x {y_voxels} x {z_voxels}")
print(f"Total number of voxels (N): {N}")

# Create a 3D grid to map (x,y,z) voxel coordinates to a 1D index
voxel_indices = np.arange(N).reshape((x_voxels, y_voxels, z_voxels))

# --- Detector Setup ---

# Define positions and orientations for 8 detectors arranged in a circle
detectors = []
detector_angles = np.arange(0, 360, 45) # 8 detectors every 45 degrees

for angle in detector_angles:
    rad = np.deg2rad(angle)
    # Detector position on a circle of radius 20
    x = round(20 * np.cos(rad), 10)
    y = round(20 * np.sin(rad), 10)
    z = 1
    
    # Central direction vector pointing towards the origin with a downward angle
    dx = round(-np.cos(rad) * np.sqrt(3) / 2, 10)
    dy = round(-np.sin(rad) * np.sqrt(3) / 2, 10)
    dz = 0.5
    
    detectors.append((np.array([x, y, z]), np.array([dx, dy, dz])))

def calculate_path_length_for_ray(theta, phi, detector_position):
    """
    Calculates the path length of a single ray through each voxel it intersects.
    Uses an optimized voxel traversal algorithm (Amanatides-Woo).

    Args:
        theta (float): Polar angle of the ray direction (radians).
        phi (float): Azimuthal angle of the ray direction (radians).
        detector_position (np.ndarray): The starting position of the ray.

    Returns:
        scipy.sparse.lil_matrix: A sparse row vector where non-zero elements
                                 are the path lengths in the corresponding voxels.
    """
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Avoid division by zero by replacing 0 with a very small number
    direction[direction == 0] = 1e-10
    
    # Calculate the intersection points of the ray with the ROI bounding box
    t_min = (np.array([x_begin, y_begin, z_begin]) - detector_position) / direction
    t_max = (np.array([x_end, y_end, z_end]) - detector_position) / direction
    t_enter = np.max(np.minimum(t_min, t_max))
    t_exit = np.min(np.maximum(t_min, t_max))
    
    # If the ray does not intersect the ROI, return an empty vector
    if t_enter >= t_exit or t_exit <= 0:
        return lil_matrix((1, N))

    # Start tracing from the entry point
    current_pos = detector_position + max(0, t_enter) * direction
    
    lengths = lil_matrix((1, N))
    
    # Initial voxel coordinates
    voxel_coord = np.floor((current_pos - np.array([x_begin, y_begin, z_begin])) / voxel_size).astype(int)
    voxel_coord = np.clip(voxel_coord, 0, np.array([x_voxels-1, y_voxels-1, z_voxels-1]))
    
    # Voxel traversal parameters
    step = np.sign(direction).astype(int)
    t_delta = voxel_size / np.abs(direction)
    next_boundary_t = ((voxel_coord + (step > 0)) * voxel_size + np.array([x_begin, y_begin, z_begin]) - current_pos) / direction
    
    prev_t = max(0, t_enter)

    # Traverse the grid until the ray exits the ROI
    while (0 <= voxel_coord[0] < x_voxels and 
           0 <= voxel_coord[1] < y_voxels and 
           0 <= voxel_coord[2] < z_voxels):
        
        # Determine which voxel boundary is crossed next
        axis = np.argmin(next_boundary_t)
        t_next = next_boundary_t[axis]

        # Calculate path length in the current voxel
        path_length = t_next - prev_t
        
        # Store the path length if it's positive
        if path_length > 0:
            idx = voxel_indices[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
            lengths[0, idx] = path_length
        
        prev_t = t_next
        
        # Move to the next voxel
        voxel_coord[axis] += step[axis]
        next_boundary_t[axis] += t_delta[axis]

    return lengths

# --- Main Matrix Calculation ---

# Define the angular dimensions of the detector's field of view
# 121 vertical steps, 101 horizontal steps
dtheta_range = np.linspace(-pi / 180 * 30, pi / 180 * 30, 121)
dphi_range = np.linspace(-pi / 180 * 25, pi / 180 * 25, 101)
num_rays_per_detector = len(dtheta_range) * len(dphi_range)

# Initialize the full path-length matrix L
L = lil_matrix((len(detectors) * num_rays_per_detector, N))
print(f"Calculating L matrix with shape: {L.shape}")

# Loop through each detector and each ray in its field of view
for det_idx, (det_pos, det_dir) in enumerate(detectors):
    print(f"Processing Detector {det_idx + 1}/{len(detectors)}...")
    
    # Central direction angles of the detector
    theta0 = np.arccos(det_dir[2] / np.linalg.norm(det_dir)) # Should be approx 60 degrees
    phi0 = np.arctan2(det_dir[1], det_dir[0])
    
    ray_idx = 0
    for dtheta in dtheta_range:
        for dphi in dphi_range:
            theta = theta0 + dtheta
            phi = phi0 + dphi
            
            # Calculate path lengths for this specific ray
            lengths = calculate_path_length_for_ray(theta, phi, det_pos)
            
            # Assign the resulting row vector to the correct position in the large L matrix
            L[det_idx * num_rays_per_detector + ray_idx, :] = lengths
            ray_idx += 1

# --- Save Results ---

output_filename = f'path_lengths_matrix_8det_{voxel_size}m'

# Save the sparse matrix in compressed .npz format (recommended)
print(f"\nSaving matrix to {output_filename}.npz ...")
save_npz(f'{output_filename}.npz', L.tocsr())
print("... .npz save complete.")

# Optional: Save in a human-readable text format (can be very large!)
# print(f"Saving matrix to {output_filename}.txt ...")
# L_coo = L.tocoo()
# L_array = np.vstack((L_coo.row, L_coo.col, L_coo.data)).T
# np.savetxt(f'{output_filename}.txt', L_array, fmt='%d %d %.6f')
# print("... .txt save complete.")