# config.py

import numpy as np

# --- Reconstruction Parameters ---

# Map voxel sizes to their corresponding path-length matrix filenames
# Assumes these files are in the 'data/' directory
VOXEL_SIZES_AND_FILES = {
    1.0: 'data/path_lengths_matrix_8det_1m.npz',
    0.5: 'data/path_lengths_matrix_8det_0.5m.npz',
    0.2: 'data/path_lengths_matrix_8det_0.2m.npz',
    0.1: 'data/path_lengths_matrix_8det_0.1m.npz'
}

# Path to the opacity matrix file (muon transmittance data)
OPACITY_MATRIX_FILE = 'data/selected_opacity_matrix.txt'

# --- Region of Interest (ROI) Definition ---
# Physical dimensions of the volume to be reconstructed (in meters)
ROI_X = (-10, 10)
ROI_Y = (-10, 10)
ROI_Z = (0, 50)

# --- Output Configuration ---

# Root directory for saving all reconstruction results
OUTPUT_ROOT_DIR = "results"

# --- Algorithm Hyperparameters ---

# SART Algorithm
SART_ITERATIONS = 30
SART_RELAXATION = 0.05
SART_SAVE_INTERVAL = 2

# EM Algorithm
EM_ITERATIONS = 30
EM_TV_WEIGHT = 0.1      # Weight for Total Variation regularization (smoothing)
EM_UPDATE_STEP = 0.3    # Step size limit to stabilize convergence

# L-BFGS and Trust-Region Algorithms
OPTIMIZER_ITERATIONS = 30
OPTIMIZER_SAVE_INTERVAL = 2

# --- Data Loading ---

def get_opacity_vector():
    """Loads the opacity matrix and converts it to a 1D absorption vector."""
    opacity_matrix = np.loadtxt(OPACITY_MATRIX_FILE)
    # Flatten the 2D matrix into a 1D vector and convert percentage to fraction
    opacity_vector = opacity_matrix.flatten() / 100
    return opacity_vector