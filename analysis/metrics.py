# analysis/metrics.py

import os
import numpy as np
import pandas as pd
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def load_data(file_path, x_range, z_range, num_rows, num_cols):
    """Loads data from a text file and defines coordinate axes."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    data = np.loadtxt(file_path)
    x = np.linspace(x_range[0], x_range[1], num_cols)
    z = np.linspace(z_range[0], z_range[1], num_rows)
    return data, x, z

def calculate_metrics(data, regions, x, z):
    """
    Calculates image quality metrics: Kappa, Gradient Clarity, and G/2sigma.

    Args:
        data (np.ndarray): The 2D image data.
        regions (list): A list of polygons defining regions for Kappa calculation.
        x (np.ndarray): The x-axis coordinates.
        z (np.ndarray): The z-axis coordinates.

    Returns:
        tuple: Contains kappa, gradient_clarity, G_2sigma, gradient_magnitude,
               and the mask of the intersection line.
    """
    num_rows, num_cols = data.shape
    x_pixel_size = (x[-1] - x[0]) / (num_cols - 1)
    z_pixel_size = (z[-1] - z[0]) / (num_rows - 1)

    def to_pixel(coord):
        """Converts physical coordinates to pixel indices."""
        x_pixel = round((coord[0] - x[0]) / x_pixel_size)
        z_pixel = round((coord[1] - z[0]) / z_pixel_size)
        return int(x_pixel), int(z_pixel)

    # This can be slow. For performance, consider libraries like shapely or scikit-image
    def is_inside_polygon(polygon, x_pixel, z_pixel):
        """Checks if a point is inside a polygon using the ray casting algorithm."""
        n = len(polygon)
        inside = False
        for i in range(n):
            p1x, p1z = polygon[i]
            p2x, p2z = polygon[(i + 1) % n]
            if ((p1z > z_pixel) != (p2z > z_pixel)) and \
               (x_pixel < (p2x - p1x) * (z_pixel - p1z) / (p2z - p1z + 1e-9) + p1x):
                inside = not inside
        return inside

    # Convert polygon vertices from physical coordinates to pixel indices
    pixel_regions = [[to_pixel(pt) for pt in region] for region in regions]
    
    # Create boolean masks for each defined region
    region_masks = [np.zeros_like(data, dtype=bool) for _ in regions]
    for r in range(num_rows):
        for c in range(num_cols):
            for i, polygon in enumerate(pixel_regions):
                if is_inside_polygon(polygon, c, r):
                    region_masks[i][r, c] = True

    # --- Kappa Calculation ---
    region_values = [data[mask] for mask in region_masks]
    means = [np.mean(vals) if vals.size > 0 else 0 for vals in region_values]
    std_devs = [np.std(vals) if vals.size > 0 else 0 for vals in region_values]
    
    denominator = np.sqrt(std_devs[0]**2 + std_devs[1]**2)
    kappa = abs(means[0] - means[1]) / denominator if denominator > 1e-9 else 0
    
    # --- Gradient Clarity and G/2sigma Calculation ---
    # Sobel gradient calculation
    Gx = sobel(data, axis=1)
    Gz = sobel(data, axis=0)
    gradient_magnitude = np.sqrt(Gx**2 + Gz**2)
    
    # Define the line separating regions to measure gradient clarity
    line_points_physical = [
        (-3.79916, 11.82345), (-0.70809, 23.42705),
        (0.70809, 23.42705), (3.79916, 11.82345)
    ]
    line_points_pixel = [to_pixel(p) for p in line_points_physical]

    # Use Bresenham's algorithm to get pixels along the line segments
    intersection_line_mask = np.zeros_like(data, dtype=bool)
    for i in range(len(line_points_pixel) - 1):
        x1, z1 = line_points_pixel[i]
        x2, z2 = line_points_pixel[i+1]
        dx, dy = abs(x2 - x1), -abs(z2 - z1)
        sx, sy = (1, -1)[x1 > x2], (1, -1)[z1 > z2]
        err = dx + dy
        while True:
            if 0 <= z1 < num_rows and 0 <= x1 < num_cols:
                intersection_line_mask[z1, x1] = True
            if x1 == x2 and z1 == z2: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x1 += sx
            if e2 <= dx: err += dx; z1 += sy
    
    intersection_gradient_values = gradient_magnitude[intersection_line_mask]
    gradient_clarity = np.mean(intersection_gradient_values) if intersection_gradient_values.size > 0 else 0
        
    # Calculate G/2σ using standard deviation of a larger, uniform region
    x_sigma_range, z_sigma_range = (-8, 8), (0, 42)
    x_start_pix, _ = to_pixel((x_sigma_range[0], 0))
    x_end_pix, _ = to_pixel((x_sigma_range[1], 0))
    _, z_start_pix = to_pixel((0, z_sigma_range[0]))
    _, z_end_pix = to_pixel((0, z_sigma_range[1]))
    
    sigma_region_data = data[z_start_pix:z_end_pix, x_start_pix:x_end_pix]
    sigma = np.std(sigma_region_data) if sigma_region_data.size > 0 else 0
    G_2sigma = gradient_clarity / (2 * sigma) if sigma > 1e-9 else 0
    
    return kappa, gradient_clarity, G_2sigma, gradient_magnitude, intersection_line_mask

def plot_and_save_analysis(data, x, z, regions, gradient_magnitude, intersection_mask, output_path):
    """Plots the data, gradients, and defined regions, and saves the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original Data with Regions and Line
    im0 = axes[0].imshow(data, extent=[x[0], x[-1], z[0], z[-1]], cmap="jet", origin='lower')
    axes[0].set_title("Data with Analysis Regions")
    axes[0].set_xlabel("X-axis (m)")
    axes[0].set_ylabel("Z-axis (m)")
    fig.colorbar(im0, ax=axes[0], label="Density")
    
    colors = ['cyan', 'magenta']
    for idx, region in enumerate(regions):
        polygon = Polygon(region, closed=True, edgecolor=colors[idx], facecolor='none', lw=2, label=f'Region {idx+1}')
        axes[0].add_patch(polygon)
    
    z_coords, x_coords = np.where(intersection_mask)
    x_phys = x[0] + x_coords * (x[-1] - x[0]) / (data.shape[1] - 1)
    z_phys = z[0] + z_coords * (z[-1] - z[0]) / (data.shape[0] - 1)
    axes[0].scatter(x_phys, z_phys, color='lime', s=1, label='Interface Line')
    axes[0].legend()

    # Plot 2: Gradient Magnitude
    im1 = axes[1].imshow(gradient_magnitude, extent=[x[0], x[-1], z[0], z[-1]], cmap="inferno", origin='lower')
    axes[1].set_title("Gradient Magnitude")
    axes[1].set_xlabel("X-axis (m)")
    axes[1].set_ylabel("Z-axis (m)")
    fig.colorbar(im1, ax=axes[1], label="Gradient")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def run_analysis(file_list, save_path):
    """Main function to process a list of files and calculate metrics."""
    x_range, z_range = [-10, 10], [0, 50]
    regions = [
        # Region 1 (inner trapezoid)
        [(-0.70809, 23.42705), (0.70809, 23.42705), (3.79916, 11.82345), (-3.79916, 11.82345)],
        # Region 2 (outer polygon)
        [(-0.70809, 23.42705), (0.70809, 23.42705), (3.79916, 11.82345), (5.79916, 11.82345),
         (2, 26.08505), (-2, 26.08505), (-5.79916, 11.82345), (-3.79916, 11.82345)]
    ]

    w_kappa, w_G = 1, 0.5
    metrics_data = []
    
    for file_info in file_list:
        file_path = file_info["path"]
        algorithm = file_info["algorithm"]
        voxel_size = file_info["voxel_size"]

        num_rows = int((z_range[1] - z_range[0]) / voxel_size) + 1
        num_cols = int((x_range[1] - x_range[0]) / voxel_size) + 1
        
        try:
            data, x, z = load_data(file_path, x_range, z_range, num_rows, num_cols)
            
            kappa, grad_clarity, g_2sigma, grad_mag, intersection_mask = calculate_metrics(data, regions, x, z)
            
            relative_kappa = kappa / voxel_size
            q_factor = (relative_kappa ** w_kappa) * (g_2sigma ** w_G)
            
            metrics_data.append({
                "File": os.path.basename(file_path), "Algorithm": algorithm, "Voxel Size": voxel_size,
                "Kappa": kappa, "Gradient Clarity": grad_clarity, "G/2σ": g_2sigma,
                "Relative Kappa": relative_kappa, "Q Factor": q_factor
            })
            
            # Save analysis plots
            plot_output_path = file_path.replace(".txt", "_analysis.png")
            plot_and_save_analysis(data, x, z, regions, grad_mag, intersection_mask, plot_output_path)
            print(f"Generated analysis plot for {os.path.basename(file_path)}")

        except FileNotFoundError as e:
            print(e)
            continue

    results_df = pd.DataFrame(metrics_data)
    results_df.to_csv(save_path, index=False)
    print(f"\nAnalysis complete. Results saved to: {save_path}")
    return results_df

# --- Example of how to run this analysis script ---
if __name__ == "__main__":
    # This block allows the script to be run directly for testing or use.
    # It assumes the results are in a directory structure like:
    # results/SART/voxel_1.0/x_0/density_slice_iter_30.txt
    
    print("Running analysis script...")
    
    # Define the files to be analyzed
    files_to_process = []
    algorithms = ["EM", "SART", "LBFGS", "TR"]
    voxel_sizes = [1.0, 0.5, 0.2, 0.1]
    
    for alg in algorithms:
        # Assuming different final iteration numbers for different algorithms
        iter_num = 30 
        for voxel in voxel_sizes:
            # Construct the full path to the result file
            path = os.path.join("..", "results", alg, f"voxel_{voxel}", "x_0", f"density_slice_iter_{iter_num}.txt")
            files_to_process.append({"path": path, "algorithm": alg, "voxel_size": voxel})
            
    output_csv_path = "image_quality_metrics.csv"
    results = run_analysis(files_to_process, output_csv_path)
    
    print("\n--- Results Summary ---")
    print(results)