# reconstruction/optimizers.py

import time
import os
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator

import config
from .common import apply_constraints
from utils import plotting

# --- Shared Loss and Gradient Functions ---

def _loss_function(P, L, D):
    """Least squares loss function."""
    residual = L.dot(P) - D
    return np.sum(residual**2)

def _gradient(P, L, D):
    """Gradient of the least squares loss function."""
    residual = L.dot(P) - D
    return 2 * L.T.dot(residual)

# --- L-BFGS Implementation ---

class LBFGSCallback:
    """Callback class to manage state for L-BFGS optimizer."""
    def __init__(self, voxel_size, algo_name="LBFGS"):
        self.iteration_count = 0
        self.voxel_size = voxel_size
        self.algo_name = algo_name
        self.start_time = time.time()
        self.algo_output_dir = os.path.join(config.OUTPUT_ROOT_DIR, self.algo_name, f"voxel_{self.voxel_size}")
        if not os.path.exists(self.algo_output_dir):
            os.makedirs(self.algo_output_dir)
        # Clear previous runtime log
        open(os.path.join(self.algo_output_dir, "RunTime.txt"), "w").close()

    def __call__(self, P):
        self.iteration_count += 1
        iteration_time = time.time() - self.start_time
        
        with open(os.path.join(self.algo_output_dir, "RunTime.txt"), "a") as f:
            f.write(f"Iteration {self.iteration_count}: {iteration_time:.6f} seconds\n")

        if self.iteration_count % config.OPTIMIZER_SAVE_INTERVAL == 0:
            print(f"{self.algo_name}: Iteration {self.iteration_count}: Saving plots...")
            # Note: Scipy's L-BFGS-B handles bounds, but geometric constraints must be applied manually
            P_constrained = apply_constraints(P, self.voxel_size)
            plotting.save_plots_and_data(P_constrained, self.iteration_count, self.algo_name, self.voxel_size)

def run_lbfgs(L, D, voxel_size):
    """Executes the L-BFGS optimization algorithm."""
    start_time_total = time.time()
    
    N = L.shape[1]
    L_linop = LinearOperator((L.shape[0], N), matvec=L.dot, rmatvec=L.T.dot)
    bounds = [(0, 7) for _ in range(N)]
    
    callback_handler = LBFGSCallback(voxel_size, "LBFGS")
    
    result = minimize(
        fun=_loss_function,
        x0=np.zeros(N),
        args=(L_linop, D),
        method='L-BFGS-B',
        jac=_gradient,
        bounds=bounds,
        callback=callback_handler,
        options={'maxiter': config.OPTIMIZER_ITERATIONS, 'disp': False}
    )
    
    P_final = apply_constraints(result.x, voxel_size)
    plotting.save_plots_and_data(P_final, callback_handler.iteration_count, "LBFGS", voxel_size)
    
    total_duration = time.time() - start_time_total
    print(f"L-BFGS Total Time: {total_duration:.2f} seconds for Voxel Size {voxel_size}")
    return P_final

# --- Trust-Region Implementation ---

class TRCallback:
    """Callback class to manage state for Trust-Region optimizer."""
    def __init__(self, voxel_size, algo_name="TR"):
        # Same structure as LBFGSCallback
        self.iteration_count = 0
        self.voxel_size = voxel_size
        self.algo_name = algo_name
        self.start_time = time.time()
        self.algo_output_dir = os.path.join(config.OUTPUT_ROOT_DIR, self.algo_name, f"voxel_{self.voxel_size}")
        if not os.path.exists(self.algo_output_dir):
            os.makedirs(self.algo_output_dir)
        open(os.path.join(self.algo_output_dir, "RunTime.txt"), "w").close()

    def __call__(self, P, state): # Note the different signature for trust-constr
        self.iteration_count += 1
        iteration_time = time.time() - self.start_time

        with open(os.path.join(self.algo_output_dir, "RunTime.txt"), "a") as f:
            f.write(f"Iteration {self.iteration_count}: {iteration_time:.6f} seconds\n")
            
        if self.iteration_count % config.OPTIMIZER_SAVE_INTERVAL == 0:
            print(f"{self.algo_name}: Iteration {self.iteration_count}: Saving plots...")
            P_constrained = apply_constraints(P, self.voxel_size)
            plotting.save_plots_and_data(P_constrained, self.iteration_count, self.algo_name, self.voxel_size)

def run_tr(L, D, voxel_size):
    """Executes the Trust-Region optimization algorithm."""
    start_time_total = time.time()
    
    N = L.shape[1]
    L_linop = LinearOperator((L.shape[0], N), matvec=L.dot, rmatvec=L.T.dot)
    bounds = [(0, 7) for _ in range(N)]
    
    callback_handler = TRCallback(voxel_size, "TR")
    
    result = minimize(
        fun=_loss_function,
        x0=np.zeros(N),
        args=(L_linop, D),
        method='trust-constr',
        jac=_gradient,
        bounds=bounds,
        callback=callback_handler,
        options={'maxiter': config.OPTIMIZER_ITERATIONS, 'disp': False}
    )
    
    P_final = apply_constraints(result.x, voxel_size)
    plotting.save_plots_and_data(P_final, callback_handler.iteration_count, "TR", voxel_size)

    total_duration = time.time() - start_time_total
    print(f"Trust-Region Total Time: {total_duration:.2f} seconds for Voxel Size {voxel_size}")
    return P_final