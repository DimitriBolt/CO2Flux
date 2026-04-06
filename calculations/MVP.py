import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.linalg as la
from scipy.interpolate import CubicSpline

"""
Spatiotemporal Least Squares Inverse Solver (72-Hour LEO Time Series)

INPUTS:
- C_data_matrix: (N_time, 3) array of empirical CO2 concentrations.
- time_array: (N_time,) array of seconds elapsed (e.g., [0, 900, 1800, ...]).
- C_surface_series: (N_time,) array of atmospheric boundary conditions.

OUTPUTS:
- beta_opt: [D_s, S_0, S_1]
- Cov_beta: Parameter covariance matrix bounding the solver error.
"""

# --- 1. PHYSICAL GRID & SETUP ---
L = 1.0                     # Total depth (meters)
N_z = 101                   # Spatial nodes
dz = L / (N_z - 1)
z_grid = np.linspace(0, L, N_z)

# Sensor mapping
sensor_depths = np.array([0.05, 0.20, 0.50])
sensor_indices = np.array([int(depth / dz) for depth in sensor_depths])

# --- 2. MOCK 72-HOUR DATA GENERATION (For Prototype Testing) ---
dt = 900.0                  # 15-minute polling rate (seconds)
N_time = 288                # 72 hours * 4 polls/hour
time_array = np.arange(0, N_time * dt, dt)

# Mocking a steady surface boundary of ~420 ppm
C_surface_series = np.full(N_time, 420.0)

# Mocking the 72-hour data matrix (N_time x 3 sensors)
# In reality, you will load this from your clean 72-hour CSV slice
C_data_matrix = np.zeros((N_time, 3))
C_data_matrix[:, 0] = 4985.0 + 10 * np.sin(2 * np.pi * time_array / 86400) # 5 cm
C_data_matrix[:, 1] = 6880.0 + 5  * np.sin(2 * np.pi * time_array / 86400) # 20 cm
C_data_matrix[:, 2] = 6500.0                                               # 50 cm


# --- 3. THE TRANSIENT FORWARD MODEL ---
def forward_solver_transient(beta):
    """
    Solves the 1D Diffusion-Reaction PDE over the full 72 hours using Crank-Nicolson.
    """
    D_s, S_0, S_1 = beta
    
    # Precompute the biological source term (assuming isothermal for this prototype)
    S_bio = S_0 + S_1 * z_grid
    
    # Crank-Nicolson Fourier Number (alpha)
    alpha = (D_s * dt) / (2.0 * dz**2)
    
    # Matrix A (Implicit Future State)
    A_main = (1.0 + 2.0 * alpha) * np.ones(N_z)
    A_off = -alpha * np.ones(N_z - 1)
    A = diags([A_off, A_main, A_off], [-1, 0, 1], format='csr')
    
    # Matrix B (Explicit Current State)
    B_main = (1.0 - 2.0 * alpha) * np.ones(N_z)
    B_off = alpha * np.ones(N_z - 1)
    B = diags([B_off, B_main, B_off], [-1, 0, 1], format='csr')
    
    # Apply Boundaries to Matrix A (Constant across time)
    A.data[0:3] = 0.0 ; A[0, 0] = 1.0       # Top Dirichlet
    A[N_z-1, N_z-1] = 1.0 ; A[N_z-1, N_z-2] = -1.0 # Bottom Neumann (dC/dz = 0)
    
    # Initialize concentration vector C at t=0
    # Interpolate the first row of your empirical data to fill the 101 nodes
    C_current = np.zeros(N_z)
    cs = CubicSpline([0.0, 0.05, 0.20, 0.50, 1.0], 
                     [C_surface_series[0], C_data_matrix[0,0], C_data_matrix[0,1], C_data_matrix[0,2], C_data_matrix[0,2]])
    C_current[:] = cs(z_grid)
    
    # Storage for the predictions at sensor locations
    C_model_sensors = np.zeros((N_time, 3))
    C_model_sensors[0, :] = C_current[sensor_indices]
    
    # The Time-Stepping Loop
    for n in range(1, N_time):
        # 1. Compute explicit right-hand side: B * C^n
        d = B.dot(C_current)
        
        # 2. Add the source term (integrated over the time step)
        d += S_bio * dt
        
        # 3. Enforce boundary conditions on the right-hand vector
        d[0] = C_surface_series[n]  # Top Dirichlet boundary at time n
        d[N_z-1] = 0.0              # Bottom Neumann boundary (matches Matrix A logic)
        
        # 4. Solve the linear system A * C^{n+1} = d
        C_next = spsolve(A, d)
        
        # 5. Store the predictions at the exact physical sensor nodes
        C_model_sensors[n, :] = C_next[sensor_indices]
        
        # 6. Step forward in time
        C_current = C_next
        
    return C_model_sensors

# --- 4. THE OPTIMIZATION LOOP ---
def cost_function(beta):
    """
    Calculates the residual distance across all 72 hours.
    Flattens the 2D spatiotemporal matrix into a 1D vector for Least Squares.
    """
    C_model_sensors = forward_solver_transient(beta)
    
    # Residual matrix: (N_time x 3)
    residual_matrix = C_model_sensors - C_data_matrix
    
    # Flatten to a 1D array (Length: 864)
    return residual_matrix.flatten()

# --- 5. EXECUTION & ERROR BOUNDING ---
if __name__ == "__main__":
    beta_initial = np.array([4.5e-6, 5.0, -2.0])
    lower_bounds = [1.0e-7, 0.0, -100.0]
    upper_bounds = [1.0e-5, 20.0, 0.0] 
    
    print(f"Projecting {N_time} time steps onto mathematical subspace...")
    
    result = least_squares(
        cost_function, 
        x0=beta_initial, 
        bounds=(lower_bounds, upper_bounds),
        method='trf',      
        diff_step=1e-8     
    )
    
    beta_opt = result.x
    J = result.jac
    
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimized D_s : {beta_opt[0]:.2e} m^2/s")
    print(f"Optimized S_0 : {beta_opt[1]:.4f} units")
    print(f"Optimized S_1 : {beta_opt[2]:.4f} units")
    
    # --- RIGOROUS COVARIANCE CALCULATION (THE GABITOV AUDIT) ---
    residuals = result.fun
    degrees_of_freedom = len(residuals) - len(beta_opt) # 864 - 3 = 861
    
    sigma_sq = np.sum(residuals**2) / degrees_of_freedom
    Cov_beta = sigma_sq * la.pinv(J.T @ J)
    
    print(f"\nDegrees of Freedom: {degrees_of_freedom}")
    print("=== PARAMETER UNCERTAINTY (COVARIANCE DIAGONAL) ===")
    print(f"Variance D_s : {Cov_beta[0,0]:.2e}")
    print(f"Variance S_0 : {Cov_beta[1,1]:.2e}")