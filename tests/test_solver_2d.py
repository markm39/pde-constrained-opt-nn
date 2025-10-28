"""Quick test to verify 2D linear heat solver with ground truth forcing."""
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax
from pde_opt.problems import get_problem
from pde_opt.solvers import get_solver

print("Testing 2D Linear Heat Solver with cossinsin problem...")
print("="*60)

# Setup problem and solver
problem = get_problem('linear-heat-2d', prob='cossinsin')  # Test cossinsin with corrected formula
# Need very fine grid for < 1% solver error for accurate forcing recovery
solver = get_solver('nonlinear-heat-2d', 'crank-nicolson', nx=100, ny=100, nt=200)

x_grid = solver.x_grid
y_grid = solver.y_grid
t_grid = solver.t_grid
k = solver.k
n_spatial = solver.nx * solver.ny

# Get ground truth
u_target_3d = jnp.stack([problem.analytical_solution(x_grid, y_grid, t)
                          for t in t_grid], axis=-1)
u_target = u_target_3d.reshape(n_spatial, solver.nt)

u0_2d = problem.initial_condition(x_grid, y_grid)
u0 = u0_2d.flatten()

# Get TRUE forcing
X, Y = jnp.meshgrid(x_grid, y_grid, indexing='ij')
f_true_3d = jnp.stack([problem.source_term(x_grid, y_grid, t)
                        for t in t_grid], axis=-1)  # (nx, ny, nt)
f_true = f_true_3d.reshape(n_spatial, solver.nt)  # (nx*ny, nt)

print(f"u0 shape: {u0.shape}, min/max: {jnp.min(u0):.6f}/{jnp.max(u0):.6f}")
print(f"u_target shape: {u_target.shape}, min/max: {jnp.min(u_target):.6f}/{jnp.max(u_target):.6f}")
print(f"f_true shape: {f_true.shape}, min/max: {jnp.min(f_true):.6f}/{jnp.max(f_true):.6f}")

# Setup Crank-Nicolson solver
# Heat equation: du/dt = Δu + f  where Δ is the Laplacian
# solver.K represents -Δ (negative Laplacian, has negative diagonal)
# So: du/dt = -K·u + f
# Crank-Nicolson: (I + k/2 * (-K)) u_{n+1} = (I - k/2 * (-K)) u_n + k * f_n
# Which is: (I - k/2 * K) u_{n+1} = (I + k/2 * K) u_n + k * f_n
A = solver.K  # -Δ (negative Laplacian)
A_cn = jnp.eye(n_spatial) - (k/2.0) * A

print(f"\nMatrix A_cn min/max: {jnp.min(A_cn):.6f}/{jnp.max(A_cn):.6f}")

# Check if positive definite
try:
    L_cn = jnp.linalg.cholesky(A_cn)
    print("✓ Cholesky decomposition successful (matrix is positive definite)")
except:
    print("✗ Cholesky failed (matrix is NOT positive definite)")
    exit(1)

def chol_solve(L, b):
    y = jsp.solve_triangular(L, b, lower=True)
    u = jsp.solve_triangular(L.T, y, lower=False)
    return u

# Time-stepping with TRUE forcing
def forward_with_true_forcing(u0, f_true):
    def step(u_prev, i):
        f_n = f_true[:, i]
        # Crank-Nicolson RHS: (I + k/2 * K) u_n + k * f_n
        rhs = (jnp.eye(n_spatial) + (k/2.0) * A) @ u_prev + k * f_n
        u_next = chol_solve(L_cn, rhs)
        return u_next, u_next

    indices = jnp.arange(solver.nt)
    _, U_seq = lax.scan(step, u0, indices)
    return U_seq  # (nt, nx*ny)

print("\nSolving with TRUE forcing...")
U_pred = forward_with_true_forcing(u0, f_true)

print(f"U_pred shape: {U_pred.shape}, min/max: {jnp.min(U_pred):.6f}/{jnp.max(U_pred):.6f}")
print(f"Has NaN in U_pred? {jnp.any(jnp.isnan(U_pred))}")

# Compare with target
error = U_pred.T - u_target
mse = jnp.mean(error**2)
rel_error = jnp.linalg.norm(error) / jnp.linalg.norm(u_target)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"  MSE: {mse:.6e}")
print(f"  Relative L2 error: {rel_error:.6e} ({100*rel_error:.2f}%)")

if rel_error < 1e-3:
    print("  ✓ SOLVER IS CORRECT (error < 0.1%)")
elif rel_error < 1e-2:
    print("  ⚠ SOLVER HAS SMALL ERROR (error < 1%)")
else:
    print("  ✗ SOLVER HAS LARGE ERROR")
print(f"{'='*60}")
