"""Test the heat-1d-mixed problem with mixed positive/negative forcing."""

import jax.numpy as jnp
from pde_opt.problems import get_problem
from pde_opt.utils.benchmarking import benchmark_single_config

# Create the problem
problem = get_problem('heat-1d-mixed')

print("Testing heat-1d-mixed problem")
print("=" * 80)
print()
print("Problem description:")
print(f"  Solution: u(x,t) = sin(πx)·(3t - 4t²)")
print(f"  Initial condition: u₀(x) = 0")
print(f"  Forcing: f(x,t) = sin(πx)·[(3 - 8t) + π²(3t - 4t²)]")
print()

# Analyze forcing at different times
print("Forcing behavior at x=0.5 (where sin(πx) = 1):")
print("-" * 80)
x_test = jnp.array([0.5])
for t_val in [0.0, 0.25, 0.5, 0.65, 0.75, 1.0]:
    t_test = jnp.array([t_val])
    f_val = problem.source_term(x_test, t_test)[0, 0]
    sign = "POSITIVE" if f_val > 0 else "NEGATIVE"
    print(f"  t = {t_val:.2f}: f = {f_val:+7.3f} ({sign})")
print()

# Test solver accuracy with different grid sizes
print("Testing solver accuracy:")
print("-" * 80)

grid_configs = [
    {'nx': 50, 'nt': 100},
    {'nx': 100, 'nt': 150},
    {'nx': 150, 'nt': 200},
]

for grid_params in grid_configs:
    result = benchmark_single_config(
        'heat-1d-mixed',
        'crank-nicolson',
        grid_params
    )

    nx, nt = grid_params['nx'], grid_params['nt']
    rel_error = result['metrics']['rel_error_pct']
    mse = result['metrics']['mse']
    is_valid = result['metrics']['is_valid']

    print(f"Grid: nx={nx:3d}, nt={nt:3d}  →  Error: {rel_error:5.2f}%  "
          f"MSE: {mse:.2e}  Status: {'✓ PASS' if is_valid else '✗ FAIL'}")

print()
print("This problem is suitable for testing ReLU enforcement on forcing terms")
print("because the true forcing has both positive (early times) and negative")
print("(late times) components with meaningful magnitudes.")
