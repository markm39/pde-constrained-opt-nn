"""Test the cossinsin problem with fixed IC."""

from pde_opt.utils.benchmarking import benchmark_single_config

# Test with different grid sizes
grid_configs = [
    {'nx': 20, 'ny': 20, 'nt': 100},
    {'nx': 40, 'ny': 40, 'nt': 100},
    {'nx': 60, 'ny': 60, 'nt': 100},
    {'nx': 80, 'ny': 80, 'nt': 100},
]

print("Testing linear-heat-2d[prob=cossinsin] with corrected IC (u_0 = 0)")
print("=" * 80)

for grid_params in grid_configs:
    result = benchmark_single_config(
        'linear-heat-2d',
        'crank-nicolson',
        grid_params,
        prob='cossinsin'
    )

    nx, ny, nt = grid_params['nx'], grid_params['ny'], grid_params['nt']
    rel_error = result['metrics']['rel_error_pct']
    mse = result['metrics']['mse']
    is_valid = result['metrics']['is_valid']

    print(f"\nGrid: nx={nx}, ny={ny}, nt={nt}")
    print(f"Relative error: {rel_error:.2f}%")
    print(f"MSE: {mse:.2e}")
    print(f"Status: {'✓ PASS' if is_valid else '✗ FAIL'}")
