"""
Benchmarking utilities for PDE solvers.

This module provides functions to systematically benchmark solver accuracy
across different grid configurations and problem types.
"""

import jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Optional
from pde_opt.problems import get_problem
from pde_opt.examples import get_example
from pde_opt.utils.solver_validation import validate_solver


def compute_minimum_grid_requirements(
    problem_name: str,
    discretization: str,
    problem_kwargs: Dict[str, Any]
) -> Dict[str, int]:
    """
    Compute minimum grid requirements to meet accuracy targets.

    For oscillatory problems, spatial resolution must satisfy Nyquist criterion:
    - Need 10-20 points per wavelength for accurate resolution
    - Wavelength = 1 / n_oscillations
    - Therefore: nx >= 10 * n_oscillations (minimum), 20 * n_oscillations (recommended)

    For Crank-Nicolson, temporal resolution must be sufficient to keep O(k²) errors small.

    Args:
        problem_name: Name of the problem
        discretization: Solver discretization method
        problem_kwargs: Problem parameters (may include n_oscillations, T, etc.)

    Returns:
        Dictionary with minimum grid parameters: {'nx': int, 'nt': int, 'ny': int (if 2D)}
    """
    is_2d = 'heat-2d' in problem_name
    n_oscillations = problem_kwargs.get('n_oscillations', 1)
    T = problem_kwargs.get('T', 1.0)

    # Spatial resolution requirements
    if n_oscillations > 1:
        # Oscillatory problems: need 20 points per wavelength for 1% accuracy
        # Wavelength = 1 / n_oscillations
        min_nx = max(50, 20 * n_oscillations)
    else:
        # Smooth problems: standard grid
        min_nx = 50 if not is_2d else 20

    # Temporal resolution requirements
    if discretization == 'crank-nicolson':
        # Crank-Nicolson is O(k²), but needs finer time steps for oscillatory problems
        # and to keep truncation error < 1%
        if n_oscillations >= 10:
            # High-frequency temporal behavior requires more time steps
            min_nt = max(150, int(200 * T))
        else:
            # Standard: aim for k ≤ 0.01 for 1% accuracy
            min_nt = max(100, int(100 * T))
    else:
        # Backward Euler (FD/FEM) is O(k), needs finer time steps
        min_nt = max(100, int(150 * T))

    # Build requirements
    requirements = {'nx': min_nx, 'nt': min_nt}
    if is_2d:
        requirements['ny'] = min_nx  # Square grids for 2D

    return requirements


def benchmark_single_config(
    problem_name: str,
    discretization: str,
    grid_params: Dict[str, int],
    **problem_kwargs
) -> Dict[str, Any]:
    """
    Benchmark a single problem/solver/grid configuration.

    Args:
        problem_name: Name of the problem (e.g., 'heat-1d')
        discretization: Solver discretization ('fd', 'fem', 'crank-nicolson')
        grid_params: Grid parameters {'nx': int, 'nt': int, 'ny': int (if 2D)}
        **problem_kwargs: Additional problem parameters (e.g., n_oscillations, T)

    Returns:
        Dictionary with benchmark results including:
        - problem_name, problem_params, discretization
        - grid_params, total_dofs
        - metrics: rel_error, rel_error_pct, mse, is_valid
    """
    # Create example instance based on problem type
    example_map = {
        'heat-1d': 'example-3.3',
        'heat-1d-oscillating': 'example-3.3',
        'heat-1d-oscillating-cosine': 'example-3.3',
        'linear-heat-2d': 'example-3.5',
        'nonlinear-heat-2d': 'example-3.6',
    }

    if problem_name not in example_map:
        raise ValueError(f"Unknown problem: {problem_name}. Heat equations only: {list(example_map.keys())}")

    example_name = example_map[problem_name]

    # Create example with specific grid params
    if 'linear-heat-2d' in problem_name:
        # Extract prob variant if present
        prob = problem_kwargs.pop('prob', 'default')
        example = get_example(
            example_name,
            prob=prob,
            regularization=1e-6,
            **problem_kwargs
        )
        problem_kwargs['prob'] = prob  # Restore for result tracking
    else:
        example = get_example(
            example_name,
            problem_name=problem_name,
            regularization=1e-6,
            **problem_kwargs
        )

    # Check grid requirements and warn if inadequate
    min_requirements = compute_minimum_grid_requirements(problem_name, discretization, problem_kwargs)
    grid_warnings = []

    if grid_params['nx'] < min_requirements['nx']:
        grid_warnings.append(
            f"nx={grid_params['nx']} is below minimum recommended {min_requirements['nx']} "
            f"(need ~20 pts/wavelength for n_oscillations={problem_kwargs.get('n_oscillations', 1)})"
        )

    if grid_params['nt'] < min_requirements['nt']:
        grid_warnings.append(
            f"nt={grid_params['nt']} is below minimum recommended {min_requirements['nt']} "
            f"for {discretization} discretization"
        )

    if 'ny' in min_requirements and grid_params.get('ny', 0) < min_requirements['ny']:
        grid_warnings.append(
            f"ny={grid_params.get('ny')} is below minimum recommended {min_requirements['ny']}"
        )

    # Override grid params
    example.grid_params = grid_params
    example.discretization = discretization

    # Run validation
    is_valid, rel_error, details = validate_solver(example, threshold=0.01, verbose=False)

    # Calculate total DOFs
    if 'ny' in grid_params:
        # 2D problem
        total_dofs = grid_params['nx'] * grid_params['ny'] * grid_params['nt']
    else:
        # 1D problem
        total_dofs = grid_params['nx'] * grid_params['nt']

    # Build result
    result = {
        'problem_name': problem_name,
        'problem_params': problem_kwargs.copy(),
        'discretization': discretization,
        'grid_params': grid_params.copy(),
        'min_requirements': min_requirements,
        'grid_warnings': grid_warnings,
        'total_dofs': total_dofs,
        'metrics': {
            'rel_error': rel_error,
            'rel_error_pct': rel_error * 100,
            'mse': details['mse'],
            'is_valid': is_valid,
            'threshold': 0.01,
        }
    }

    return result


def benchmark_grid_refinement(
    problem_name: str,
    discretization: str,
    target_error: float = 0.01,
    max_nx: int = 500,
    **problem_kwargs
) -> List[Dict[str, Any]]:
    """
    Benchmark a problem by progressively refining the grid until target error is achieved.

    Args:
        problem_name: Name of the problem
        discretization: Solver discretization method
        target_error: Target relative L2 error (default: 0.01 = 1%)
        max_nx: Maximum grid size to try (default: 500)
        **problem_kwargs: Additional problem parameters

    Returns:
        List of benchmark results, one for each grid size tried
    """
    results = []

    # Determine if 2D problem
    is_2d = 'heat-2d' in problem_name

    # Compute minimum grid requirements based on problem characteristics
    min_req = compute_minimum_grid_requirements(problem_name, discretization, problem_kwargs)
    min_nx = min_req['nx']
    min_nt = min_req['nt']

    # Starting grid size - use minimum requirements as baseline
    if is_2d:
        # Start from minimum nx, increment by 20
        start_nx = ((min_nx + 19) // 20) * 20  # Round up to nearest 20
        nx_values = list(range(start_nx, min(max_nx, 300) + 1, 20))
        nt = max(min_nt, 100 if problem_kwargs.get('prob') == 'cossinsin' else 50)
    else:
        # Start from minimum nx, increment by 50
        start_nx = ((min_nx + 49) // 50) * 50  # Round up to nearest 50
        nx_values = list(range(start_nx, max_nx + 1, 50))
        nt = max(min_nt, 100)

    for nx in nx_values:
        if is_2d:
            grid_params = {'nx': nx, 'ny': nx, 'nt': nt}
        else:
            grid_params = {'nx': nx, 'nt': nt}

        try:
            result = benchmark_single_config(
                problem_name,
                discretization,
                grid_params,
                **problem_kwargs
            )
            results.append(result)

            # Check if target achieved
            if result['metrics']['rel_error'] < target_error:
                print(f"  ✓ Target error {target_error*100:.1f}% achieved at nx={nx}")
                break
        except Exception as e:
            print(f"  ✗ Failed at nx={nx}: {e}")
            continue

    return results


def benchmark_convergence_study(
    problem_name: str,
    problem_kwargs: Dict[str, Any],
    solvers: List[str],
    spatial_grids: List[Tuple],
    temporal_grids: List[Tuple]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Run convergence study with predefined grid sizes.

    Args:
        problem_name: Name of the problem
        problem_kwargs: Problem parameters
        solvers: List of solver discretizations to test
        spatial_grids: List of (nx, nt) tuples for 1D or (nx, ny, nt) tuples for 2D
        temporal_grids: List of (nx, nt) tuples for 1D or (nx, ny, nt) tuples for 2D

    Returns:
        {'spatial': {solver: [results]}, 'temporal': {solver: [results]}}
    """
    results = {'spatial': {}, 'temporal': {}}
    is_2d = 'heat-2d' in problem_name

    for solver in solvers:
        print(f"\n  Solver: {solver}")

        # Spatial convergence (varying nx/ny, fixed nt)
        print(f"    Spatial convergence...")
        results['spatial'][solver] = []
        for grid in spatial_grids:
            if is_2d:
                nx, ny, nt = grid
                grid_params = {'nx': nx, 'ny': ny, 'nt': nt}
                grid_str = f"nx={nx}, ny={ny}, nt={nt}"
            else:
                nx, nt = grid
                grid_params = {'nx': nx, 'nt': nt}
                grid_str = f"nx={nx}, nt={nt}"

            try:
                result = benchmark_single_config(
                    problem_name,
                    solver,
                    grid_params,
                    **problem_kwargs
                )
                results['spatial'][solver].append(result)
                error_pct = result['metrics']['rel_error_pct']
                print(f"      {grid_str}: {error_pct:.2f}%")
            except Exception as e:
                # Check if it's a memory error
                error_msg = str(e)
                if 'RESOURCE_EXHAUSTED' in error_msg or 'Out of memory' in error_msg:
                    # Extract memory size from error message
                    import re
                    match = re.search(r'allocate (\d+) bytes', error_msg)
                    if match:
                        bytes_requested = int(match.group(1))
                        gb_requested = bytes_requested / (1024**3)
                        print(f"      ✗ Out of memory at {grid_str}: attempted to allocate {gb_requested:.2f} GB")
                    else:
                        print(f"      ✗ Out of memory at {grid_str}")
                    # Add a failed result with memory error flag
                    results['spatial'][solver].append({
                        'problem_name': problem_name,
                        'problem_params': problem_kwargs.copy(),
                        'discretization': solver,
                        'grid_params': grid_params,
                        'error': 'RESOURCE_EXHAUSTED',
                        'error_message': error_msg
                    })
                else:
                    print(f"      ✗ Failed at {grid_str}: {e}")

        # Temporal convergence (fixed nx/ny, varying nt)
        print(f"    Temporal convergence...")
        results['temporal'][solver] = []
        for grid in temporal_grids:
            if is_2d:
                nx, ny, nt = grid
                grid_params = {'nx': nx, 'ny': ny, 'nt': nt}
                grid_str = f"nx={nx}, ny={ny}, nt={nt}"
            else:
                nx, nt = grid
                grid_params = {'nx': nx, 'nt': nt}
                grid_str = f"nx={nx}, nt={nt}"

            try:
                result = benchmark_single_config(
                    problem_name,
                    solver,
                    grid_params,
                    **problem_kwargs
                )
                results['temporal'][solver].append(result)
                error_pct = result['metrics']['rel_error_pct']
                print(f"      {grid_str}: {error_pct:.2f}%")
            except Exception as e:
                # Check if it's a memory error
                error_msg = str(e)
                if 'RESOURCE_EXHAUSTED' in error_msg or 'Out of memory' in error_msg:
                    # Extract memory size from error message
                    import re
                    match = re.search(r'allocate (\d+) bytes', error_msg)
                    if match:
                        bytes_requested = int(match.group(1))
                        gb_requested = bytes_requested / (1024**3)
                        print(f"      ✗ Out of memory at {grid_str}: attempted to allocate {gb_requested:.2f} GB")
                    else:
                        print(f"      ✗ Out of memory at {grid_str}")
                    # Add a failed result with memory error flag
                    results['temporal'][solver].append({
                        'problem_name': problem_name,
                        'problem_params': problem_kwargs.copy(),
                        'discretization': solver,
                        'grid_params': grid_params,
                        'error': 'RESOURCE_EXHAUSTED',
                        'error_message': error_msg
                    })
                else:
                    print(f"      ✗ Failed at {grid_str}: {e}")

    return results


def benchmark_heat_equations(
    target_error: float = 0.01,
    max_nx: int = 500,
    convergence_study: bool = False
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Benchmark all heat equation problems with their applicable solvers.

    Args:
        target_error: Target relative L2 error (default: 0.01 = 1%)
        max_nx: Maximum grid size to try (default: 500)
        convergence_study: If True, run convergence study with predefined grids

    Returns:
        Nested dictionary: {problem_name: {discretization: [results]}}
        Or if convergence_study: {problem_name: {'spatial': ..., 'temporal': ...}}
    """
    # Define problems and their configurations
    # Note: FEM is not currently supported by Example33, so we only use fd and crank-nicolson
    problems = [
        ('heat-1d', {}, ['fd', 'crank-nicolson']),
        ('heat-1d-oscillating', {'n_oscillations': 10}, ['fd', 'crank-nicolson']),
        ('heat-1d-oscillating-cosine', {'n_oscillations': 10, 'T': 0.5}, ['fd', 'crank-nicolson']),
        ('linear-heat-2d', {'prob': 'default'}, ['crank-nicolson']),
        ('linear-heat-2d', {'prob': 'cossinsin'}, ['crank-nicolson']),
        ('nonlinear-heat-2d', {}, ['crank-nicolson']),
    ]

    all_results = {}

    for problem_name, problem_kwargs, solvers in problems:
        # Create unique key for problems with variants
        if problem_kwargs:
            param_str = '_'.join(f"{k}={v}" for k, v in problem_kwargs.items())
            problem_key = f"{problem_name}[{param_str}]"
        else:
            problem_key = problem_name

        print(f"\nBenchmarking: {problem_key}")
        print("=" * 60)

        # Determine if 1D or 2D problem
        is_2d = 'heat-2d' in problem_name

        if convergence_study:
            # Convergence study mode with predefined grids
            if is_2d:
                # 2D grids: (nx, ny, nt)
                # Spatial convergence: vary nx=ny, fix nt
                # Use smaller grids for 2D to avoid memory issues
                spatial_grids = [(n, n, 50) for n in [20, 40, 60, 80, 100, 120]]
                # Temporal convergence: fix nx=ny, vary nt
                temporal_grids = [(60, 60, nt) for nt in [25, 50, 75, 100, 125, 150]]
            else:
                # 1D grids: (nx, nt)
                spatial_grids = [(nx, 200) for nx in [50, 100, 150, 200, 250]]
                temporal_grids = [(150, nt) for nt in [50, 100, 150, 200, 250]]

            try:
                conv_results = benchmark_convergence_study(
                    problem_name,
                    problem_kwargs,
                    solvers,
                    spatial_grids,
                    temporal_grids
                )
                all_results[problem_key] = conv_results
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_results[problem_key] = {'spatial': {}, 'temporal': {}}

        else:
            # Standard refinement mode
            all_results[problem_key] = {}

            for discretization in solvers:
                print(f"\n  Solver: {discretization}")
                try:
                    results = benchmark_grid_refinement(
                        problem_name,
                        discretization,
                        target_error=target_error,
                        max_nx=max_nx,
                        **problem_kwargs
                    )
                    all_results[problem_key][discretization] = results

                    if results:
                        final = results[-1]
                        status = "✓" if final['metrics']['is_valid'] else "✗"
                        print(f"    {status} Final: nx={final['grid_params']['nx']}, "
                              f"error={final['metrics']['rel_error_pct']:.2f}%")
                    else:
                        print(f"    ✗ No results obtained")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    all_results[problem_key][discretization] = []

    return all_results
