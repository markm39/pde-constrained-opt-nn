"""
Solver validation module for PDE-constrained optimization.

This module provides tools to validate solver accuracy and find optimal grid configurations.
Ensures that the forward solver has sufficiently low error (< 1%) before attempting to learn
the forcing term with neural networks.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax
from typing import Dict, Tuple, Any
from pde_opt.problems import get_problem
from pde_opt.solvers import get_solver


def validate_solver(example, threshold: float = 0.01, verbose: bool = True) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Validate that the forward solver has acceptable accuracy with ground truth forcing.

    Args:
        example: OptimizationExample instance with problem and solver configuration
        threshold: Maximum acceptable relative L2 error (default: 0.01 for 1%)
        verbose: Print validation details

    Returns:
        is_valid: True if error < threshold
        rel_error: Relative L2 error as a fraction (e.g., 0.05 for 5%)
        details: Dictionary with MSE, grid params, problem name, etc.
    """
    # Get problem configuration
    if hasattr(example, 'prob'):
        # Example35 with prob parameter
        problem = get_problem(example.problem_name, prob=example.prob)
        prob_name = f"{example.problem_name} ({example.prob})"
    elif hasattr(example, 'problem_kwargs'):
        # Example33 with n_oscillations
        problem = get_problem(example.problem_name, **example.problem_kwargs)
        prob_name = f"{example.problem_name} (k={example.problem_kwargs.get('n_oscillations', 1)})"
    else:
        problem = get_problem(example.problem_name)
        prob_name = example.problem_name

    # Get solver
    solver = get_solver(example.solver_type, example.discretization, **example.grid_params)

    # Check problem type
    is_2d = hasattr(solver, 'y_grid')
    is_time_dependent = hasattr(solver, 't_grid')

    # Skip validation for static problems (Poisson)
    if not is_time_dependent:
        if verbose:
            print(f"\n⚠ Skipping validation for static problem: {prob_name}")
            print(f"Static problems (Poisson) solve a linear system directly.")
            print(f"Discretization error depends only on grid resolution.")
        return True, 0.0, {'problem_name': prob_name, 'is_static': True}

    if is_2d:
        # 2D spatial problem
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

        f_true_3d = jnp.stack([problem.source_term(x_grid, y_grid, t)
                                for t in t_grid], axis=-1)
        f_true = f_true_3d.reshape(n_spatial, solver.nt)

        # Also get forcing at t=0 for Crank-Nicolson force averaging
        # For 2D, source_term expects scalar time, returns (nx, ny)
        f_at_t0_2d = problem.source_term(x_grid, y_grid, 0.0)
        f_at_t0 = f_at_t0_2d.flatten()  # (nx*ny,)

        # Setup Crank-Nicolson solver
        A = solver.K  # -Δ (negative Laplacian)
        A_cn = jnp.eye(n_spatial) - (k/2.0) * A
        L_cn = jnp.linalg.cholesky(A_cn)

        def chol_solve(L, b):
            y = jsp.solve_triangular(L, b, lower=True)
            u = jsp.solve_triangular(L.T, y, lower=False)
            return u

        # Time-stepping with TRUE forcing using proper CN force averaging
        def forward_with_true_forcing(u0, f_true, f_at_t0):
            # Augment forcing to include t=0 for proper averaging
            f_augmented = jnp.column_stack([f_at_t0, f_true])  # (nx*ny, nt+1)

            def step(u_prev, i):
                # Average force between old and new time levels for 2nd order accuracy
                f_old = f_augmented[:, i]      # Force at old time level
                f_new = f_augmented[:, i+1]    # Force at new time level
                f_avg = 0.5 * (f_old + f_new)  # Trapezoid rule

                rhs = (jnp.eye(n_spatial) + (k/2.0) * A) @ u_prev + k * f_avg
                u_next = chol_solve(L_cn, rhs)
                return u_next, u_next

            indices = jnp.arange(solver.nt)
            _, U_seq = lax.scan(step, u0, indices)
            return U_seq  # (nt, nx*ny)

        U_pred = forward_with_true_forcing(u0, f_true, f_at_t0)

    else:
        # 1D spatial problem (Example 3.3)
        x_grid = solver.x_grid
        t_grid = solver.t_grid
        k = solver.k

        # Get ground truth
        u_target = problem.analytical_solution(x_grid, t_grid)  # (nx, nt)
        u0 = problem.initial_condition(x_grid)

        # Get TRUE forcing - for CN, we need forcing at t=0 as well for averaging
        f_true = problem.source_term(x_grid, t_grid)  # (nx, nt) at t_grid times
        # Also get forcing at t=0 for Crank-Nicolson force averaging
        f_at_t0 = problem.source_term(x_grid, jnp.array([0.0]))[:, 0]  # (nx,)

        # Setup time-stepping solver based on discretization method
        def chol_solve(L, b):
            y = jsp.solve_triangular(L, b, lower=True)
            u = jsp.solve_triangular(L.T, y, lower=False)
            return u

        if example.discretization == 'crank-nicolson':
            # Crank-Nicolson: (I - 0.5*k*K) * u_next = (I + 0.5*k*K) * u_prev + k*f_avg
            # Note: CN's K is negative definite (diagonal = -2/h²)
            # For 2nd order accuracy, force should be averaged: f_avg = 0.5*(f_old + f_new)
            K_h = solver.create_spatial_matrix()
            A_cn = jnp.eye(solver.nx) - (k/2.0) * K_h
            B_cn = jnp.eye(solver.nx) + (k/2.0) * K_h
            L_cn = jnp.linalg.cholesky(A_cn)

            def forward_with_true_forcing(u0, f_true, f_at_t0):
                # Augment forcing to include t=0 for proper averaging
                # f_augmented[:, i] is force at time just before computing u at t_grid[i]
                f_augmented = jnp.column_stack([f_at_t0, f_true])  # (nx, nt+1)

                def step(u_prev, i):
                    # Average force between old and new time levels for 2nd order accuracy
                    f_old = f_augmented[:, i]      # Force at old time level
                    f_new = f_augmented[:, i+1]    # Force at new time level
                    f_avg = 0.5 * (f_old + f_new)  # Trapezoid rule

                    rhs = B_cn @ u_prev + k * f_avg
                    u_next = chol_solve(L_cn, rhs)
                    return u_next, u_next

                indices = jnp.arange(solver.nt)
                _, U_seq = lax.scan(step, u0, indices)
                return U_seq.T  # (nx, nt)

        else:
            # Backward Euler for FD and FEM
            if hasattr(solver, 'create_fem_matrices'):
                # FEM: (M/k + K) * u_next = M/k * u_prev + M * f
                M, K_h = solver.create_fem_matrices()
                A_be = M / k + K_h
            else:
                # FD: (I/k - K) * u_next = u_prev/k + f
                # Note: K_h uses negative definite convention (K_h ≈ -Δ)
                K_h = solver.create_spatial_matrix()
                A_be = (1.0/k) * jnp.eye(solver.nx) - K_h  # Changed + to -

            L_be = jnp.linalg.cholesky(A_be)

            def forward_with_true_forcing(u0, f_true):
                def step(u_prev, i):
                    f_n = f_true[:, i]
                    if hasattr(solver, 'create_fem_matrices'):
                        # FEM: rhs = M/k * u_prev + M * f (weak form)
                        rhs = (M / k) @ u_prev + M @ f_n
                    else:
                        # FD: rhs = u_prev/k + f
                        rhs = u_prev / k + f_n
                    u_next = chol_solve(L_be, rhs)
                    return u_next, u_next

                indices = jnp.arange(solver.nt)
                _, U_seq = lax.scan(step, u0, indices)
                return U_seq.T  # (nx, nt)

        # Call solver with appropriate arguments
        if example.discretization == 'crank-nicolson':
            U_pred = forward_with_true_forcing(u0, f_true, f_at_t0)
        else:
            U_pred = forward_with_true_forcing(u0, f_true)

    # Compute errors
    if is_2d:
        error = U_pred.T - u_target
    else:
        error = U_pred - u_target

    mse = jnp.mean(error**2)
    rel_error = jnp.linalg.norm(error) / jnp.linalg.norm(u_target)

    is_valid = rel_error < threshold

    # Prepare details
    details = {
        'problem_name': prob_name,
        'grid_params': example.grid_params,
        'mse': float(mse),
        'rel_error': float(rel_error),
        'rel_error_pct': float(rel_error * 100),
        'threshold_pct': threshold * 100,
        'is_2d': is_2d
    }

    if verbose:
        print("\n" + "="*60)
        print("SOLVER VALIDATION")
        print("="*60)
        print(f"Problem: {prob_name}")
        print(f"Grid: nx={example.grid_params['nx']}", end="")
        if is_2d:
            print(f", ny={example.grid_params['ny']}", end="")
        print(f", nt={example.grid_params['nt']}")
        print(f"\nResults:")
        print(f"  MSE:              {mse:.6e}")
        print(f"  Relative L2:      {rel_error:.6e} ({rel_error*100:.2f}%)")
        print(f"  Threshold:        {threshold:.6e} ({threshold*100:.2f}%)")
        print(f"\nStatus: ", end="")
        if is_valid:
            print(f"✓ PASS - Solver accuracy is sufficient")
        else:
            print(f"✗ FAIL - Solver error {rel_error*100:.2f}% exceeds threshold {threshold*100:.2f}%")
            print(f"\nRecommendation: Use find_optimal_config() to find better grid size")
        print("="*60 + "\n")

    return is_valid, rel_error, details


def find_optimal_config(example, threshold: float = 0.01, max_nx: int = 200) -> Dict[str, int]:
    """
    Find optimal grid configuration to achieve desired solver accuracy.

    Uses adaptive search: doubles grid size until error < threshold, then refines.

    Args:
        example: OptimizationExample instance
        threshold: Target relative L2 error (default: 0.01 for 1%)
        max_nx: Maximum grid size to try (prevents runaway)

    Returns:
        grid_params: Dictionary with recommended nx, ny (if 2D), nt
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL GRID CONFIGURATION")
    print("="*60)
    print(f"Target error: < {threshold*100:.2f}%")
    print(f"Max grid size: {max_nx}")
    print()

    # Start with current config
    current_nx = example.grid_params['nx']
    current_nt = example.grid_params['nt']
    is_2d = 'ny' in example.grid_params

    if is_2d:
        current_ny = example.grid_params['ny']

    # Create a modified example for testing
    import copy
    test_example = copy.deepcopy(example)

    print(f"{'nx':<8} {'ny':<8} {'nt':<8} {'Error (%)':<12} {'Status'}")
    print("-" * 60)

    # Adaptive search
    results = []
    nx = current_nx
    nt = current_nt

    while nx <= max_nx:
        # Update test config
        test_example.grid_params['nx'] = nx
        test_example.grid_params['nt'] = nt

        if is_2d:
            ny = nx  # Keep square grid
            test_example.grid_params['ny'] = ny

        # Validate
        is_valid, rel_error, details = validate_solver(test_example, threshold=threshold, verbose=False)

        results.append((nx, ny if is_2d else None, nt, rel_error))

        # Print row
        if is_2d:
            print(f"{nx:<8} {ny:<8} {nt:<8} {rel_error*100:<12.4f} {'✓ PASS' if is_valid else '✗ FAIL'}")
        else:
            print(f"{nx:<8} {'-':<8} {nt:<8} {rel_error*100:<12.4f} {'✓ PASS' if is_valid else '✗ FAIL'}")

        if is_valid:
            # Found good config, try to refine down
            print(f"\n✓ Found acceptable config!")
            break

        # Double grid size
        nx = int(nx * 1.5)
        nt = int(nt * 1.5)

    if not is_valid:
        # Find best achievable error
        best_result = min(results, key=lambda x: x[3])
        best_nx, best_ny, best_nt, best_error = best_result

        print(f"\n✗ Could not achieve {threshold*100:.2f}% threshold within max_nx={max_nx}")
        print(f"\n📊 Best achievable configuration:")
        print(f"  Grid: nx={best_nx}" + (f", ny={best_ny}" if is_2d else "") + f", nt={best_nt}")
        print(f"  Error: {best_error*100:.2f}%")
        print(f"\n💡 Recommendations:")
        print(f"  • For oscillatory problems, {best_error*100:.1f}% error may be acceptable")
        print(f"  • Strict 1% threshold requires prohibitively large grids")
        print(f"  • Consider using a smoother problem or accepting higher error")
        print(f"  • Typical acceptable ranges: 1-5% for smooth, 5-15% for oscillatory")

        recommended = {
            'nx': best_nx,
            'nt': best_nt,
            'achievable_error': float(best_error)
        }
        if is_2d:
            recommended['ny'] = best_ny

        print(f"\nBest config (achieves {best_error*100:.2f}% error):")
        for key, val in recommended.items():
            if key != 'achievable_error':
                print(f"  {key}: {val}")
        print("="*60 + "\n")

        return recommended

    # Threshold was achieved
    recommended = {
        'nx': results[-1][0],
        'nt': results[-1][2]
    }
    if is_2d:
        recommended['ny'] = results[-1][1]

    print(f"\n✓ Found configuration meeting {threshold*100:.2f}% threshold!")
    print(f"\nRecommended grid_params:")
    for key, val in recommended.items():
        print(f"  {key}: {val}")
    print(f"\nThis achieves {results[-1][3]*100:.2f}% relative error")
    print("="*60 + "\n")

    return recommended
