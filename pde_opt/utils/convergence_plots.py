"""
Convergence plotting utilities for solver benchmarks.

Creates plots showing error reduction as grid size increases,
and computes convergence orders.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple


def compute_convergence_order(h_values: List[float], error_values: List[float]) -> float:
    """
    Compute convergence order from log-log slope.

    Uses least squares fit to log(error) vs log(h) to determine slope.

    Args:
        h_values: Grid spacing values (h = 1/(nx+1) or k = T/nt)
        error_values: Corresponding relative errors

    Returns:
        Convergence order (e.g., 2.0 for O(h²))
    """
    if len(h_values) < 2 or len(error_values) < 2:
        return 0.0

    # Filter out zero/negative values
    valid_indices = [i for i in range(len(error_values)) if error_values[i] > 0]
    if len(valid_indices) < 2:
        return 0.0

    log_h = np.log([h_values[i] for i in valid_indices])
    log_err = np.log([error_values[i] for i in valid_indices])

    # Linear fit: log(err) = slope * log(h) + intercept
    # slope = convergence order
    coeffs = np.polyfit(log_h, log_err, 1)
    order = coeffs[0]

    return float(order)


def plot_spatial_convergence(
    results: Dict[str, List[Dict[str, Any]]],
    problem_name: str,
    filename: str,
    T: float = 1.0
) -> None:
    """
    Plot spatial convergence (error vs nx) for all solvers.

    Args:
        results: Dict mapping solver name to list of benchmark results
        problem_name: Name of the problem for title
        filename: Output file path
        T: Time domain length (for computing k)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'fd': 'blue', 'crank-nicolson': 'red', 'fem': 'green'}
    markers = {'fd': 'o', 'crank-nicolson': 's', 'fem': '^'}

    for solver, bench_results in results.items():
        if not bench_results:
            continue

        # Extract data
        nx_values = [r['grid_params']['nx'] for r in bench_results]
        h_values = [1.0/(nx+1) for nx in nx_values]
        errors = [r['metrics']['rel_error'] for r in bench_results]

        # Compute convergence order
        order = compute_convergence_order(h_values, errors)

        # Plot
        color = colors.get(solver, 'black')
        marker = markers.get(solver, 'x')
        label = f"{solver} (order={order:.2f})"

        ax.loglog(h_values, errors, marker=marker, color=color,
                  linewidth=2, markersize=8, label=label)

    ax.set_xlabel('Grid spacing h = 1/(nx+1)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'Spatial Convergence: {problem_name}', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_temporal_convergence(
    results: Dict[str, List[Dict[str, Any]]],
    problem_name: str,
    filename: str,
    T: float = 1.0
) -> None:
    """
    Plot temporal convergence (error vs nt) for all solvers.

    Args:
        results: Dict mapping solver name to list of benchmark results
        problem_name: Name of the problem for title
        filename: Output file path
        T: Time domain length
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'fd': 'blue', 'crank-nicolson': 'red', 'fem': 'green'}
    markers = {'fd': 'o', 'crank-nicolson': 's', 'fem': '^'}

    for solver, bench_results in results.items():
        if not bench_results:
            continue

        # Extract data
        nt_values = [r['grid_params']['nt'] for r in bench_results]
        k_values = [T/nt for nt in nt_values]
        errors = [r['metrics']['rel_error'] for r in bench_results]

        # Compute convergence order
        order = compute_convergence_order(k_values, errors)

        # Plot
        color = colors.get(solver, 'black')
        marker = markers.get(solver, 'x')
        label = f"{solver} (order={order:.2f})"

        ax.loglog(k_values, errors, marker=marker, color=color,
                  linewidth=2, markersize=8, label=label)

    ax.set_xlabel('Time step k = T/nt', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'Temporal Convergence: {problem_name}', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_all_convergence(
    all_results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    output_dir: str
) -> Dict[str, List[str]]:
    """
    Create all convergence plots for all problems.

    Args:
        all_results: Nested dict {problem: {'spatial': {solver: [results]},
                                            'temporal': {solver: [results]}}}
        output_dir: Output directory for plots

    Returns:
        Dictionary mapping problem to list of generated plot filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    for problem_key, conv_results in all_results.items():
        # Skip if not convergence study format
        if 'spatial' not in conv_results or 'temporal' not in conv_results:
            continue

        print(f"\nGenerating plots for {problem_key}...")

        # Clean problem name for filename
        problem_filename = problem_key.replace('[', '_').replace(']', '').replace(',', '_').replace('=', '')

        # Extract T value if present (for temporal convergence k = T/nt)
        T = 1.0  # default
        if 'T=0.5' in problem_key:
            T = 0.5

        files = []

        # Spatial convergence plot
        if conv_results['spatial']:
            spatial_file = output_path / f"{problem_filename}_spatial.png"
            plot_spatial_convergence(
                conv_results['spatial'],
                problem_key,
                str(spatial_file),
                T=T
            )
            files.append(str(spatial_file))

        # Temporal convergence plot
        if conv_results['temporal']:
            temporal_file = output_path / f"{problem_filename}_temporal.png"
            plot_temporal_convergence(
                conv_results['temporal'],
                problem_key,
                str(temporal_file),
                T=T
            )
            files.append(str(temporal_file))

        generated_files[problem_key] = files

    return generated_files
