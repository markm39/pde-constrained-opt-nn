#!/usr/bin/env python3
"""
NN architecture comparison study for PDE-constrained optimization.

Compares different neural network architectures on simple 1D heat equation
force recovery problems. Saves per-run metrics and generates comparison plots.

Usage:
    python scripts/nn_architecture_study.py [options]

Examples:
    python scripts/nn_architecture_study.py --dry-run
    python scripts/nn_architecture_study.py --architectures baseline-tanh-256x2 --problems heat-1d --max-iter 500
    python scripts/nn_architecture_study.py --max-iter 3000
"""

import argparse
import json
import csv
import sys
import time
import os
from pathlib import Path
from dataclasses import asdict

import jax
import jax.numpy as jnp
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pde_opt.examples import (
    get_example,
    create_network_from_config,
    ArchitectureConfig,
    ARCHITECTURE_CONFIGS,
)
from pde_opt.problems import get_problem
from pde_opt.solvers import get_solver

# Default problems to test on (simple 1D heat equations)
DEFAULT_PROBLEMS = [
    'heat-1d',
    'heat-1d-mixed',
    'heat-1d-spatial-mixed',
    'heat-1d-multimode',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='NN architecture comparison study for PDE-constrained optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/nn_study',
        help='Output directory for results (default: results/nn_study/)',
    )
    parser.add_argument(
        '--problems', type=str, default=None,
        help='Comma-separated problem names (default: all simple 1D heat problems)',
    )
    parser.add_argument(
        '--architectures', type=str, default=None,
        help='Comma-separated architecture names or omit for all (default: all)',
    )
    parser.add_argument(
        '--max-iter', type=int, default=3000,
        help='Training iterations per run (default: 3000)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--lr-schedule', type=str, default='cosine',
        choices=['exponential', 'cosine'],
        help='Learning rate schedule (default: cosine)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print configurations without running experiments',
    )
    return parser.parse_args()


def get_arch_configs(names: str = None) -> list[ArchitectureConfig]:
    """Get architecture configs, optionally filtered by name."""
    if names is None:
        return list(ARCHITECTURE_CONFIGS)
    requested = [n.strip() for n in names.split(',')]
    configs_by_name = {c.name: c for c in ARCHITECTURE_CONFIGS}
    result = []
    for name in requested:
        if name not in configs_by_name:
            available = ', '.join(configs_by_name.keys())
            raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
        result.append(configs_by_name[name])
    return result


def get_problem_names(names: str = None) -> list[str]:
    """Get problem names, optionally filtered."""
    if names is None:
        return list(DEFAULT_PROBLEMS)
    return [n.strip() for n in names.split(',')]


def count_params(params) -> int:
    """Count total number of trainable parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def compute_metrics(
    problem_name: str,
    solver,
    problem,
    params,
    losses: list,
    force: jnp.ndarray,
    solution: jnp.ndarray,
    training_time: float,
    arch_config: ArchitectureConfig,
) -> dict:
    """Compute all metrics for a single experiment run."""
    x_grid = solver.x_grid
    t_grid = solver.t_grid
    nx, nt = solver.nx, solver.nt

    # Reshape outputs
    u_pred = solution.reshape(nx, nt)
    f_pred = force.reshape(nx, nt)

    # True values
    u_true = problem.analytical_solution(x_grid, t_grid)
    f_true = problem.source_term(x_grid, t_grid)

    # Relative L2 errors
    u_err = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))
    f_err = float(jnp.linalg.norm(f_pred - f_true) / jnp.linalg.norm(f_true))

    # Find iteration where loss first drops below 1e-4 (convergence proxy)
    converged_iter = None
    for i, loss in enumerate(losses):
        if loss < 1e-4:
            converged_iter = i
            break

    return {
        'arch_name': arch_config.name,
        'problem_name': problem_name,
        'final_loss': losses[-1],
        'rel_l2_error_solution': u_err,
        'rel_l2_error_force': f_err,
        'training_time_seconds': round(training_time, 1),
        'param_count': count_params(params),
        'converged_iter': converged_iter,
        'loss_history': losses,
    }


def run_single_experiment(
    arch_config: ArchitectureConfig,
    problem_name: str,
    max_iter: int,
    seed: int,
    lr_schedule_type: str,
) -> dict:
    """Run a single architecture on a single problem and return metrics."""
    print(f"\n{'─' * 60}")
    print(f"  Arch: {arch_config.name}")
    print(f"  Problem: {problem_name}")
    print(f"{'─' * 60}")

    # Create the model from config
    model = create_network_from_config(arch_config)

    # Create the example (uses default grid params for simple problems)
    ex = get_example('example-3.3', problem_name=problem_name)

    # Run training
    t_start = time.time()
    params, losses, force, solution = ex.run(
        max_iter=max_iter,
        model=model,
        lr_schedule_type=lr_schedule_type,
        seed=seed,
    )
    training_time = time.time() - t_start

    # Get solver and problem for metrics
    T = ex.problem_kwargs.get('T', 1.0)
    solver = get_solver(ex.solver_type, ex.discretization,
                        nx=ex.grid_params['nx'], nt=ex.grid_params['nt'], T=T)
    problem = get_problem(ex.problem_name, **ex.problem_kwargs)

    metrics = compute_metrics(
        problem_name, solver, problem, params, losses,
        force, solution, training_time, arch_config,
    )

    n_params = metrics['param_count']
    print(f"\n  Results: loss={metrics['final_loss']:.2e}  "
          f"sol_err={metrics['rel_l2_error_solution']:.2e}  "
          f"force_err={metrics['rel_l2_error_force']:.2e}  "
          f"params={n_params}  time={training_time:.1f}s")

    return metrics


def save_individual_results(metrics: dict, output_dir: Path):
    """Save per-run results (metrics JSON and loss plot)."""
    run_dir = output_dir / metrics['problem_name'] / metrics['arch_name']
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics (without loss_history to keep it small)
    metrics_small = {k: v for k, v in metrics.items() if k != 'loss_history'}
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_small, f, indent=2)

    # Save loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(metrics['loss_history'], linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{metrics["arch_name"]} on {metrics["problem_name"]}')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(metrics['loss_history'], linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title('Log Scale')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(run_dir / 'loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_summary(all_results: list[dict], output_dir: Path):
    """Save summary JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary (without loss histories)
    summary_data = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d'),
            'jax_backend': str(jax.default_backend()),
        },
        'results': [
            {k: v for k, v in r.items() if k != 'loss_history'}
            for r in all_results
        ],
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    # CSV summary
    fieldnames = [
        'arch_name', 'problem_name', 'final_loss',
        'rel_l2_error_solution', 'rel_l2_error_force',
        'training_time_seconds', 'param_count', 'converged_iter',
    ]
    with open(output_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            # Format floats for readability
            for key in ['final_loss', 'rel_l2_error_solution', 'rel_l2_error_force']:
                if row[key] is not None:
                    row[key] = f'{row[key]:.6e}'
            writer.writerow(row)

    print(f"\nSummary saved to {output_dir / 'summary.csv'}")


def plot_loss_comparison(all_results: list[dict], output_dir: Path):
    """Plot loss curves overlaid per problem."""
    problems = sorted(set(r['problem_name'] for r in all_results))
    n_problems = len(problems)

    fig, axes = plt.subplots(1, n_problems, figsize=(5 * n_problems, 4), squeeze=False)

    for j, prob in enumerate(problems):
        ax = axes[0, j]
        prob_results = [r for r in all_results if r['problem_name'] == prob]

        for r in prob_results:
            ax.semilogy(r['loss_history'], linewidth=1.2, label=r['arch_name'], alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(prob)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_loss_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss comparison saved to {output_dir / 'comparison_loss_curves.png'}")


def plot_error_comparison(all_results: list[dict], output_dir: Path):
    """Plot bar chart of force recovery errors grouped by problem."""
    problems = sorted(set(r['problem_name'] for r in all_results))
    archs = sorted(set(r['arch_name'] for r in all_results))

    # Build error matrix
    errors = {}
    for r in all_results:
        errors[(r['problem_name'], r['arch_name'])] = r['rel_l2_error_force']

    x = np.arange(len(problems))
    width = 0.8 / len(archs)

    fig, ax = plt.subplots(figsize=(max(8, len(problems) * 3), 5))

    for i, arch in enumerate(archs):
        vals = [errors.get((p, arch), 0) for p in problems]
        offset = (i - len(archs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=arch, alpha=0.85)

    ax.set_xlabel('Problem')
    ax.set_ylabel('Relative L2 Error (Force)')
    ax.set_title('Force Recovery Error by Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=15, ha='right')
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_errors.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Error comparison saved to {output_dir / 'comparison_errors.png'}")


def print_summary_table(all_results: list[dict]):
    """Print a formatted summary table to stdout."""
    print(f"\n{'=' * 90}")
    print(f"{'Architecture':<30} {'Problem':<25} {'Force Err':>10} {'Sol Err':>10} {'Loss':>10}")
    print(f"{'=' * 90}")

    # Sort by problem then architecture
    sorted_results = sorted(all_results, key=lambda r: (r['problem_name'], r['arch_name']))
    current_problem = None

    for r in sorted_results:
        if r['problem_name'] != current_problem:
            if current_problem is not None:
                print(f"{'─' * 90}")
            current_problem = r['problem_name']

        print(f"{r['arch_name']:<30} {r['problem_name']:<25} "
              f"{r['rel_l2_error_force']:>10.2e} {r['rel_l2_error_solution']:>10.2e} "
              f"{r['final_loss']:>10.2e}")

    print(f"{'=' * 90}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Get configurations
    arch_configs = get_arch_configs(args.architectures)
    problem_names = get_problem_names(args.problems)

    total_runs = len(arch_configs) * len(problem_names)

    # Print configuration
    print("=" * 60)
    print("NN ARCHITECTURE STUDY")
    print("=" * 60)
    print(f"Architectures ({len(arch_configs)}):")
    for c in arch_configs:
        print(f"  - {c.name} ({c.arch_type}, {c.activation}, layers={c.hidden_layers}"
              f"{', fourier' if c.use_fourier_features else ''})")
    print(f"\nProblems ({len(problem_names)}):")
    for p in problem_names:
        print(f"  - {p}")
    print(f"\nTotal runs: {total_runs}")
    print(f"Max iterations: {args.max_iter}")
    print(f"LR schedule: {args.lr_schedule}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return 0

    # Run experiments
    all_results = []
    for run_idx, (arch_config, problem_name) in enumerate(
        [(a, p) for a in arch_configs for p in problem_names]
    ):
        print(f"\n[{run_idx + 1}/{total_runs}]", end='')

        metrics = run_single_experiment(
            arch_config=arch_config,
            problem_name=problem_name,
            max_iter=args.max_iter,
            seed=args.seed,
            lr_schedule_type=args.lr_schedule,
        )

        # Save individual results immediately (in case of crash)
        save_individual_results(metrics, output_dir)
        all_results.append(metrics)

    # Save summary
    save_summary(all_results, output_dir)

    # Generate comparison plots
    plot_loss_comparison(all_results, output_dir)
    plot_error_comparison(all_results, output_dir)

    # Print summary table
    print_summary_table(all_results)

    print(f"\nAll results saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
