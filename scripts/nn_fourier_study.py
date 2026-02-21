#!/usr/bin/env python3
"""
Fourier-space NN architecture study for PDE-constrained optimization.

This study keeps the PDE time-stepping in physical space but moves the NN I/O
to spatial Fourier space. It compares:
- architecture variants
- NN input schemes in Fourier space
- Fourier mode budgets

Usage:
    python scripts/nn_fourier_study.py [options]

Examples:
    python scripts/nn_fourier_study.py --dry-run
    python scripts/nn_fourier_study.py --architectures baseline-tanh-256x2 --problems heat-1d --max-iter 500
    python scripts/nn_fourier_study.py --input-schemes state_time,state_only,time_only --mode-budgets 8,16,32,full
"""

import argparse
import csv
import json
import sys
import time
import os
from pathlib import Path
from typing import Union

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
    resolve_fourier_mode_count,
)
from pde_opt.problems import get_problem
from pde_opt.solvers import get_solver


DEFAULT_PROBLEMS = [
    'heat-1d',
    'heat-1d-mixed',
    'heat-1d-spatial-mixed',
    'heat-1d-multimode',
]

VALID_INPUT_SCHEMES = ['state_time', 'state_only', 'time_only']
DEFAULT_MODE_BUDGETS = ['8', '16', '32', 'full']


def mode_budget_to_label(mode_budget: Union[str, int]) -> str:
    """Normalize mode budget for display and output paths."""
    if isinstance(mode_budget, str):
        token = mode_budget.strip().lower()
        if token == 'full':
            return 'full'
        return str(int(token))
    return str(int(mode_budget))


def parse_mode_budgets(mode_budgets: str = None) -> list[Union[str, int]]:
    """Parse comma-separated mode budgets."""
    if mode_budgets is None:
        tokens = DEFAULT_MODE_BUDGETS
    else:
        tokens = [t.strip() for t in mode_budgets.split(',')]

    parsed: list[Union[str, int]] = []
    for token in tokens:
        if token.lower() == 'full':
            parsed.append('full')
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"Mode budgets must be positive integers or 'full', got: {token}")
            parsed.append(value)
    return parsed


def parse_input_schemes(input_schemes: str = None) -> list[str]:
    """Parse and validate input schemes."""
    if input_schemes is None:
        return list(VALID_INPUT_SCHEMES)

    requested = [s.strip() for s in input_schemes.split(',')]
    for scheme in requested:
        if scheme not in VALID_INPUT_SCHEMES:
            valid = ', '.join(VALID_INPUT_SCHEMES)
            raise ValueError(f"Unknown input scheme '{scheme}'. Valid options: {valid}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fourier-space NN architecture study for PDE-constrained optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/nn_fourier_study',
        help='Output directory for results (default: results/nn_fourier_study/)',
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
        '--input-schemes', type=str, default=None,
        help='Comma-separated schemes from state_time,state_only,time_only (default: all)',
    )
    parser.add_argument(
        '--mode-budgets', type=str, default=None,
        help='Comma-separated mode budgets (e.g. 8,16,32,full). Default: 8,16,32,full',
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
    input_scheme: str,
    mode_budget: Union[str, int],
    n_modes_used: int,
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

    converged_iter = None
    for i, loss in enumerate(losses):
        if loss < 1e-4:
            converged_iter = i
            break

    return {
        'arch_name': arch_config.name,
        'problem_name': problem_name,
        'input_scheme': input_scheme,
        'mode_budget': mode_budget_to_label(mode_budget),
        'n_modes_used': int(n_modes_used),
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
    input_scheme: str,
    mode_budget: Union[str, int],
    max_iter: int,
    seed: int,
    lr_schedule_type: str,
) -> dict:
    """Run a single Fourier-space experiment and return metrics."""
    mode_label = mode_budget_to_label(mode_budget)
    print(f"\n{'─' * 72}")
    print(f"  Arch: {arch_config.name}")
    print(f"  Problem: {problem_name}")
    print(f"  Input scheme: {input_scheme} | mode budget: {mode_label}")
    print(f"{'─' * 72}")

    ex = get_example('example-3.3-fourier', problem_name=problem_name)
    n_modes, _ = resolve_fourier_mode_count(mode_budget, ex.grid_params['nx'])
    model = create_network_from_config(arch_config, output_dim=2 * n_modes)

    t_start = time.time()
    params, losses, force, solution = ex.run(
        max_iter=max_iter,
        model=model,
        lr_schedule_type=lr_schedule_type,
        seed=seed,
        input_scheme=input_scheme,
        mode_budget=mode_budget,
    )
    training_time = time.time() - t_start

    T = ex.problem_kwargs.get('T', 1.0)
    solver = get_solver(ex.solver_type, ex.discretization,
                        nx=ex.grid_params['nx'], nt=ex.grid_params['nt'], T=T)
    problem = get_problem(ex.problem_name, **ex.problem_kwargs)

    metrics = compute_metrics(
        problem_name=problem_name,
        solver=solver,
        problem=problem,
        params=params,
        losses=losses,
        force=force,
        solution=solution,
        training_time=training_time,
        arch_config=arch_config,
        input_scheme=input_scheme,
        mode_budget=mode_budget,
        n_modes_used=n_modes,
    )

    print(
        f"\n  Results: loss={metrics['final_loss']:.2e}  "
        f"sol_err={metrics['rel_l2_error_solution']:.2e}  "
        f"force_err={metrics['rel_l2_error_force']:.2e}  "
        f"params={metrics['param_count']}  time={training_time:.1f}s"
    )
    return metrics


def save_individual_results(metrics: dict, output_dir: Path):
    """Save per-run metrics and loss plot."""
    run_dir = (
        output_dir
        / f"scheme-{metrics['input_scheme']}"
        / f"modes-{metrics['mode_budget']}"
        / metrics['problem_name']
        / metrics['arch_name']
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_small = {k: v for k, v in metrics.items() if k != 'loss_history'}
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_small, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(metrics['loss_history'], linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title(
        f"{metrics['arch_name']}\n{metrics['problem_name']} | "
        f"{metrics['input_scheme']} | K={metrics['mode_budget']}"
    )
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

    fieldnames = [
        'arch_name', 'problem_name', 'input_scheme', 'mode_budget', 'n_modes_used',
        'final_loss', 'rel_l2_error_solution', 'rel_l2_error_force',
        'training_time_seconds', 'param_count', 'converged_iter',
    ]
    with open(output_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            for key in ['final_loss', 'rel_l2_error_solution', 'rel_l2_error_force']:
                if row[key] is not None:
                    row[key] = f"{row[key]:.6e}"
            writer.writerow(row)

    print(f"\nSummary saved to {output_dir / 'summary.csv'}")


def plot_error_summary(all_results: list[dict], output_dir: Path):
    """Plot median force error by (input scheme, mode budget)."""
    input_schemes = sorted(set(r['input_scheme'] for r in all_results))
    mode_budgets = sorted(
        set(r['mode_budget'] for r in all_results),
        key=lambda x: float('inf') if x == 'full' else int(x)
    )

    fig, axes = plt.subplots(1, len(input_schemes), figsize=(5 * len(input_schemes), 4), squeeze=False)

    for i, scheme in enumerate(input_schemes):
        ax = axes[0, i]
        medians = []
        for mode in mode_budgets:
            vals = [
                r['rel_l2_error_force']
                for r in all_results
                if r['input_scheme'] == scheme and r['mode_budget'] == mode
            ]
            medians.append(float(np.median(vals)) if vals else np.nan)

        ax.bar(np.arange(len(mode_budgets)), medians, alpha=0.85)
        ax.set_xticks(np.arange(len(mode_budgets)))
        ax.set_xticklabels(mode_budgets)
        ax.set_xlabel('Mode budget')
        ax.set_ylabel('Median relative L2 force error')
        ax.set_title(f"Input scheme: {scheme}")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_error_by_scheme_mode.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Error summary saved to {output_dir / 'comparison_error_by_scheme_mode.png'}")


def plot_architecture_ranking(all_results: list[dict], output_dir: Path):
    """Plot median force error by architecture across all Fourier-study runs."""
    archs = sorted(set(r['arch_name'] for r in all_results))
    medians = []
    for arch in archs:
        vals = [r['rel_l2_error_force'] for r in all_results if r['arch_name'] == arch]
        medians.append(float(np.median(vals)))

    order = np.argsort(medians)
    archs_sorted = [archs[i] for i in order]
    medians_sorted = [medians[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(archs_sorted) * 0.45)))
    y = np.arange(len(archs_sorted))
    ax.barh(y, medians_sorted, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(archs_sorted)
    ax.set_xlabel('Median relative L2 force error')
    ax.set_title('Architecture ranking across Fourier-study runs')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_architecture_ranking.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Architecture ranking saved to {output_dir / 'comparison_architecture_ranking.png'}")


def print_summary_table(all_results: list[dict]):
    """Print compact summary table to stdout."""
    print(f"\n{'=' * 120}")
    print(
        f"{'Architecture':<30} {'Problem':<24} {'Scheme':<12} {'Mode':<6} "
        f"{'Force Err':>10} {'Sol Err':>10} {'Loss':>10}"
    )
    print(f"{'=' * 120}")

    sort_key = lambda r: (r['input_scheme'], r['mode_budget'], r['problem_name'], r['arch_name'])
    for r in sorted(all_results, key=sort_key):
        print(
            f"{r['arch_name']:<30} {r['problem_name']:<24} {r['input_scheme']:<12} "
            f"{r['mode_budget']:<6} {r['rel_l2_error_force']:>10.2e} "
            f"{r['rel_l2_error_solution']:>10.2e} {r['final_loss']:>10.2e}"
        )
    print(f"{'=' * 120}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    arch_configs = get_arch_configs(args.architectures)
    problem_names = get_problem_names(args.problems)
    input_schemes = parse_input_schemes(args.input_schemes)
    mode_budgets = parse_mode_budgets(args.mode_budgets)

    total_runs = len(arch_configs) * len(problem_names) * len(input_schemes) * len(mode_budgets)

    print("=" * 72)
    print("FOURIER-SPACE NN ARCHITECTURE STUDY")
    print("=" * 72)
    print(f"Architectures ({len(arch_configs)}):")
    for c in arch_configs:
        print(
            f"  - {c.name} ({c.arch_type}, {c.activation}, layers={c.hidden_layers}"
            f"{', fourier' if c.use_fourier_features else ''})"
        )
    print(f"\nProblems ({len(problem_names)}):")
    for p in problem_names:
        print(f"  - {p}")
    print(f"\nInput schemes ({len(input_schemes)}): {', '.join(input_schemes)}")
    print(f"Mode budgets ({len(mode_budgets)}): {', '.join(mode_budget_to_label(m) for m in mode_budgets)}")
    print(f"\nTotal runs: {total_runs}")
    print(f"Max iterations: {args.max_iter}")
    print(f"LR schedule: {args.lr_schedule}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 72)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return 0

    all_results = []
    run_matrix = [
        (a, p, s, m)
        for a in arch_configs
        for p in problem_names
        for s in input_schemes
        for m in mode_budgets
    ]

    for run_idx, (arch_config, problem_name, input_scheme, mode_budget) in enumerate(run_matrix):
        print(f"\n[{run_idx + 1}/{total_runs}]", end='')
        metrics = run_single_experiment(
            arch_config=arch_config,
            problem_name=problem_name,
            input_scheme=input_scheme,
            mode_budget=mode_budget,
            max_iter=args.max_iter,
            seed=args.seed,
            lr_schedule_type=args.lr_schedule,
        )
        save_individual_results(metrics, output_dir)
        all_results.append(metrics)

    save_summary(all_results, output_dir)
    plot_error_summary(all_results, output_dir)
    plot_architecture_ranking(all_results, output_dir)
    print_summary_table(all_results)

    print(f"\nAll results saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
