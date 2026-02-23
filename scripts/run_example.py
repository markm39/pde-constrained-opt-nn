#!/usr/bin/env python3
"""General-purpose runner for PDE-constrained optimization examples.

Runs any example/problem/architecture combination with full plotting and metrics.

Usage examples:
    # Run Example 3.3 with fourier-tanh-256x2 on heat-1d
    python scripts/run_example.py --example 3.3 --arch fourier-tanh-256x2 --problem heat-1d

    # Run with custom iterations and output directory
    python scripts/run_example.py --example 3.3 --arch baseline-tanh-256x2 --problem heat-1d-multimode --max-iter 5000

    # Run Example 3.3 with oscillating problem
    python scripts/run_example.py --example 3.3 --problem heat-1d-oscillating-cosine --n-oscillations 15 --T 0.5

    # Run Example 3.5 (2D heat)
    python scripts/run_example.py --example 3.5

    # Run Example 3.3-fourier with Fourier-space I/O
    python scripts/run_example.py --example 3.3-fourier --arch baseline-tanh-256x2 --input-scheme state_time --mode-budget 32

    # Run Vlasov-Poisson two-stream with electric energy cost
    python scripts/run_example.py --example vp --problem vp-two-stream --cost-fn ee --max-iter 50

    # Run Vlasov-Poisson bump-on-tail with time-integrated energy
    python scripts/run_example.py --example vp --problem vp-bump-on-tail --cost-fn eet --max-iter 50

    # List available options
    python scripts/run_example.py --list-archs
    python scripts/run_example.py --list-problems
    python scripts/run_example.py --list-examples
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Registry helpers (lazy-imported to keep --list-* fast)
# ---------------------------------------------------------------------------

ARCH_NAMES = [
    "baseline-tanh-256x2", "wide-tanh-512x2", "deep-tanh-128x4",
    "deep-tanh-256x3", "gelu-256x2", "silu-256x2", "resnet-tanh-256x4",
    "modified-mlp-tanh-256x2", "fourier-tanh-256x2", "modified-mlp-fourier-256x2",
]

EXAMPLE_NAMES = [
    "3.1", "3.2", "3.3", "3.3-fourier", "3.5", "3.6",
    "vp",
]

PROBLEM_NAMES = [
    "poisson-1d-scalar", "poisson-1d-vector",
    "heat-1d", "heat-1d-oscillating", "heat-1d-oscillating-cosine",
    "heat-1d-cosine", "heat-1d-mixed", "heat-1d-spatial-mixed",
    "heat-1d-multimode", "heat-1d-spatial-mixed-nonzero-ic",
    "poisson-2d", "linear-heat-2d", "nonlinear-heat-2d",
    "wave-1d", "advection-diffusion-1d",
    "vp-two-stream", "vp-bump-on-tail",
]

# Examples that accept a model= argument in run()
NN_EXAMPLES = {"3.3", "3.3-fourier"}


def list_options(category: str) -> None:
    items = {"archs": ARCH_NAMES, "problems": PROBLEM_NAMES, "examples": EXAMPLE_NAMES}[category]
    print(f"\nAvailable {category}:")
    for item in items:
        print(f"  {item}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def run_vp(args: argparse.Namespace) -> dict:
    """Run a Vlasov-Poisson optimization example and return metrics."""
    import jax
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pde_opt.examples import get_example
    from pde_opt.utils.vp_plotting import plot_vp_results

    print(f"JAX backend: {jax.default_backend()}  |  devices: {jax.devices()}")

    # Build example kwargs
    ex_kwargs = {}
    if args.problem is not None:
        ex_kwargs["problem_name"] = args.problem
    else:
        ex_kwargs["problem_name"] = "vp-two-stream"
    if args.cost_fn is not None:
        ex_kwargs["cost_function"] = args.cost_fn
    if args.n_modes is not None:
        ex_kwargs["n_fourier_modes"] = args.n_modes
    if args.T is not None:
        ex_kwargs["t_final"] = args.T
    if args.nx is not None:
        ex_kwargs["nx"] = args.nx

    ex = get_example("example-vp", **ex_kwargs)

    print(f"\n{'='*70}")
    print(f"  Vlasov-Poisson  |  {ex.problem_name}  |  {ex.cost_type.upper()} cost")
    print(f"{'='*70}")

    # Build run kwargs
    run_kwargs = {"max_iter": args.max_iter}
    if args.seed is not None:
        run_kwargs["seed"] = args.seed
    if args.vp_optimizer is not None:
        run_kwargs["optimizer"] = args.vp_optimizer
    if args.lr is not None:
        run_kwargs["learning_rate"] = args.lr

    import time
    t0 = time.perf_counter()
    result = ex.run(**run_kwargs)
    elapsed = time.perf_counter() - t0

    # Metrics
    metrics = {
        "example": "vp",
        "problem": ex.problem_name,
        "cost_function": result.cost_type,
        "n_fourier_modes": ex.n_fourier_modes,
        "final_cost": float(result.losses[-1]),
        "electric_energy_final": float(result.ee_array[-1]),
        "electric_energy_baseline": float(result.ee_baseline[-1]),
        "suppression_ratio": float(result.ee_array[-1]) / max(float(result.ee_baseline[-1]), 1e-30),
        "training_time_seconds": round(elapsed, 1),
        "max_iter": args.max_iter,
    }

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Final cost:              {metrics['final_cost']:.6e}")
    print(f"    Electric energy (opt):   {metrics['electric_energy_final']:.6e}")
    print(f"    Electric energy (H=0):   {metrics['electric_energy_baseline']:.6e}")
    print(f"    Suppression ratio:       {metrics['suppression_ratio']:.4f}")

    # Plotting
    figures = plot_vp_results(result)

    cost_label = args.cost_fn or "ee"
    output_dir = args.output_dir / "vp" / f"{ex.problem_name}_{cost_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for fig_name, fig in figures.items():
        filepath = output_dir / f"{fig_name}.png"
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        size_kb = filepath.stat().st_size / 1024
        print(f"    Saved: {filepath.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path.relative_to(PROJECT_ROOT)}")

    plt.close("all")
    return metrics


def run(args: argparse.Namespace) -> dict:
    """Run a single example/problem/architecture combo and return metrics."""
    # VP examples use a separate code path
    if args.example == "vp":
        return run_vp(args)

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pde_opt.examples import get_example, create_network_from_config
    from pde_opt.examples.examples import ARCHITECTURE_CONFIGS
    from pde_opt.solvers import get_solver
    from pde_opt.problems import get_problem
    from pde_opt.utils.plotting import plot_example_results

    print(f"JAX backend: {jax.default_backend()}  |  devices: {jax.devices()}")

    # ---- Build the model (only for NN examples) ----
    model = None
    arch_name = args.arch
    if args.example in NN_EXAMPLES and arch_name is not None:
        config = next((c for c in ARCHITECTURE_CONFIGS if c.name == arch_name), None)
        if config is None:
            print(f"Error: unknown architecture '{arch_name}'")
            print(f"Available: {', '.join(c.name for c in ARCHITECTURE_CONFIGS)}")
            sys.exit(1)
        model = create_network_from_config(config)
        print(f"\nArchitecture: {config.name}")
        print(f"  layers: {config.hidden_layers}  activation: {config.activation}  "
              f"type: {config.arch_type}  fourier: {config.use_fourier_features}"
              + (f"  scale: {config.fourier_scale}" if config.use_fourier_features else ""))
    elif args.example not in NN_EXAMPLES and arch_name is not None:
        print(f"Note: --arch is ignored for example {args.example} (non-NN example)")

    # ---- Build example kwargs ----
    example_key = f"example-{args.example}"
    ex_kwargs = {}
    if args.problem is not None:
        ex_kwargs["problem_name"] = args.problem
    if args.reg is not None:
        ex_kwargs["regularization"] = args.reg
    if args.T is not None:
        ex_kwargs["T"] = args.T
    if args.n_oscillations is not None:
        ex_kwargs["n_oscillations"] = args.n_oscillations
    if args.prob is not None:
        ex_kwargs["prob"] = args.prob
    if args.nx is not None:
        ex_kwargs["nx"] = args.nx
    if args.nt is not None:
        ex_kwargs["nt"] = args.nt

    ex = get_example(example_key, **ex_kwargs)

    # ---- Run ----
    print(f"\n{'='*70}")
    print(f"  Example {args.example}  |  {getattr(ex, 'problem_name', '?')}  |  {arch_name or 'default'}")
    print(f"{'='*70}")

    run_kwargs = {"max_iter": args.max_iter}
    if model is not None:
        run_kwargs["model"] = model
    if args.seed is not None:
        run_kwargs["seed"] = args.seed
    if args.nonneg:
        run_kwargs["nonneg"] = True
        run_kwargs["nonneg_mode"] = args.nonneg_mode
    if args.lr is not None:
        run_kwargs["learning_rate"] = args.lr
    if args.no_clip:
        run_kwargs["grad_clip"] = None
    if args.lr_schedule is not None:
        run_kwargs["lr_schedule_type"] = args.lr_schedule
    if args.example == "3.3-fourier":
        if args.input_scheme is not None:
            run_kwargs["input_scheme"] = args.input_scheme
        if args.mode_budget is not None:
            run_kwargs["mode_budget"] = args.mode_budget

    t0 = time.perf_counter()
    result = ex.run(**run_kwargs)
    elapsed = time.perf_counter() - t0

    # ---- Unpack results (different examples return different shapes) ----
    if args.example in ("3.1",):
        force_scalar, losses, solution = result
        params, force = None, force_scalar
    elif args.example in ("3.2",):
        force_vector, losses, solution = result
        params, force = None, force_vector
    else:
        params, losses, force, solution = result

    # ---- Error metrics (for NN examples with space-time problems) ----
    metrics = {
        "example": args.example,
        "problem": getattr(ex, "problem_name", None),
        "arch": arch_name or "default",
        "final_loss": float(losses[-1]),
        "training_time_seconds": round(elapsed, 1),
        "max_iter": args.max_iter,
    }

    T_val = getattr(ex, "problem_kwargs", {}).get("T", 1.0) if hasattr(ex, "problem_kwargs") else 1.0
    problem_kwargs = getattr(ex, "problem_kwargs", {})

    try:
        solver = get_solver(ex.solver_type, ex.discretization, T=T_val, **ex.grid_params)
        problem = get_problem(ex.problem_name, **problem_kwargs)
        x_grid = solver.x_grid
        t_grid = getattr(solver, "t_grid", None)

        if t_grid is not None and solution is not None:
            nx, nt = solver.nx, solver.nt
            u_pred = jnp.array(solution).reshape(nx, nt)
            u_true = problem.analytical_solution(x_grid, t_grid)
            rel_l2_sol = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))
            metrics["rel_l2_solution"] = rel_l2_sol

            if hasattr(problem, "source_term") and force is not None:
                f_pred = jnp.array(force).reshape(nx, nt)
                f_true = problem.source_term(x_grid, t_grid)
                rel_l2_force = float(jnp.linalg.norm(f_pred - f_true) / jnp.linalg.norm(f_true))
                metrics["rel_l2_force"] = rel_l2_force
    except Exception as e:
        print(f"  Warning: could not compute error metrics: {e}")
        solver, problem = None, None

    # ---- Print results ----
    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Final loss:         {metrics['final_loss']:.6e}")
    if "rel_l2_solution" in metrics:
        print(f"    Rel L2 (solution):  {metrics['rel_l2_solution']:.6e}  "
              f"({100*metrics['rel_l2_solution']:.4f}%)")
    if "rel_l2_force" in metrics:
        print(f"    Rel L2 (force):     {metrics['rel_l2_force']:.6e}  "
              f"({100*metrics['rel_l2_force']:.4f}%)")

    # ---- Plotting ----
    if solver is not None and problem is not None:
        label = f"Example {args.example}: {arch_name or 'default'} / {ex.problem_name}"
        figures = plot_example_results(
            example_name=label,
            solver=solver,
            problem=problem,
            params=params,
            losses=losses,
            force=force,
            solution=solution,
            max_snapshots=5,
            figsize_scale=1.0,
        )

        # Save figures
        arch_label = (arch_name or "default") + ("-nonneg" if args.nonneg else "")
        output_dir = args.output_dir / arch_label / ex.problem_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for fig_name, fig in figures.items():
            filepath = output_dir / f"{fig_name}.png"
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            size_kb = filepath.stat().st_size / 1024
            print(f"    Saved: {filepath.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")

        # Save metrics json alongside the figures
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"    Saved: {metrics_path.relative_to(PROJECT_ROOT)}")

        plt.close("all")
    else:
        print("  (skipped plotting -- solver/problem not available)")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PDE-constrained optimization examples with configurable architecture, problem, and solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # List flags
    p.add_argument("--list-archs", action="store_true", help="List available architectures and exit")
    p.add_argument("--list-problems", action="store_true", help="List available problems and exit")
    p.add_argument("--list-examples", action="store_true", help="List available examples and exit")

    # Core config
    p.add_argument("--example", default="3.3", choices=EXAMPLE_NAMES,
                    help="Example to run (default: 3.3)")
    p.add_argument("--arch", default=None,
                    help="Architecture name (default: example's built-in default)")
    p.add_argument("--problem", default=None,
                    help="Problem name (default: example's built-in default)")
    p.add_argument("--max-iter", type=int, default=3000,
                    help="Maximum training iterations (default: 3000)")

    # Problem-specific
    p.add_argument("--T", type=float, default=None, help="Time horizon")
    p.add_argument("--n-oscillations", type=int, default=None, help="Number of oscillations (oscillating problems)")
    p.add_argument("--reg", type=float, default=None, help="Regularization weight")
    p.add_argument("--prob", default=None, help="Sub-problem variant for Example 3.5")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--lr-schedule", default=None, choices=["exponential", "cosine"],
                    help="Learning rate schedule (default: exponential)")
    p.add_argument("--nonneg", action="store_true",
                    help="Enforce non-negative force on NN output")
    p.add_argument("--nonneg-mode", default="relu", choices=["relu", "softplus", "square"],
                    help="Non-negative activation: relu, softplus, or square (default: relu)")
    p.add_argument("--lr", type=float, default=None,
                    help="Override learning rate (default: 3e-3)")
    p.add_argument("--nx", type=int, default=None,
                    help="Override spatial grid size (default: problem-dependent)")
    p.add_argument("--nt", type=int, default=None,
                    help="Override temporal grid size (default: 50)")
    p.add_argument("--no-clip", action="store_true",
                    help="Disable gradient clipping (default: clip_by_global_norm(1.0))")

    # Fourier-space specific (example 3.3-fourier)
    p.add_argument("--input-scheme", default=None, choices=["state_time", "state_only", "time_only"],
                    help="Fourier-space input scheme (3.3-fourier only)")
    p.add_argument("--mode-budget", default=None,
                    help="Fourier mode budget: integer or 'full' (3.3-fourier only)")

    # Vlasov-Poisson specific (example vp)
    p.add_argument("--cost-fn", default=None, choices=["kl", "ee", "eet"],
                    help="VP cost function: kl, ee (final energy), eet (time-integrated energy)")
    p.add_argument("--n-modes", type=int, default=None,
                    help="Number of Fourier modes for H(x) parameterization (VP only)")
    p.add_argument("--vp-optimizer", default=None, choices=["linesearch", "adam"],
                    help="Optimizer for VP examples (default: linesearch)")

    # Output
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "runs",
                    help="Output directory for figures and metrics")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_archs:
        list_options("archs")
    if args.list_problems:
        list_options("problems")
    if args.list_examples:
        list_options("examples")

    # Parse mode_budget to int if numeric
    if args.mode_budget is not None and args.mode_budget != "full":
        try:
            args.mode_budget = int(args.mode_budget)
        except ValueError:
            print(f"Error: --mode-budget must be an integer or 'full', got '{args.mode_budget}'")
            sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
