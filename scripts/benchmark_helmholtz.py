#!/usr/bin/env python3
"""Benchmark Helmholtz inverse-medium runs across profiles and parameterizations."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from pde_opt.examples import create_network_from_config, get_example
from pde_opt.examples.examples import ARCHITECTURE_CONFIGS
from scripts.run_example import parse_k_stage_multipliers

METHOD_CONFIGS = (
    {"name": "grid", "mode": "grid", "k_continuation": False},
    {"name": "grid_kcont", "mode": "grid", "k_continuation": True},
    {"name": "nn", "mode": "nn", "k_continuation": False},
    {"name": "nn_kcont", "mode": "nn", "k_continuation": True},
)


def format_output_path(path: Path) -> str:
    """Format an output path relative to the repo root when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Expected at least one integer value")
    return [int(token) for token in tokens]


def parse_name_list(value: str) -> list[str]:
    """Parse a comma-separated list of non-empty identifiers."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Expected at least one value")
    return tokens


def get_architecture_config(arch_name: str):
    """Resolve an architecture configuration by name."""
    config = next((cfg for cfg in ARCHITECTURE_CONFIGS if cfg.name == arch_name), None)
    if config is None:
        available = ", ".join(cfg.name for cfg in ARCHITECTURE_CONFIGS)
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {available}")
    return config


def summarize_runs(raw_runs: list[dict], success_threshold: float) -> list[dict]:
    """Aggregate raw benchmark runs by profile and method."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in raw_runs:
        grouped.setdefault((run["profile"], run["method"]), []).append(run)

    summary = []
    for (profile, method), runs in sorted(grouped.items()):
        rel_l2_values = [run["rel_l2_coefficient"] for run in runs]
        success_rate = sum(value <= success_threshold for value in rel_l2_values) / len(rel_l2_values)
        summary.append({
            "profile": profile,
            "method": method,
            "num_runs": len(runs),
            "best_rel_l2": min(rel_l2_values),
            "median_rel_l2": statistics.median(rel_l2_values),
            "success_rate": success_rate,
            "deterministic": len(runs) == 1,
        })

    return summary


def write_summary_markdown(summary_rows: list[dict], success_threshold: float, output_path: Path) -> None:
    """Write a compact markdown summary table."""
    threshold_pct = 100.0 * success_threshold
    lines = [
        "# Helmholtz Benchmark Summary",
        "",
        f"Success threshold: relative L2 <= {threshold_pct:.2f}%",
        "",
        "| Profile | Method | Runs | Best Rel L2 (%) | Median Rel L2 (%) | Success Rate | Notes |",
        "|---------|--------|------|-----------------|-------------------|--------------|-------|",
    ]

    for row in summary_rows:
        note = "deterministic" if row["deterministic"] else "seed sweep"
        lines.append(
            f"| {row['profile']} | {row['method']} | {row['num_runs']} | "
            f"{100.0 * row['best_rel_l2']:.4f} | {100.0 * row['median_rel_l2']:.4f} | "
            f"{100.0 * row['success_rate']:.1f}% | {note} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_benchmark(method_config: dict, profile: str, seed: int,
                         arch_config, args: argparse.Namespace) -> dict:
    """Run one Helmholtz benchmark configuration and return its metrics."""
    example = get_example(
        "helmholtz-medium",
        k=args.k * jnp.pi,
        nx=args.nx,
        ny=args.ny,
        profile=profile,
        regularization=args.reg,
    )

    run_kwargs = {
        "max_iter": args.stage_iters if method_config["k_continuation"] else args.direct_iters,
        "mode": method_config["mode"],
        "seed": seed,
        "lr_schedule_type": args.lr_schedule,
        "k_continuation": method_config["k_continuation"],
        "k_stages": args.k_stages,
    }
    if args.lr is not None:
        run_kwargs["learning_rate"] = args.lr
    if method_config["mode"] == "nn":
        run_kwargs["model"] = create_network_from_config(arch_config)

    t0 = time.perf_counter()
    _, losses, field_pred, field_true = example.run(**run_kwargs)
    elapsed = time.perf_counter() - t0
    rel_l2 = float(jnp.linalg.norm(field_pred - field_true) / jnp.linalg.norm(field_true))

    return {
        "profile": profile,
        "method": method_config["name"],
        "mode": method_config["mode"],
        "k_continuation": method_config["k_continuation"],
        "seed": seed,
        "arch": arch_config.name if method_config["mode"] == "nn" else "default",
        "k_multiple_of_pi": args.k,
        "k_stages": args.k_stages,
        "nx": args.nx,
        "ny": args.ny,
        "regularization": args.reg,
        "max_iter": run_kwargs["max_iter"],
        "final_loss": float(losses[-1]),
        "rel_l2_coefficient": rel_l2,
        "training_time_seconds": round(elapsed, 1),
    }


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Run the full Helmholtz benchmark matrix."""
    arch_config = get_architecture_config(args.arch)
    raw_runs: list[dict] = []

    for profile in args.profiles:
        for method_config in METHOD_CONFIGS:
            seeds = args.seeds if method_config["mode"] == "nn" else [args.seeds[0]]
            for seed in seeds:
                print(
                    f"Running profile={profile} method={method_config['name']} "
                    f"seed={seed} k={args.k}pi grid={args.nx}x{args.ny}"
                )
                result = run_single_benchmark(
                    method_config=method_config,
                    profile=profile,
                    seed=seed,
                    arch_config=arch_config,
                    args=args,
                )
                raw_runs.append(result)
                print(
                    f"  rel_l2={100.0 * result['rel_l2_coefficient']:.4f}% "
                    f"loss={result['final_loss']:.6e} time={result['training_time_seconds']:.1f}s"
                )

    return raw_runs, summarize_runs(raw_runs, success_threshold=args.success_threshold)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Helmholtz inverse-medium runs across smooth profiles and methods."
    )
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "benchmarks" / "helmholtz",
                        help="Directory for raw benchmark data and markdown summary")
    parser.add_argument("--profiles", default="gaussian_lens,double_lens",
                        help="Comma-separated Helmholtz profiles to benchmark")
    parser.add_argument("--seeds", default="0,1,2,3,4,5",
                        help="Comma-separated seeds for NN runs")
    parser.add_argument("--arch", default="modified-mlp-tanh-256x2",
                        help="NN architecture for Helmholtz runs")
    parser.add_argument("--k", type=float, default=5.0,
                        help="Target Helmholtz wavenumber in multiples of pi")
    parser.add_argument("--nx", type=int, default=30, help="Number of x-direction interior points")
    parser.add_argument("--ny", type=int, default=30, help="Number of y-direction interior points")
    parser.add_argument("--reg", type=float, default=1e-4, help="Helmholtz regularization weight")
    parser.add_argument("--direct-iters", type=int, default=2000,
                        help="Iterations for direct runs")
    parser.add_argument("--stage-iters", type=int, default=500,
                        help="Iterations per continuation stage")
    parser.add_argument("--k-stages", default="1,2,3,4,5",
                        help="Comma-separated continuation stages in multiples of pi")
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "exponential"],
                        help="Learning-rate schedule passed to the Helmholtz example")
    parser.add_argument("--lr", type=float, default=None,
                        help="Optional learning-rate override")
    parser.add_argument("--success-threshold", type=float, default=0.05,
                        help="Success threshold on relative L2 coefficient error")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write outputs."""
    args = parse_args(argv)
    try:
        args.profiles = parse_name_list(args.profiles)
        args.seeds = parse_int_list(args.seeds)
        args.k_stages = parse_k_stage_multipliers(args.k_stages)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_runs, summary = run_benchmark(args)

    raw_path = output_dir / "raw_runs.json"
    raw_path.write_text(json.dumps(raw_runs, indent=2), encoding="utf-8")
    print(f"Saved raw runs to {format_output_path(raw_path)}")

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON to {format_output_path(summary_json_path)}")

    summary_md_path = output_dir / "summary.md"
    write_summary_markdown(summary, args.success_threshold, summary_md_path)
    print(f"Saved summary Markdown to {format_output_path(summary_md_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
