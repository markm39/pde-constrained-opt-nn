"""Tests for Helmholtz profile definitions, continuation, and benchmark helpers."""

import numpy as np
import jax.numpy as jnp

from pde_opt.examples import get_example
from pde_opt.problems import HELMHOLTZ_PROFILES, get_problem
from scripts.benchmark_helmholtz import summarize_runs
from scripts.run_example import parse_args, parse_k_stage_multipliers


def test_double_lens_profile_values():
    assert "double_lens" in HELMHOLTZ_PROFILES

    problem = get_problem("helmholtz-inverse-2d", k=5 * jnp.pi, profile="double_lens")
    x = jnp.array([0.35, 0.70])
    y = jnp.array([0.40, 0.65])
    n_true = problem.true_refractive_index(x, y)

    def expected_value(x_val: float, y_val: float) -> float:
        lens_1 = 0.22 * np.exp(-((x_val - 0.35) ** 2 + (y_val - 0.40) ** 2) / 0.012)
        lens_2 = 0.16 * np.exp(-((x_val - 0.70) ** 2 + (y_val - 0.65) ** 2) / 0.020)
        return 1.0 + lens_1 + lens_2

    assert n_true.shape == (2, 2)
    assert jnp.all(n_true > 1.0)
    assert np.isclose(float(n_true[0, 0]), expected_value(0.35, 0.40))
    assert np.isclose(float(n_true[1, 1]), expected_value(0.70, 0.65))


def test_grid_k_continuation_smoke_run():
    ex = get_example(
        "helmholtz-medium",
        k=2 * jnp.pi,
        nx=6,
        ny=6,
        profile="gaussian_lens",
        regularization=1e-4,
    )

    params, losses, field_pred, field_true = ex.run(
        max_iter=1,
        mode="grid",
        lr_schedule_type="exponential",
        seed=0,
        k_continuation=True,
        k_stages=[1.0, 2.0],
    )

    assert params.shape == (36,)
    assert len(losses) == 2
    assert np.all(np.isfinite(losses))
    assert field_pred.shape == (36,)
    assert field_true.shape == (36,)


def test_k_stage_parsing_from_cli():
    args = parse_args(["--example", "helmholtz", "--k-stages", "1,2.5,4"])
    assert args.k_stages == "1,2.5,4"
    assert parse_k_stage_multipliers(args.k_stages) == [1.0, 2.5, 4.0]


def test_helmholtz_benchmark_summary_aggregation():
    raw_runs = [
        {"profile": "gaussian_lens", "method": "nn", "rel_l2_coefficient": 0.04},
        {"profile": "gaussian_lens", "method": "nn", "rel_l2_coefficient": 0.08},
        {"profile": "gaussian_lens", "method": "grid", "rel_l2_coefficient": 0.60},
    ]

    summary = summarize_runs(raw_runs, success_threshold=0.05)
    summary_by_method = {row["method"]: row for row in summary}

    nn_row = summary_by_method["nn"]
    assert nn_row["num_runs"] == 2
    assert np.isclose(nn_row["best_rel_l2"], 0.04)
    assert np.isclose(nn_row["median_rel_l2"], 0.06)
    assert np.isclose(nn_row["success_rate"], 0.5)
    assert not nn_row["deterministic"]

    grid_row = summary_by_method["grid"]
    assert grid_row["num_runs"] == 1
    assert np.isclose(grid_row["best_rel_l2"], 0.60)
    assert grid_row["deterministic"]
