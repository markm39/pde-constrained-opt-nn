"""Tests for Fourier-space study utilities and Example 3.3 Fourier runner."""

import numpy as np
import jax
import jax.numpy as jnp

from pde_opt.examples import (
    get_example,
    create_network_from_config,
    ARCHITECTURE_CONFIGS,
    resolve_fourier_mode_count,
    fourier_complex_to_realimag,
    fourier_realimag_to_complex,
)


def test_fourier_realimag_roundtrip():
    coeffs = jnp.array([1.0 + 2.0j, -0.5 + 0.3j, 0.2 - 1.1j], dtype=jnp.complex64)
    encoded = fourier_complex_to_realimag(coeffs)
    decoded = fourier_realimag_to_complex(encoded)
    assert decoded.shape == coeffs.shape
    assert jnp.allclose(decoded, coeffs, atol=1e-6)


def test_resolve_fourier_mode_count():
    n_modes_full, full_modes = resolve_fourier_mode_count('full', nx=15)
    assert full_modes == 8
    assert n_modes_full == 8

    n_modes_small, _ = resolve_fourier_mode_count(4, nx=15)
    assert n_modes_small == 4

    n_modes_clamped, _ = resolve_fourier_mode_count(64, nx=15)
    assert n_modes_clamped == 8


def test_create_network_from_config_output_dim():
    config = ARCHITECTURE_CONFIGS[0]
    model = create_network_from_config(config, output_dim=6)

    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 5)))
    output = model.apply(params, jnp.zeros((2, 5)))
    assert output.shape == (2, 6)


def test_fourier_example33_smoke_run():
    config = ARCHITECTURE_CONFIGS[0]
    ex = get_example('example-3.3-fourier', problem_name='heat-1d', nx=15, nt=4)
    n_modes, _ = resolve_fourier_mode_count(8, ex.grid_params['nx'])
    model = create_network_from_config(config, output_dim=2 * n_modes)

    params, losses, force, solution = ex.run(
        max_iter=1,
        model=model,
        lr_schedule_type='exponential',
        seed=0,
        input_scheme='state_time',
        mode_budget=8,
    )

    assert params is not None
    assert len(losses) == 1
    assert not np.isnan(losses[-1])
    expected_size = ex.grid_params['nx'] * ex.grid_params['nt']
    assert force.shape[0] == expected_size
    assert solution.shape[0] == expected_size
