"""Plotting functions for Vlasov-Poisson optimization results.

Produces figures for phase-space distributions, electric energy traces,
external field profiles, and optimization convergence. Returns Dict[str, Figure]
matching the pattern used by plot_example_results().
"""

from typing import Dict

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_vp_results(result, figsize_scale: float = 1.0) -> Dict[str, Figure]:
    """Plot comprehensive VP optimization results.

    Args:
        result: VPResult from ExampleVP_FourierControl.run()
        figsize_scale: Scale factor for figure sizes

    Returns:
        Dictionary of named figures for saving
    """
    figures = {}

    figures['loss'] = _plot_loss(result, figsize_scale)
    figures['electric_energy'] = _plot_electric_energy(result, figsize_scale)
    figures['phase_space'] = _plot_phase_space(result, figsize_scale)
    figures['external_field'] = _plot_external_field(result, figsize_scale)
    figures['electric_field_snapshots'] = _plot_electric_field_snapshots(result, figsize_scale)

    return figures


def _plot_loss(result, scale: float) -> Figure:
    """Objective function convergence."""
    fig, ax = plt.subplots(figsize=(8 * scale, 5 * scale))
    ax.plot(result.losses, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title(f'Convergence ({result.cost_type.upper()} cost)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_electric_energy(result, scale: float) -> Figure:
    """Electric energy over time: controlled vs uncontrolled baseline."""
    num_steps = len(result.ee_array)
    t_values = jnp.linspace(0, float(result.config.t_final), num_steps)

    fig, ax = plt.subplots(figsize=(10 * scale, 5 * scale))
    ax.plot(t_values, result.ee_baseline, label='No control (H=0)',
            alpha=0.7, linestyle='--')
    ax.plot(t_values, result.ee_array, label='Optimized H(x)',
            linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('Electric energy')
    ax.set_title('Instability suppression: electric energy over time')
    ax.legend()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_phase_space(result, scale: float) -> Figure:
    """Side-by-side: equilibrium f_eq and final controlled f(T)."""
    mesh = result.mesh
    extent = [float(mesh.xs[0]), float(mesh.xs[-1]),
              float(mesh.vs[0]), float(mesh.vs[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(18 * scale, 5 * scale))

    # Equilibrium
    im0 = axes[0].imshow(result.f_eq.T, extent=extent,
                          aspect='auto', origin='lower', cmap='plasma')
    axes[0].set_title('Equilibrium $f_{eq}(x,v)$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('v')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Final controlled distribution
    im1 = axes[1].imshow(result.f_final.T, extent=extent,
                          aspect='auto', origin='lower', cmap='plasma')
    axes[1].set_title(f'Controlled $f(T={float(result.config.t_final):.0f}, x, v)$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference from equilibrium
    diff = result.f_final - result.f_eq
    vmax = max(abs(float(jnp.min(diff))), abs(float(jnp.max(diff))))
    if vmax < 1e-15:
        vmax = 1.0
    im2 = axes[2].imshow(diff.T, extent=extent,
                          aspect='auto', origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
    axes[2].set_title('$f(T) - f_{eq}$')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('v')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def _plot_external_field(result, scale: float) -> Figure:
    """Optimized external electric field H(x)."""
    mesh = result.mesh

    fig, ax = plt.subplots(figsize=(10 * scale, 4 * scale))
    ax.plot(mesh.xs, result.H_field, linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('H(x)')
    ax.set_title('Optimized external field H(x)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_electric_field_snapshots(result, scale: float,
                                    n_snapshots: int = 4) -> Figure:
    """Self-generated electric field E(t,x) at selected time snapshots."""
    mesh = result.mesh
    num_steps = result.E_array.shape[0]
    indices = jnp.linspace(0, num_steps - 1, n_snapshots + 1, dtype=int)[1:]
    dt = float(result.config.dt)

    fig, ax = plt.subplots(figsize=(10 * scale, 5 * scale))

    # Plot H(x) for reference
    ax.plot(mesh.xs, result.H_field, label='H(x)', linewidth=2,
            color='black', linestyle='--')

    # Plot E_self = E_total - H at selected times
    for idx in indices:
        t_val = float(idx) * dt
        E_self = result.E_array[idx] - result.H_field
        ax.plot(mesh.xs, E_self, label=f'E(t={t_val:.0f}, x)', alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('Field')
    ax.set_title('Electric fields: H(x) and self-generated E(t,x)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
