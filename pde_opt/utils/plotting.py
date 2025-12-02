"""
Plotting utilities for PDE-constrained optimization examples.
Provides generic plotting functions that work with any example type.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import numpy as np


def plot_loss_curves(losses: list, title: str = "Training Loss", figsize: Tuple[int, int] = (10, 4)):
    """
    Plot loss curves over training iterations.

    Args:
        losses: List of loss values
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Linear scale
    ax1.plot(losses, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} (Linear Scale)')
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(losses, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title(f'{title} (Log Scale)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_1d_solution(x_grid: jnp.ndarray, u_pred: jnp.ndarray, u_true: Optional[jnp.ndarray] = None,
                     title: str = "1D Solution", xlabel: str = "x", ylabel: str = "u(x)",
                     figsize: Tuple[int, int] = (8, 5)):
    """
    Plot 1D solution comparison.

    Args:
        x_grid: Spatial grid points
        u_pred: Predicted solution
        u_true: True/target solution (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if u_true is not None:
        ax.plot(x_grid, u_true, 'b-', linewidth=2, label='True')
        ax.plot(x_grid, u_pred, 'r--', linewidth=2, label='Predicted')
        ax.legend()
    else:
        ax.plot(x_grid, u_pred, 'b-', linewidth=2, label='Solution')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_1d_force(x_grid: jnp.ndarray, f_pred: jnp.ndarray, f_true: Optional[jnp.ndarray] = None,
                  title: str = "Force Field", xlabel: str = "x", ylabel: str = "f(x)",
                  figsize: Tuple[int, int] = (8, 5)):
    """
    Plot 1D force field comparison.

    Args:
        x_grid: Spatial grid points
        f_pred: Predicted force
        f_true: True force (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if f_true is not None:
        ax.plot(x_grid, f_true, 'b-', linewidth=2, label='True')
        ax.plot(x_grid, f_pred, 'r--', linewidth=2, label='Predicted')
        ax.legend()
    else:
        ax.plot(x_grid, f_pred, 'b-', linewidth=2, label='Force')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spacetime_heatmap(data: jnp.ndarray, x_grid: jnp.ndarray, t_grid: jnp.ndarray,
                           title: str = "Space-Time Evolution", xlabel: str = "x", ylabel: str = "t",
                           figsize: Tuple[int, int] = (10, 6), cmap: str = 'viridis',
                           vmin: Optional[float] = None, vmax: Optional[float] = None):
    """
    Plot 1+1D space-time evolution as heatmap.

    Args:
        data: 2D array of shape (nx, nt) or (nt, nx)
        x_grid: Spatial grid points
        t_grid: Temporal grid points
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for color scale (optional)
        vmax: Maximum value for color scale (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure data is (nx, nt)
    if data.shape[0] == len(t_grid) and data.shape[1] == len(x_grid):
        data = data.T

    im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')

    plt.tight_layout()
    return fig


def plot_spacetime_snapshots(data: jnp.ndarray, x_grid: jnp.ndarray, t_grid: jnp.ndarray,
                             num_snapshots: int = 5, title_prefix: str = "Solution",
                             xlabel: str = "x", ylabel: str = "u(x,t)",
                             figsize: Tuple[int, int] = (12, 4)):
    """
    Plot temporal snapshots of 1+1D solution.

    Args:
        data: 2D array of shape (nx, nt)
        x_grid: Spatial grid points
        t_grid: Temporal grid points
        num_snapshots: Number of time snapshots to plot
        title_prefix: Prefix for subplot titles
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    # Ensure data is (nx, nt)
    if data.shape[0] == len(t_grid) and data.shape[1] == len(x_grid):
        data = data.T

    nt = len(t_grid)
    snapshot_indices = np.linspace(0, nt - 1, num_snapshots, dtype=int)

    fig, axes = plt.subplots(1, num_snapshots, figsize=figsize, sharey=True)
    if num_snapshots == 1:
        axes = [axes]

    for i, (ax, idx) in enumerate(zip(axes, snapshot_indices)):
        ax.plot(x_grid, data[:, idx], linewidth=2)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(f'{title_prefix} at t={t_grid[idx]:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_2d_heatmap(data: jnp.ndarray, x_grid: jnp.ndarray, y_grid: jnp.ndarray,
                    title: str = "2D Solution", xlabel: str = "x", ylabel: str = "y",
                    figsize: Tuple[int, int] = (8, 6), cmap: str = 'viridis'):
    """
    Plot 2D spatial solution as heatmap.

    Args:
        data: 2D array of shape (nx, ny)
        x_grid: X spatial grid points
        y_grid: Y spatial grid points
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')

    plt.tight_layout()
    return fig


def plot_2d_snapshots(data: jnp.ndarray, x_grid: jnp.ndarray, y_grid: jnp.ndarray,
                      t_grid: jnp.ndarray, num_snapshots: int = 4,
                      title_prefix: str = "Solution", xlabel: str = "x", ylabel: str = "y",
                      figsize: Tuple[int, int] = (12, 10), cmap: str = 'viridis',
                      vmin: Optional[float] = None, vmax: Optional[float] = None):
    """
    Plot temporal snapshots of 2+1D solution.

    Args:
        data: 3D array of shape (nx, ny, nt)
        x_grid: X spatial grid points
        y_grid: Y spatial grid points
        t_grid: Temporal grid points
        num_snapshots: Number of time snapshots to plot
        title_prefix: Prefix for subplot titles
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for color scale (optional, computed from data if not provided)
        vmax: Maximum value for color scale (optional, computed from data if not provided)
    """
    nt = data.shape[2]
    snapshot_indices = np.linspace(0, nt - 1, num_snapshots, dtype=int)

    # Use provided vmin/vmax or compute from data
    if vmin is None:
        vmin = float(jnp.min(data))
    if vmax is None:
        vmax = float(jnp.max(data))

    ncols = min(2, num_snapshots)
    nrows = (num_snapshots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_snapshots == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for i, idx in enumerate(snapshot_indices):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        im = ax.imshow(data[:, :, idx].T, aspect='auto', origin='lower', cmap=cmap,
                      extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                      vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_prefix} at t={t_grid[idx]:.3f}')
        plt.colorbar(im, ax=ax, label='Value')

    # Hide empty subplots
    for i in range(num_snapshots, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def plot_comparison_heatmaps(data1: jnp.ndarray, data2: jnp.ndarray,
                            x_grid: jnp.ndarray, y_grid: jnp.ndarray,
                            title1: str = "True", title2: str = "Predicted",
                            xlabel: str = "x", ylabel: str = "y",
                            figsize: Tuple[int, int] = (14, 5), cmap: str = 'viridis'):
    """
    Plot two 2D solutions side by side with difference.

    Args:
        data1: First 2D array (nx, ny)
        data2: Second 2D array (nx, ny)
        x_grid: X spatial grid points
        y_grid: Y spatial grid points
        title1: Title for first plot
        title2: Title for second plot
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Compute shared color scale for both datasets
    vmin = min(float(jnp.min(data1)), float(jnp.min(data2)))
    vmax = max(float(jnp.max(data1)), float(jnp.max(data2)))

    # First solution
    im1 = axes[0].imshow(data1.T, aspect='auto', origin='lower', cmap=cmap,
                        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                        vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title1)
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Second solution
    im2 = axes[1].imshow(data2.T, aspect='auto', origin='lower', cmap=cmap,
                        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                        vmin=vmin, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(title2)
    plt.colorbar(im2, ax=axes[1], label='Value')

    # Absolute error and relative L2 error
    abs_error = jnp.abs(data1 - data2)
    rel_l2_error = jnp.linalg.norm(abs_error.flatten()) / jnp.linalg.norm(data1.flatten())
    im3 = axes[2].imshow(abs_error.T, aspect='auto', origin='lower', cmap='hot',
                        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Absolute Error (Rel L2: {rel_l2_error:.2%}, max: {jnp.max(abs_error):.2e})')
    plt.colorbar(im3, ax=axes[2], label='Absolute Error')

    plt.tight_layout()
    return fig


def plot_example_results(example_name: str, solver, problem, params, losses, force, solution,
                         max_snapshots: int = 5, figsize_scale: float = 1.0):
    """
    Automatically plot results based on example type.

    Args:
        example_name: Name of the example (e.g., 'example-3.1')
        solver: Solver instance
        problem: Problem instance
        params: Trained parameters (or scalar force for simple examples)
        losses: List of loss values
        force: Predicted force field (flattened)
        solution: Predicted solution (flattened)
        max_snapshots: Maximum number of temporal snapshots for time-dependent problems
        figsize_scale: Scale factor for figure sizes

    Returns:
        Dictionary of figures
    """
    figures = {}

    # Always plot loss curves
    figures['loss'] = plot_loss_curves(losses, title=f"{example_name} - Training Loss")

    # Get grid information
    if hasattr(solver, 't_grid'):
        # Time-dependent problem
        x_grid = solver.x_grid
        t_grid = solver.t_grid

        if hasattr(solver, 'y_grid'):
            # 2+1D problem
            y_grid = solver.y_grid
            nx, ny, nt = solver.nx, solver.ny, solver.nt

            # Reshape solution and force
            u_pred = solution.reshape(nx, ny, nt)
            f_pred = force.reshape(nx, ny, nt)

            # Get true solution
            u_true_3d = jnp.stack([problem.analytical_solution(x_grid, y_grid, t)
                                   for t in t_grid], axis=-1)

            # Get true force at different time steps
            f_true_3d = jnp.stack([problem.source_term(x_grid, y_grid, t)
                                   for t in t_grid], axis=-1)

            # Compute shared color scales for solution (true vs predicted)
            sol_vmin = min(float(jnp.min(u_true_3d)), float(jnp.min(u_pred)))
            sol_vmax = max(float(jnp.max(u_true_3d)), float(jnp.max(u_pred)))

            # Compute shared color scales for force (true vs predicted)
            force_vmin = min(float(jnp.min(f_true_3d)), float(jnp.min(f_pred)))
            force_vmax = max(float(jnp.max(f_true_3d)), float(jnp.max(f_pred)))

            # Plot temporal snapshots of predicted solution
            figures['solution_snapshots'] = plot_2d_snapshots(
                u_pred, x_grid, y_grid, t_grid, num_snapshots=min(max_snapshots, nt),
                title_prefix="Predicted Solution", figsize=(12*figsize_scale, 10*figsize_scale),
                vmin=sol_vmin, vmax=sol_vmax
            )

            # Plot temporal snapshots of true solution
            figures['true_solution_snapshots'] = plot_2d_snapshots(
                u_true_3d, x_grid, y_grid, t_grid, num_snapshots=min(max_snapshots, nt),
                title_prefix="True Solution", figsize=(12*figsize_scale, 10*figsize_scale),
                vmin=sol_vmin, vmax=sol_vmax
            )

            # Plot temporal snapshots of predicted force
            figures['force_snapshots'] = plot_2d_snapshots(
                f_pred, x_grid, y_grid, t_grid, num_snapshots=min(max_snapshots, nt),
                title_prefix="Predicted Force", figsize=(12*figsize_scale, 10*figsize_scale),
                vmin=force_vmin, vmax=force_vmax
            )

            # Plot temporal snapshots of true force
            figures['true_force_snapshots'] = plot_2d_snapshots(
                f_true_3d, x_grid, y_grid, t_grid, num_snapshots=min(max_snapshots, nt),
                title_prefix="True Force", figsize=(12*figsize_scale, 10*figsize_scale),
                vmin=force_vmin, vmax=force_vmax
            )

            # Plot final time comparison for solution
            figures['final_solution_comparison'] = plot_comparison_heatmaps(
                u_true_3d[:, :, -1], u_pred[:, :, -1], x_grid, y_grid,
                title1=f"True Solution (t={t_grid[-1]:.3f})",
                title2=f"Predicted Solution (t={t_grid[-1]:.3f})",
                figsize=(14*figsize_scale, 5*figsize_scale)
            )

            # Plot final time comparison for force
            figures['final_force_comparison'] = plot_comparison_heatmaps(
                f_true_3d[:, :, -1], f_pred[:, :, -1], x_grid, y_grid,
                title1=f"True Force (t={t_grid[-1]:.3f})",
                title2=f"Predicted Force (t={t_grid[-1]:.3f})",
                figsize=(14*figsize_scale, 5*figsize_scale)
            )

        else:
            # 1+1D problem
            nx, nt = solver.nx, solver.nt

            # Reshape solution and force
            u_pred = solution.reshape(nx, nt)
            f_pred = force.reshape(nx, nt)

            # Get true solution
            u_true = problem.analytical_solution(x_grid, t_grid)

            # Get true force
            f_true = problem.source_term(x_grid, t_grid)

            # Plot space-time heatmaps - True vs Predicted Solution
            fig_sol, axes = plt.subplots(1, 3, figsize=(15*figsize_scale, 5*figsize_scale))

            # Compute shared color scale for true and predicted solutions
            sol_vmin = min(float(jnp.min(u_true)), float(jnp.min(u_pred)))
            sol_vmax = max(float(jnp.max(u_true)), float(jnp.max(u_pred)))

            # True solution
            im1 = axes[0].imshow(u_true.T, aspect='auto', origin='lower', cmap='viridis',
                                extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                                vmin=sol_vmin, vmax=sol_vmax)
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('t')
            axes[0].set_title('True Solution - Space-Time')
            plt.colorbar(im1, ax=axes[0], label='u(x,t)')

            # Predicted solution
            im2 = axes[1].imshow(u_pred.T, aspect='auto', origin='lower', cmap='viridis',
                                extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                                vmin=sol_vmin, vmax=sol_vmax)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('t')
            axes[1].set_title('Predicted Solution - Space-Time')
            plt.colorbar(im2, ax=axes[1], label='u(x,t)')

            # Absolute error and relative L2 error
            abs_error = jnp.abs(u_true - u_pred)
            rel_l2_error = jnp.linalg.norm(abs_error.flatten()) / jnp.linalg.norm(u_true.flatten())
            im3 = axes[2].imshow(abs_error.T, aspect='auto', origin='lower', cmap='hot',
                                extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]])
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('t')
            axes[2].set_title(f'Absolute Error (Rel L2: {rel_l2_error:.2%}, max: {jnp.max(abs_error):.2e})')
            plt.colorbar(im3, ax=axes[2], label='Absolute Error')

            plt.tight_layout()
            figures['solution_spacetime'] = fig_sol

            # Plot space-time heatmaps - True vs Predicted Force
            fig_force, axes_f = plt.subplots(1, 3, figsize=(15*figsize_scale, 5*figsize_scale))

            # Compute shared color scale for true and predicted forces
            force_vmin = min(float(jnp.min(f_true)), float(jnp.min(f_pred)))
            force_vmax = max(float(jnp.max(f_true)), float(jnp.max(f_pred)))

            # True force
            im1_f = axes_f[0].imshow(f_true.T, aspect='auto', origin='lower', cmap='viridis',
                                    extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                                    vmin=force_vmin, vmax=force_vmax)
            axes_f[0].set_xlabel('x')
            axes_f[0].set_ylabel('t')
            axes_f[0].set_title('True Force - Space-Time')
            plt.colorbar(im1_f, ax=axes_f[0], label='f(x,t)')

            # Predicted force
            im2_f = axes_f[1].imshow(f_pred.T, aspect='auto', origin='lower', cmap='viridis',
                                    extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                                    vmin=force_vmin, vmax=force_vmax)
            axes_f[1].set_xlabel('x')
            axes_f[1].set_ylabel('t')
            axes_f[1].set_title('Predicted Force - Space-Time')
            plt.colorbar(im2_f, ax=axes_f[1], label='f(x,t)')

            # Absolute error and relative L2 error
            abs_error_f = jnp.abs(f_true - f_pred)
            rel_l2_error_f = jnp.linalg.norm(abs_error_f.flatten()) / jnp.linalg.norm(f_true.flatten())
            im3_f = axes_f[2].imshow(abs_error_f.T, aspect='auto', origin='lower', cmap='hot',
                                    extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]])
            axes_f[2].set_xlabel('x')
            axes_f[2].set_ylabel('t')
            axes_f[2].set_title(f'Absolute Error (Rel L2: {rel_l2_error_f:.2%}, max: {jnp.max(abs_error_f):.2e})')
            plt.colorbar(im3_f, ax=axes_f[2], label='Absolute Error')

            plt.tight_layout()
            figures['force_spacetime'] = fig_force

            # Plot temporal snapshots - True vs Predicted
            snapshot_indices = np.linspace(0, nt - 1, min(max_snapshots, nt), dtype=int)
            fig_snaps, axes_snaps = plt.subplots(2, len(snapshot_indices),
                                                 figsize=(3*len(snapshot_indices)*figsize_scale, 6*figsize_scale),
                                                 sharex=True, sharey=True)
            if len(snapshot_indices) == 1:
                axes_snaps = axes_snaps.reshape(2, 1)

            for i, idx in enumerate(snapshot_indices):
                # True solution
                axes_snaps[0, i].plot(x_grid, u_true[:, idx], 'b-', linewidth=2)
                axes_snaps[0, i].set_title(f't={t_grid[idx]:.3f}')
                axes_snaps[0, i].grid(True, alpha=0.3)
                if i == 0:
                    axes_snaps[0, i].set_ylabel('True u(x,t)')

                # Predicted solution
                axes_snaps[1, i].plot(x_grid, u_pred[:, idx], 'r--', linewidth=2)
                axes_snaps[1, i].set_xlabel('x')
                axes_snaps[1, i].grid(True, alpha=0.3)
                if i == 0:
                    axes_snaps[1, i].set_ylabel('Predicted u(x,t)')

            plt.tight_layout()
            figures['solution_snapshots'] = fig_snaps

            # Plot comparison at final time
            fig, ax = plt.subplots(figsize=(8*figsize_scale, 5*figsize_scale))
            ax.plot(x_grid, u_true[:, -1], 'b-', linewidth=2, label='True')
            ax.plot(x_grid, u_pred[:, -1], 'r--', linewidth=2, label='Predicted')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x,t)')
            ax.set_title(f'Solution Comparison at t={t_grid[-1]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            figures['final_comparison'] = fig

    else:
        # Steady-state problem
        x_grid = solver.x_grid

        if hasattr(solver, 'y_grid'):
            # 2D problem
            y_grid = solver.y_grid
            nx, ny = solver.nx, solver.ny

            u_pred = solution.reshape(nx, ny)
            f_pred = force.reshape(nx, ny)

            # True solution
            u_true = problem.analytical_solution(x_grid, y_grid)

            # Plot comparison
            figures['solution_comparison'] = plot_comparison_heatmaps(
                u_true, u_pred, x_grid, y_grid,
                title1="True Solution", title2="Predicted Solution",
                figsize=(14*figsize_scale, 5*figsize_scale)
            )

            # Plot force
            figures['force'] = plot_2d_heatmap(
                f_pred, x_grid, y_grid, title="Predicted Force Field",
                figsize=(8*figsize_scale, 6*figsize_scale)
            )

        else:
            # 1D problem
            nx = solver.nx

            # Handle both vector and scalar forces
            if isinstance(force, (int, float, jnp.ndarray)) and jnp.size(force) == 1:
                # Scalar force (Example 3.1)
                f_pred = jnp.full(nx, float(force))
                f_true = jnp.full(nx, problem.force_value)
            else:
                # Vector force
                f_pred = force.reshape(nx)
                f_true = problem.source_term(x_grid)

            u_pred = solution.reshape(nx)
            u_true = problem.analytical_solution(x_grid)

            # Plot solution comparison
            figures['solution'] = plot_1d_solution(
                x_grid, u_pred, u_true, title="Solution Comparison",
                figsize=(8*figsize_scale, 5*figsize_scale)
            )

            # Plot force comparison
            figures['force'] = plot_1d_force(
                x_grid, f_pred, f_true, title="Force Field Comparison",
                figsize=(8*figsize_scale, 5*figsize_scale)
            )

    return figures