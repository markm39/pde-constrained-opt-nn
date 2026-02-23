"""Utility functions for solver validation and visualization."""

from pde_opt.utils.solver_validation import validate_solver, find_optimal_config
from pde_opt.utils.plotting import plot_example_results, plot_loss_curves
from pde_opt.utils.vp_plotting import plot_vp_results

__all__ = [
    'validate_solver',
    'find_optimal_config',
    'plot_example_results',
    'plot_loss_curves',
    'plot_vp_results',
]
