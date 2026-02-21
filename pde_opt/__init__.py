"""
PDE-Constrained Optimization with Neural Networks

A modular framework for solving inverse problems in PDEs using neural networks
to learn forcing terms, parameters, or boundary conditions.

Based on: https://arxiv.org/abs/2408.12404
"""

from pde_opt.examples import (
    get_example, create_neural_network, create_network_from_config,
    ArchitectureConfig, ARCHITECTURE_CONFIGS, resolve_fourier_mode_count,
)
from pde_opt.problems import get_problem
from pde_opt.solvers import get_solver
from pde_opt.utils.solver_validation import validate_solver, find_optimal_config
from pde_opt.utils.plotting import plot_example_results, plot_loss_curves

__version__ = "0.1.0"

__all__ = [
    'get_example',
    'create_neural_network',
    'create_network_from_config',
    'ArchitectureConfig',
    'ARCHITECTURE_CONFIGS',
    'resolve_fourier_mode_count',
    'get_problem',
    'get_solver',
    'validate_solver',
    'find_optimal_config',
    'plot_example_results',
    'plot_loss_curves',
]
