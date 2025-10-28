"""Solvers module for different PDE discretization schemes."""

from pde_opt.solvers.solvers import (
    get_solver,
    HeatEquationFD,
    HeatEquationFEM,
    HeatEquationCrankNicolson,
    NonlinearHeat2DCrankNicolson,
    WaveEquationFD,
    Poisson1DFD,
    PoissonFD,
    AdvectionDiffusionFD,
)

__all__ = [
    'get_solver',
    'HeatEquationFD',
    'HeatEquationFEM',
    'HeatEquationCrankNicolson',
    'NonlinearHeat2DCrankNicolson',
    'WaveEquationFD',
    'Poisson1DFD',
    'PoissonFD',
    'AdvectionDiffusionFD',
]
