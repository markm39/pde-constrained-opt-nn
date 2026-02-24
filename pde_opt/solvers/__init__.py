"""Solvers module for different PDE discretization schemes."""

from pde_opt.solvers.solvers import (
    get_solver,
    HeatEquationFD,
    HeatEquationFEM,
    HeatEquationCrankNicolson,
    Heat2DCrankNicolson,
    NonlinearHeat2DCrankNicolson,  # Backwards compatibility alias
    WaveEquationFD,
    Poisson1DFD,
    PoissonFD,
    AdvectionDiffusionFD,
    Helmholtz2DFD,
    Wave2DFD,
    VariableDiffusion1DFD,
)

__all__ = [
    'get_solver',
    'HeatEquationFD',
    'HeatEquationFEM',
    'HeatEquationCrankNicolson',
    'Heat2DCrankNicolson',
    'NonlinearHeat2DCrankNicolson',  # Backwards compatibility alias
    'WaveEquationFD',
    'Poisson1DFD',
    'PoissonFD',
    'AdvectionDiffusionFD',
    'Helmholtz2DFD',
    'Wave2DFD',
    'VariableDiffusion1DFD',
]
