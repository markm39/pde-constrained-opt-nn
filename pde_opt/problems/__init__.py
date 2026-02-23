"""Problems module defining different PDE types and their properties."""

from pde_opt.problems.problems import (
    get_problem,
    PDEProblem,
    Poisson1DScalar,
    Poisson1DVector,
    HeatEquation1D,
    HeatEquation1DOscillating,
    HeatEquation1DOscillatingCosine,
    LinearHeat2D,
    NonlinearHeat2D,
    Poisson2D,
    WaveEquation1D,
    AdvectionDiffusion1D,
)
from pde_opt.problems.vlasov_poisson import (
    VPProblemConfig,
    TwoStreamConfig,
    BumpOnTailConfig,
)

__all__ = [
    'get_problem',
    'PDEProblem',
    'Poisson1DScalar',
    'Poisson1DVector',
    'HeatEquation1D',
    'HeatEquation1DOscillating',
    'HeatEquation1DOscillatingCosine',
    'LinearHeat2D',
    'NonlinearHeat2D',
    'Poisson2D',
    'WaveEquation1D',
    'AdvectionDiffusion1D',
    'VPProblemConfig',
    'TwoStreamConfig',
    'BumpOnTailConfig',
]
