"""Examples module with ground-truth test cases."""

from pde_opt.examples.examples import (
    get_example,
    create_neural_network,
    OptimizationExample,
    Example31_Poisson1D_ScalarForce,
    Example32_Poisson1D_VectorForce,
    Example33_HeatEquation_ForceNN,
    Example35_LinearHeat2D,
    Example36_NonlinearHeat2D,
)

__all__ = [
    'get_example',
    'create_neural_network',
    'OptimizationExample',
    'Example31_Poisson1D_ScalarForce',
    'Example32_Poisson1D_VectorForce',
    'Example33_HeatEquation_ForceNN',
    'Example35_LinearHeat2D',
    'Example36_NonlinearHeat2D',
]
