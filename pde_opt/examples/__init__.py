"""Examples module with ground-truth test cases."""

from pde_opt.examples.examples import (
    get_example,
    create_neural_network,
    create_network_from_config,
    ArchitectureConfig,
    ARCHITECTURE_CONFIGS,
    resolve_fourier_mode_count,
    fourier_complex_to_realimag,
    fourier_realimag_to_complex,
    OptimizationExample,
    Example31_Poisson1D_ScalarForce,
    Example32_Poisson1D_VectorForce,
    Example33_HeatEquation_ForceNN,
    Example33_HeatEquation_ForceNNFourier,
    Example35_LinearHeat2D,
    Example36_NonlinearHeat2D,
)

# VP example uses optional dependency (vp_solver)
try:
    from pde_opt.examples.vlasov_poisson import ExampleVP_FourierControl, VPResult
except ImportError:
    ExampleVP_FourierControl = None
    VPResult = None

__all__ = [
    'get_example',
    'create_neural_network',
    'create_network_from_config',
    'ArchitectureConfig',
    'ARCHITECTURE_CONFIGS',
    'resolve_fourier_mode_count',
    'fourier_complex_to_realimag',
    'fourier_realimag_to_complex',
    'OptimizationExample',
    'Example31_Poisson1D_ScalarForce',
    'Example32_Poisson1D_VectorForce',
    'Example33_HeatEquation_ForceNN',
    'Example33_HeatEquation_ForceNNFourier',
    'Example35_LinearHeat2D',
    'Example36_NonlinearHeat2D',
    'ExampleVP_FourierControl',
    'VPResult',
]
