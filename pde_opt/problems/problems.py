"""
Problems module defining different PDE types and their properties.
Based on examples from the paper: https://arxiv.org/abs/2408.12404
"""

import jax.numpy as jnp
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass

HELMHOLTZ_PROFILES = (
    'gaussian_lens',
    'double_lens',
    'circular_inclusion',
    'layered',
)


@dataclass
class PDEProblem:
    """Base class for PDE problems."""

    name: str
    description: str
    domain: Tuple[float, ...]  # Spatial and/or temporal domain bounds
    boundary_conditions: str
    parameters: Dict[str, Any]

    def source_term(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Define the source/forcing term."""
        raise NotImplementedError

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Define initial condition for time-dependent problems."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Analytical solution if available."""
        return None


class Poisson1DScalar(PDEProblem):
    """
    1D Poisson with scalar force (Example 3.1 from paper).
    -u''(x) = f, x  (0,1)
    u(0) = u(1) = 0
    """

    def __init__(self, force_value: float = -1.0):
        super().__init__(
            name="1D Poisson (Scalar Force)",
            description="1D Poisson equation with spatially-constant scalar force",
            domain=(0.0, 1.0),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"force": force_value}
        )
        self.force_value = force_value

    def source_term(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Constant force."""
        return self.force_value * jnp.ones_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """u(x) = -f/2 * x * (1-x)"""
        return -self.force_value / 2.0 * x * (1.0 - x)


class Poisson1DVector(PDEProblem):
    """
    1D Poisson with vector force (Example 3.2 from paper).
    -u''(x) = f(x), x  (0,1)
    u(0) = u(1) = 0
    """

    def __init__(self):
        super().__init__(
            name="1D Poisson (Vector Force)",
            description="1D Poisson equation with spatially-variable force",
            domain=(0.0, 1.0),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={}
        )

    def source_term(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """f(x) = pi^2 * sin(pi*x)"""
        return jnp.pi**2 * jnp.sin(jnp.pi * x)

    def analytical_solution(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """u(x) = sin(pi*x)"""
        return jnp.sin(jnp.pi * x)


class HeatEquation1D(PDEProblem):
    """
    1+1D Heat equation (Example 3.3 from paper).
    u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = 0 (zero initial condition)

    Target solution: u(x,t) = sin(π*x)sin(π*t)
    This grows from zero IC with appropriate forcing term.
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="1+1D Heat Equation",
            description="Heat equation in 1D space + time",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(π*x)sin(π*t).

        Derivation:
        ∂u/∂t = π*sin(π*x)cos(π*t)
        ∂²u/∂x² = -π²*sin(π*x)sin(π*t)
        f = ∂u/∂t - ∂²u/∂x² = π*sin(π*x)[cos(π*t) + π*sin(π*t)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.pi * jnp.sin(jnp.pi * X) * (jnp.cos(jnp.pi * T_mesh) + jnp.pi * jnp.sin(jnp.pi * T_mesh))

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero initial condition."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(π*x)sin(π*t)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * T_mesh)


class Poisson2D(PDEProblem):
    """
    2D Poisson equation (Example 3.9 from paper).
    -�(�(x,y)u) = f(x,y), (x,y)  (0,1)�
    u = 0 on �
    """

    def __init__(self):
        super().__init__(
            name="2D Poisson",
            description="2D Poisson equation with spatially-variable diffusion",
            domain=(0.0, 1.0, 0.0, 1.0),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={}
        )

    def diffusion_coefficient(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """�(x,y) = 1 + 2x + 3y�"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        return 1.0 + 2.0 * X + 3.0 * Y**2

    def source_term(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Source term for analytical solution sin(�x)sin(�y)."""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        f = (-6.0 * jnp.pi * Y * jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y) +
             2.0 * jnp.pi**2 * (2.0 * X + 3.0 * Y**2 + 1.0) * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) -
             2.0 * jnp.pi * jnp.sin(jnp.pi * Y) * jnp.cos(jnp.pi * X))
        return f

    def analytical_solution(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """u(x,y) = sin(�x)sin(�y)"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


class HeatEquation1DOscillating(PDEProblem):
    """
    1+1D Heat equation with highly oscillating target solution.
    u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = 0 (zero initial condition)

    Target solution: u(x,t) = sin(k*π*x)sin(π*t)
    This grows from zero IC with appropriate forcing term.
    """

    def __init__(self, T: float = 1.0, n_oscillations: int = 10):
        super().__init__(
            name=f"1+1D Heat Equation (Oscillating k={n_oscillations})",
            description=f"Heat equation with {n_oscillations} spatial oscillations",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T, "n_oscillations": n_oscillations}
        )
        self.T = T
        self.k = n_oscillations  # Number of spatial oscillations

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(k*π*x)sin(π*t).

        Derivation:
        ∂u/∂t = π*sin(k*π*x)cos(π*t)
        ∂²u/∂x² = -k²*π²*sin(k*π*x)sin(π*t)
        f = ∂u/∂t - ∂²u/∂x² = π*sin(k*π*x)[cos(π*t) + k²*π*sin(π*t)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        k_pi = self.k * jnp.pi
        return jnp.pi * jnp.sin(k_pi * X) * (jnp.cos(jnp.pi * T_mesh) + k_pi * self.k * jnp.sin(jnp.pi * T_mesh))

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero initial condition."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(k*π*x)sin(π*t)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(self.k * jnp.pi * X) * jnp.sin(jnp.pi * T_mesh)


class HeatEquation1DOscillatingCosine(PDEProblem):
    """
    1+1D Heat equation with highly oscillating target solution (cosine in time variant).
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(πωx) (non-zero initial condition)

    Target solution: u(x,t) = sin(πωx)cos(πωt)
    This starts from non-zero IC and oscillates with cosine temporal behavior.
    """

    def __init__(self, T: float = 1.0, n_oscillations: int = 10):
        super().__init__(
            name=f"1+1D Heat Equation (Oscillating Cosine ω={n_oscillations})",
            description=f"Heat equation with {n_oscillations} spatial oscillations and cosine time",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T, "n_oscillations": n_oscillations}
        )
        self.T = T
        self.omega = n_oscillations  # Frequency parameter

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(πωx)cos(πωt).

        Derivation:
        ∂u/∂t = -πω·sin(πωx)sin(πωt)
        ∂²u/∂x² = -(πω)²·sin(πωx)cos(πωt)
        f = ∂u/∂t - ∂²u/∂x² = -πω·sin(πωx)sin(πωt) + (πω)²·sin(πωx)cos(πωt)
                              = sin(πωx)[π²ω²·cos(πωt) - πω·sin(πωt)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        pi_omega = jnp.pi * self.omega
        spatial = jnp.sin(pi_omega * X)
        temporal = pi_omega**2 * jnp.cos(pi_omega * T_mesh) - pi_omega * jnp.sin(pi_omega * T_mesh)
        return spatial * temporal

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Initial condition: u(x,0) = sin(πωx)"""
        return jnp.sin(jnp.pi * self.omega * x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(πωx)cos(πωt)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(jnp.pi * self.omega * X) * jnp.cos(jnp.pi * self.omega * T_mesh)


class HeatEquation1DCosine(PDEProblem):
    """
    1+1D Heat equation with cosine temporal behavior.
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(2πx) (non-zero initial condition)

    Target solution: u(x,t) = sin(2πx)cos(2πt)
    This starts from non-zero IC and oscillates with cosine temporal behavior.
    """

    def __init__(self, T: float = 0.5):
        super().__init__(
            name="1+1D Heat Equation (Cosine)",
            description="Heat equation with cosine temporal oscillation",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(2πx)cos(2πt).

        Derivation:
        ∂u/∂t = -2π·sin(2πx)sin(2πt)
        ∂²u/∂x² = -4π²·sin(2πx)cos(2πt)
        f = ∂u/∂t - ∂²u/∂x² = sin(2πx)·[4π²·cos(2πt) - 2π·sin(2πt)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        spatial = jnp.sin(2 * jnp.pi * X)
        temporal = 4 * jnp.pi**2 * jnp.cos(2 * jnp.pi * T_mesh) - 2 * jnp.pi * jnp.sin(2 * jnp.pi * T_mesh)
        return spatial * temporal

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Initial condition: u(x,0) = sin(2πx)"""
        return jnp.sin(2 * jnp.pi * x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(2πx)cos(2πt)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * T_mesh)


class HeatEquation1DMixed(PDEProblem):
    """
    1+1D Heat equation with mixed positive/negative forcing.
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = 0 (zero initial condition)

    Target solution: u(x,t) = sin(πx)·(3t - 4t²)
    This has smooth spatial structure but forcing that transitions from positive to negative.
    The forcing is positive for t < ~0.65 and negative for t > ~0.65.
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="1+1D Heat Equation (Mixed Forcing)",
            description="Heat equation with forcing that has positive and negative regions",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(πx)·(3t - 4t²).

        Derivation:
        ∂u/∂t = sin(πx)·(3 - 8t)
        ∂²u/∂x² = -π²·sin(πx)·(3t - 4t²)
        f = ∂u/∂t - ∂²u/∂x² = sin(πx)·[(3 - 8t) + π²(3t - 4t²)]

        This forcing is:
        - Positive for t < ~0.65 (early times)
        - Negative for t > ~0.65 (late times)
        - Ranges from ~+6 at early times to ~-9 at t=1
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        spatial = jnp.sin(jnp.pi * X)
        temporal = (3.0 - 8.0 * T_mesh) + jnp.pi**2 * (3.0 * T_mesh - 4.0 * T_mesh**2)
        return spatial * temporal

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero initial condition."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(πx)·(3t - 4t²)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(jnp.pi * X) * (3.0 * T_mesh - 4.0 * T_mesh**2)


class HeatEquation1DSpatialMixed(PDEProblem):
    """
    1+1D Heat equation with spatial sign variation in forcing.
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = 0 (zero initial condition)

    Target solution: u(x,t) = sin(2πx) * sin(πt)

    At each time t, both u and f have spatial sign changes:
    - Positive for x ∈ (0, 0.5)
    - Negative for x ∈ (0.5, 1)
    This tests ReLU networks on spatially varying positive/negative forcing.
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="1+1D Heat Equation (Spatial Mixed)",
            description="Heat equation with forcing that has spatial positive/negative regions",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(2πx) * sin(πt).

        Derivation:
        ∂u/∂t = π·cos(πt)·sin(2πx)
        ∂²u/∂x² = -4π²·sin(2πx)·sin(πt)
        f = ∂u/∂t - ∂²u/∂x² = sin(2πx)·[π·cos(πt) + 4π²·sin(πt)]

        The forcing has the same spatial sign pattern as u:
        - Positive for x ∈ (0, 0.5), negative for x ∈ (0.5, 1)
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        spatial = jnp.sin(2.0 * jnp.pi * X)
        temporal = jnp.pi * jnp.cos(jnp.pi * T_mesh) + 4.0 * jnp.pi**2 * jnp.sin(jnp.pi * T_mesh)
        return spatial * temporal

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero initial condition."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(2πx) * sin(πt)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(2.0 * jnp.pi * X) * jnp.sin(jnp.pi * T_mesh)


class HeatEquation1DMultiMode(PDEProblem):
    """
    1+1D Heat equation with multiple spatial modes.
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = 0 (zero initial condition)

    Target solution: u(x,t) = [sin(πx) - 0.5·sin(2πx)] * sin(πt)

    The superposition of two modes creates multiple zero crossings in x.
    This tests ReLU networks on complex multi-scale spatial structure.
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="1+1D Heat Equation (Multi-Mode)",
            description="Heat equation with multi-mode spatial structure",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = [sin(πx) - 0.5·sin(2πx)] * sin(πt).

        Derivation:
        ∂u/∂t = π·cos(πt)·[sin(πx) - 0.5·sin(2πx)]
        ∂²u/∂x² = sin(πt)·[-π²·sin(πx) + 2π²·sin(2πx)]
        f = ∂u/∂t - ∂²u/∂x²
          = π·cos(πt)·[sin(πx) - 0.5·sin(2πx)] + π²·sin(πt)·[sin(πx) - 2·sin(2πx)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        sin_pi_x = jnp.sin(jnp.pi * X)
        sin_2pi_x = jnp.sin(2.0 * jnp.pi * X)
        cos_pi_t = jnp.cos(jnp.pi * T_mesh)
        sin_pi_t = jnp.sin(jnp.pi * T_mesh)

        term1 = jnp.pi * cos_pi_t * (sin_pi_x - 0.5 * sin_2pi_x)
        term2 = jnp.pi**2 * sin_pi_t * (sin_pi_x - 2.0 * sin_2pi_x)
        return term1 + term2

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero initial condition."""
        return jnp.zeros_like(x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = [sin(πx) - 0.5·sin(2πx)] * sin(πt)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        spatial = jnp.sin(jnp.pi * X) - 0.5 * jnp.sin(2.0 * jnp.pi * X)
        return spatial * jnp.sin(jnp.pi * T_mesh)


class HeatEquation1DSpatialMixedNonzeroIC(PDEProblem):
    """
    1+1D Heat equation with spatial sign variation and non-zero initial condition.
    ∂u/∂t - ∂²u/∂x² = f(x,t), (x,t) ∈ (0,1) × (0,T)
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(2πx) (non-zero initial condition with sign changes)

    Target solution: u(x,t) = sin(2πx) * cos(πt)

    Similar spatial structure to HeatEquation1DSpatialMixed but with:
    - Non-zero IC that itself has spatial sign variation
    - Cosine temporal behavior (starts at maximum, decays)
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="1+1D Heat Equation (Spatial Mixed, Non-zero IC)",
            description="Heat equation with spatial sign variation and non-zero IC",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet in space",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Force for target solution u(x,t) = sin(2πx) * cos(πt).

        Derivation:
        ∂u/∂t = -π·sin(πt)·sin(2πx)
        ∂²u/∂x² = -4π²·sin(2πx)·cos(πt)
        f = ∂u/∂t - ∂²u/∂x² = sin(2πx)·[-π·sin(πt) + 4π²·cos(πt)]
        """
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        spatial = jnp.sin(2.0 * jnp.pi * X)
        temporal = -jnp.pi * jnp.sin(jnp.pi * T_mesh) + 4.0 * jnp.pi**2 * jnp.cos(jnp.pi * T_mesh)
        return spatial * temporal

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Non-zero initial condition with sign changes: u(x,0) = sin(2πx)"""
        return jnp.sin(2.0 * jnp.pi * x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Target solution: u(x,t) = sin(2πx) * cos(πt)"""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(2.0 * jnp.pi * X) * jnp.cos(jnp.pi * T_mesh)


class LinearHeat2D(PDEProblem):
    """
    2+1D Linear heat equation (same as nonlinear but without u² term).
    ∂u/∂t - Δu = f, in Ω × I
    u = 0, on ∂Ω × I
    u = u₀ on Ω × {0}

    Supports different test cases via the 'prob' parameter.
    """

    def __init__(self, T: float = 1.0, prob: str = 'default'):
        super().__init__(
            name=f"2+1D Linear Heat ({prob})",
            description="Linear heat equation in 2D space + time",
            domain=(0.0, 1.0, 0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"T": T, "prob": prob}
        )
        self.T = T
        self.prob = prob

    def source_term(self, x: jnp.ndarray, y: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Source term for PDE: ∂u/∂t - Δu = f

        Different cases based on self.prob:
        - 'default': u(x,y,t) = exp(t-t²)sin(πx)sin(πy)
        - 'cossinsin': u(x,y,t) = sin(5πx)sin(5πy)sin(5πt)
        """
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        if self.prob == 'cossinsin':
            # For u(x,y,t) = sin(5πx)sin(5πy)sin(5πt)
            # ∂u/∂t = 5π·sin(5πx)sin(5πy)cos(5πt)
            # Δu = -(5π)²·2·sin(5πx)sin(5πy)sin(5πt) = -50π²·sin(5πx)sin(5πy)sin(5πt)
            # f = ∂u/∂t - Δu = sin(5πx)sin(5πy)·(5π·cos(5πt) + 50π²·sin(5πt))
            spatial = jnp.sin(5 * jnp.pi * X) * jnp.sin(5 * jnp.pi * Y)
            temporal = 5 * jnp.pi * jnp.cos(5 * jnp.pi * t) + 50 * jnp.pi**2 * jnp.sin(5 * jnp.pi * t)
            f = spatial * temporal
        else:
            # Default: u(x,y,t) = exp(t-t²)sin(πx)sin(πy)
            sin_term = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
            exp_term = jnp.exp(t - t**2)
            # f = ∂u/∂t - Δu (no u² term compared to nonlinear version)
            f = ((1.0 - 2.0 * t) * exp_term * sin_term +  # ∂u/∂t
                 2.0 * jnp.pi**2 * exp_term * sin_term)   # -Δu

        return f

    def initial_condition(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Initial condition u₀(x,y)"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        if self.prob == 'cossinsin':
            # u₀(x,y) = sin(5πx)sin(5πy)·sin(0) = 0
            return jnp.zeros_like(X)
        else:
            # Default: u₀(x,y) = sin(πx)sin(πy)
            return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    def analytical_solution(self, x: jnp.ndarray, y: jnp.ndarray, t: float) -> jnp.ndarray:
        """Analytical solution u(x,y,t)"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        if self.prob == 'cossinsin':
            # u(x,y,t) = sin(5πx)sin(5πy)sin(5πt)
            return jnp.sin(5 * jnp.pi * X) * jnp.sin(5 * jnp.pi * Y) * jnp.sin(5 * jnp.pi * t)
        else:
            # Default: u(x,y,t) = exp(t-t²)sin(πx)sin(πy)
            return jnp.exp(t - t**2) * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


class NonlinearHeat2D(PDEProblem):
    """
    2+1D Nonlinear heat equation (Example 3.6 from paper).
    u/t - �u + u� = f, in � � I
    u = 0, on � � I
    u = u� on � � {0}
    """

    def __init__(self, T: float = 1.0):
        super().__init__(
            name="2+1D Nonlinear Heat",
            description="Nonlinear heat equation in 2D space + time",
            domain=(0.0, 1.0, 0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"T": T}
        )
        self.T = T

    def source_term(self, x: jnp.ndarray, y: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Source term for PDE: ∂u/∂t - Δu + u² = f
        With u(x,y,t) = exp(t-t²)sin(πx)sin(πy)
        f = (1-2t)exp(t-t²)sin(πx)sin(πy) + 2π²exp(t-t²)sin(πx)sin(πy) + exp(2(t-t²))sin²(πx)sin²(πy)
        """
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        sin_term = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        exp_term = jnp.exp(t - t**2)

        # f = ∂u/∂t - Δu + u²
        f = ((1.0 - 2.0 * t) * exp_term * sin_term +  # ∂u/∂t
             2.0 * jnp.pi**2 * exp_term * sin_term +   # -Δu
             jnp.exp(2.0 * (t - t**2)) * sin_term**2)  # u²
        return f

    def initial_condition(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """u�(x,y) = sin(�x)sin(�y)"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    def analytical_solution(self, x: jnp.ndarray, y: jnp.ndarray, t: float) -> jnp.ndarray:
        """u(x,y,t) = exp(t-t�)sin(�x)sin(�y)"""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        return jnp.exp(t - t**2) * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


class WaveEquation1D(PDEProblem):
    """
    1+1D Wave equation.
    �u/t� - c��u/x� = f(x,t)
    u(0,t) = u(1,t) = 0
    u(x,0) = u�(x), u/t(x,0) = v�(x)
    """

    def __init__(self, c: float = 1.0, T: float = 1.0):
        super().__init__(
            name="1+1D Wave Equation",
            description="Wave equation in 1D space + time",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"c": c, "T": T}
        )
        self.c = c
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Force for standing wave solution."""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        # For solution u = sin(�x)cos(�ct)
        return jnp.zeros_like(X)  # Homogeneous wave equation

    def initial_condition(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initial displacement and velocity."""
        u0 = jnp.sin(jnp.pi * x)  # Initial displacement
        v0 = jnp.zeros_like(x)    # Initial velocity
        return u0, v0

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """u(x,t) = sin(�x)cos(�ct) - standing wave."""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * self.c * T_mesh)


class AdvectionDiffusion1D(PDEProblem):
    """
    1+1D Advection-Diffusion equation.
    u/t + vu/x - D�u/x� = f(x,t)
    """

    def __init__(self, v: float = 1.0, D: float = 0.1, T: float = 1.0):
        super().__init__(
            name="1+1D Advection-Diffusion",
            description="Advection-diffusion equation in 1D",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"v": v, "D": D, "T": T}
        )
        self.v = v
        self.D = D
        self.T = T

    def source_term(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Source term for traveling wave solution."""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        # For solution u = sin(�(x - vt))exp(-��Dt)
        return jnp.zeros_like(X)

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Initial condition."""
        return jnp.sin(jnp.pi * x)

    def analytical_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Traveling and diffusing wave."""
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        # Simplified solution for demonstration
        return jnp.sin(jnp.pi * (X - self.v * T_mesh)) * jnp.exp(-jnp.pi**2 * self.D * T_mesh)


# --- Inverse coefficient problems ---


class HelmholtzInverseMedium2D(PDEProblem):
    """2D Helmholtz inverse medium problem: recover refractive index n(x,y).

    PDE: -Delta u - k^2 n(x,y)^2 u = f(x,y)  on (0,1)^2
    u = 0 on boundary

    Given known source f and observed field u_obs, recover n(x,y).
    """

    def __init__(self, k: float = 10 * jnp.pi, profile: str = 'gaussian_lens'):
        if profile not in HELMHOLTZ_PROFILES:
            valid = ", ".join(HELMHOLTZ_PROFILES)
            raise ValueError(f"Unknown Helmholtz profile '{profile}'. Expected one of: {valid}")
        super().__init__(
            name="2D Helmholtz Inverse Medium",
            description="Recover refractive index from Helmholtz observations",
            domain=(0.0, 1.0, 0.0, 1.0),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"k": k, "profile": profile},
        )
        self.k_wavenum = k
        self.profile = profile

    def true_refractive_index(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """True n(x,y) on a meshgrid. Returns shape (nx, ny)."""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        if self.profile == 'gaussian_lens':
            # Smooth Gaussian anomaly centered at (0.5, 0.5)
            return 1.0 + 0.3 * jnp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
        elif self.profile == 'double_lens':
            # Harder smooth target with two offset Gaussian anomalies.
            lens_1 = 0.22 * jnp.exp(-((X - 0.35)**2 + (Y - 0.40)**2) / 0.012)
            lens_2 = 0.16 * jnp.exp(-((X - 0.70)**2 + (Y - 0.65)**2) / 0.020)
            return 1.0 + lens_1 + lens_2
        elif self.profile == 'circular_inclusion':
            # Sharp circular inclusion
            r = jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            return jnp.where(r < 0.2, 2.0, 1.0)
        elif self.profile == 'layered':
            # Horizontal layers
            return 1.0 + 0.3 * jnp.where(Y > 0.5, 1.0, 0.0)
        raise ValueError(f"Unhandled Helmholtz profile '{self.profile}'")

    def source_field(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Known source f(x,y). Returns flattened shape (nx*ny,)."""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        # Localized Gaussian source at (0.2, 0.2)
        f = 100.0 * jnp.exp(-((X - 0.2)**2 + (Y - 0.2)**2) / 0.005)
        return f.ravel()

    def generate_observations(self, solver) -> jnp.ndarray:
        """Forward-solve with true n to produce observation data."""
        n_true = self.true_refractive_index(solver.x_grid, solver.y_grid).ravel()
        f_vec = self.source_field(solver.x_grid, solver.y_grid)
        return solver.solve(n_true, f_vec)


class WaveInversion2D(PDEProblem):
    """2D full waveform inversion: recover wave speed c(x,y) from seismograms.

    PDE: (1/c^2) u_tt - Delta u = f(x,y,t)  on (0,1)^2 x (0,T)
    u = 0 on boundary, zero ICs.

    Source is a Ricker wavelet at a point. Receivers record u at fixed locations.
    """

    def __init__(self, c_profile: str = 'layered', n_receivers: int = 20,
                 source_loc: Tuple[float, float] = (0.5, 0.1),
                 peak_freq: float = 8.0, T: float = 1.0):
        super().__init__(
            name="2D Full Waveform Inversion",
            description="Recover wave speed from seismic receiver data",
            domain=(0.0, 1.0, 0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"c_profile": c_profile, "n_receivers": n_receivers,
                        "peak_freq": peak_freq, "T": T},
        )
        self.c_profile = c_profile
        self.n_receivers = n_receivers
        self.source_loc = source_loc
        self.peak_freq = peak_freq
        self.T = T

    def true_wave_speed(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """True c(x,y) on a meshgrid. Returns shape (nx, ny)."""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        if self.c_profile == 'layered':
            # Background + two velocity jumps
            c = 2.0 + 0.5 * jnp.tanh(20.0 * (Y - 0.3)) + 0.3 * jnp.tanh(20.0 * (Y - 0.7))
            return c
        elif self.c_profile == 'anomaly':
            # Background with circular high-velocity anomaly
            r = jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            return 2.0 + 1.0 * jnp.exp(-r**2 / 0.01)
        else:
            return 2.0 * jnp.ones_like(X)

    def ricker_wavelet(self, t: jnp.ndarray) -> jnp.ndarray:
        """Ricker wavelet (Mexican hat) centered at t_0 = 1.5/f_peak."""
        f = self.peak_freq
        t0 = 1.5 / f
        arg = (jnp.pi * f * (t - t0))**2
        return (1.0 - 2.0 * arg) * jnp.exp(-arg)

    def source_time_function(self, t: jnp.ndarray,
                             x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Spatial source * temporal wavelet. Returns (nt, nx*ny)."""
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        # Spatial: narrow Gaussian at source location
        xs, ys = self.source_loc
        spatial = jnp.exp(-((X - xs)**2 + (Y - ys)**2) / 0.001).ravel()  # (nx*ny,)
        temporal = self.ricker_wavelet(t)  # (nt,)
        return temporal[:, None] * spatial[None, :]  # (nt, nx*ny)

    def receiver_indices(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Indices into flattened (nx*ny,) array for receiver locations.

        Receivers placed along y = 0.9 at uniformly spaced x positions.
        """
        ny = len(y)
        # Find y-index closest to 0.9
        y_idx = jnp.argmin(jnp.abs(y - 0.9))
        # Uniformly spaced x indices
        nx = len(x)
        x_indices = jnp.linspace(0, nx - 1, self.n_receivers).astype(int)
        # Flattened index: i * ny + j (row-major, x varies first)
        return x_indices * ny + y_idx


class DiffusionCoefficientInverse1D(PDEProblem):
    """1D variable-coefficient diffusion inverse problem.

    PDE: u_t = d/dx(D(x) du/dx) + f(x,t)
    Recover D(x) from sparse observations of u.
    """

    def __init__(self, D_profile: str = 'sinusoidal', obs_fraction: float = 0.2,
                 n_time_obs: int = 5, T: float = 1.0, seed: int = 42):
        super().__init__(
            name="1D Diffusion Coefficient Inverse",
            description="Recover D(x) from sparse observations",
            domain=(0.0, 1.0, T),
            boundary_conditions="Homogeneous Dirichlet",
            parameters={"D_profile": D_profile, "obs_fraction": obs_fraction,
                        "n_time_obs": n_time_obs, "T": T},
        )
        self.D_profile = D_profile
        self.obs_fraction = obs_fraction
        self.n_time_obs = n_time_obs
        self.T = T
        self.seed = seed

    def true_diffusion(self, x: jnp.ndarray) -> jnp.ndarray:
        """True D(x). Returns shape (nx,)."""
        if self.D_profile == 'sinusoidal':
            return 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * x)
        elif self.D_profile == 'layered':
            return jnp.where(x < 0.5, 1.0, 3.0)
        elif self.D_profile == 'smooth_bump':
            return 1.0 + 2.0 * jnp.exp(-((x - 0.5)**2) / 0.01)
        else:
            return jnp.ones_like(x)

    def source_term(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Known source f(x,t)."""
        if t is None:
            return jnp.sin(jnp.pi * x)
        X, T_mesh = jnp.meshgrid(x, t, indexing='ij')
        return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * T_mesh)

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    def observation_mask(self, nx: int, nt: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random spatial indices and uniform time indices for observations.

        Returns:
            (spatial_indices, time_indices) -- integer arrays for subsampling.
        """
        import jax
        key = jax.random.PRNGKey(self.seed)
        n_spatial = max(1, int(self.obs_fraction * nx))
        spatial_idx = jax.random.choice(key, nx, shape=(n_spatial,), replace=False)
        spatial_idx = jnp.sort(spatial_idx)

        # Uniformly spaced time observations
        time_idx = jnp.linspace(0, nt - 1, self.n_time_obs).astype(int)
        return spatial_idx, time_idx


# Factory function to get problem by name
def get_problem(problem_name: str, **kwargs) -> PDEProblem:
    """
    Get problem instance by name.

    Args:
        problem_name: Name of the problem
        **kwargs: Additional arguments for problem constructor

    Returns:
        PDEProblem instance
    """
    problems = {
        'poisson-1d-scalar': Poisson1DScalar,
        'poisson-1d-vector': Poisson1DVector,
        'heat-1d': HeatEquation1D,
        'heat-1d-oscillating': HeatEquation1DOscillating,
        'heat-1d-oscillating-cosine': HeatEquation1DOscillatingCosine,
        'heat-1d-cosine': HeatEquation1DCosine,
        'heat-1d-mixed': HeatEquation1DMixed,
        'heat-1d-spatial-mixed': HeatEquation1DSpatialMixed,
        'heat-1d-multimode': HeatEquation1DMultiMode,
        'heat-1d-spatial-mixed-nonzero-ic': HeatEquation1DSpatialMixedNonzeroIC,
        'poisson-2d': Poisson2D,
        'linear-heat-2d': LinearHeat2D,
        'nonlinear-heat-2d': NonlinearHeat2D,
        'wave-1d': WaveEquation1D,
        'advection-diffusion-1d': AdvectionDiffusion1D,
        'helmholtz-inverse-2d': HelmholtzInverseMedium2D,
        'wave-inversion-2d': WaveInversion2D,
        'diffusion-coefficient-1d': DiffusionCoefficientInverse1D,
    }

    # VP problems use a different config type (VPProblemConfig, not PDEProblem)
    if problem_name.startswith('vp-'):
        from pde_opt.problems.vlasov_poisson import TwoStreamConfig, BumpOnTailConfig
        vp_problems = {
            'vp-two-stream': TwoStreamConfig,
            'vp-bump-on-tail': BumpOnTailConfig,
        }
        if problem_name not in vp_problems:
            raise ValueError(f"Unknown VP problem: {problem_name}")
        return vp_problems[problem_name](**kwargs)

    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}")

    return problems[problem_name](**kwargs)
