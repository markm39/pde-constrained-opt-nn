"""
Problems module defining different PDE types and their properties.
Based on examples from the paper: https://arxiv.org/abs/2408.12404
"""

import jax.numpy as jnp
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass


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
        'heat-1d-mixed': HeatEquation1DMixed,
        'poisson-2d': Poisson2D,
        'linear-heat-2d': LinearHeat2D,
        'nonlinear-heat-2d': NonlinearHeat2D,
        'wave-1d': WaveEquation1D,
        'advection-diffusion-1d': AdvectionDiffusion1D,
    }

    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}")

    return problems[problem_name](**kwargs)