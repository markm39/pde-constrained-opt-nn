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
        'poisson-2d': Poisson2D,
        'nonlinear-heat-2d': NonlinearHeat2D,
        'wave-1d': WaveEquation1D,
        'advection-diffusion-1d': AdvectionDiffusion1D,
    }

    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}")

    return problems[problem_name](**kwargs)