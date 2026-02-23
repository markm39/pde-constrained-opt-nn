"""Vlasov-Poisson problem configurations for plasma instability suppression.

Defines equilibrium distributions and domain parameters for the 1D-1D
Vlasov-Poisson system. These do NOT inherit from PDEProblem because the
VP system has fundamentally different structure: phase space (x,v),
no analytical solution for the controlled case, and a static external field
H(x) instead of a space-time source term.

Reference: arXiv:2504.10435
"""

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp


@dataclass
class VPProblemConfig:
    """Configuration for a Vlasov-Poisson problem."""

    name: str
    description: str

    # Domain parameters
    length_x: float      # Spatial period L_x
    length_v: float      # Velocity half-range L_v (domain is [-L_v, L_v])
    t_final: float       # Final time T

    # Discretization defaults
    nx: int              # Number of spatial grid points
    nv: int              # Number of velocity grid points
    dt: float            # Time step

    # Perturbation
    epsilon: float       # Perturbation amplitude for initial condition
    k_0: float           # Fundamental wavenumber 2*pi/L_x

    # External field parameterization
    n_fourier_modes: int  # Default number of Fourier modes for H(x)

    def make_f_eq(self, mesh: Any) -> jnp.ndarray:
        """Compute equilibrium distribution f_eq(x, v) on given mesh.

        Args:
            mesh: vp_solver.Mesh with attributes V (velocity meshgrid)

        Returns:
            Array of shape (nx, nv)
        """
        raise NotImplementedError

    def make_f_iv(self, mesh: Any, f_eq: jnp.ndarray) -> jnp.ndarray:
        """Compute perturbed initial condition f_iv(x, v).

        Default: f_iv = (1 + epsilon * cos(k_0 * x)) * f_eq

        Args:
            mesh: vp_solver.Mesh with attributes X (position meshgrid)
            f_eq: Equilibrium distribution, shape (nx, nv)

        Returns:
            Array of shape (nx, nv)
        """
        return (1.0 + self.epsilon * jnp.cos(self.k_0 * mesh.X)) * f_eq


class TwoStreamConfig(VPProblemConfig):
    """Two-stream instability equilibrium.

    f_eq(v) = (alpha * exp(-0.5*(v-mu)^2) + (1-alpha) * exp(-0.5*(v+mu)^2))
              / sqrt(2*pi)

    Two counter-propagating Maxwellian beams prone to electrostatic instability.
    """

    def __init__(self, alpha: float = 0.5, mu: float = 2.4, **overrides):
        defaults = dict(
            name="Two-Stream Instability",
            description="Two counter-propagating Maxwellian beams",
            length_x=10.0 * jnp.pi,
            length_v=6.0,
            t_final=30.0,
            nx=256,
            nv=256,
            dt=0.1,
            epsilon=0.001,
            k_0=2.0 * jnp.pi / (10.0 * jnp.pi),  # = 0.2
            n_fourier_modes=2,
        )
        defaults.update(overrides)
        super().__init__(**defaults)
        self.alpha = alpha
        self.mu = mu

    def make_f_eq(self, mesh: Any) -> jnp.ndarray:
        V = mesh.V
        mu = self.mu
        alpha = self.alpha
        return (
            alpha * jnp.exp(-0.5 * (V - mu) ** 2)
            + (1.0 - alpha) * jnp.exp(-0.5 * (V + mu) ** 2)
        ) / jnp.sqrt(2.0 * jnp.pi)


class BumpOnTailConfig(VPProblemConfig):
    """Bump-on-tail instability equilibrium.

    f_eq(v) = 9/(10*sqrt(2*pi)) * exp(-0.5*(v - v1)^2)
            + sqrt(2)/(10*sqrt(pi)) * exp(-2*(v - v2)^2)

    Background Maxwellian population with an energetic particle bump.
    """

    def __init__(self, v1: float = -3.0, v2: float = 4.5, **overrides):
        defaults = dict(
            name="Bump-on-Tail Instability",
            description="Maxwellian background with energetic particle bump",
            length_x=20.0 * jnp.pi,
            length_v=9.0,
            t_final=40.0,
            nx=256,
            nv=256,
            dt=0.1,
            epsilon=0.001,
            k_0=2.0 * jnp.pi / (20.0 * jnp.pi),  # = 0.1
            n_fourier_modes=2,
        )
        defaults.update(overrides)
        super().__init__(**defaults)
        self.v1 = v1
        self.v2 = v2

    def make_f_eq(self, mesh: Any) -> jnp.ndarray:
        V = mesh.V
        return (
            9.0 / (10.0 * jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * (V - self.v1) ** 2)
            + jnp.sqrt(2.0) / (10.0 * jnp.sqrt(jnp.pi)) * jnp.exp(-2.0 * (V - self.v2) ** 2)
        )

    def make_f_iv(self, mesh: Any, f_eq: jnp.ndarray) -> jnp.ndarray:
        """Bump-on-tail uses a different perturbation form than the default."""
        bump_part = (
            jnp.sqrt(2.0) * self.epsilon / (10.0 * jnp.sqrt(jnp.pi))
            * jnp.exp(-2.0 * (mesh.V - self.v2) ** 2)
            * jnp.cos(0.1 * mesh.X)
        )
        return f_eq + bump_part
