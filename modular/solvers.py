"""
Solvers module for PDE-constrained optimization with neural network surrogates.
Implements different discretization methods (FD, FEM) and time-stepping schemes.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from functools import partial


class SpaceTimeSolver:
    """Base class for space-time PDE solvers."""

    def __init__(self, nx: int, nt: int, L: float = 1.0, T: float = 1.0):
        """
        Initialize space-time solver.

        Args:
            nx: Number of spatial grid points
            nt: Number of temporal grid points
            L: Spatial domain length
            T: Temporal domain length
        """
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T = T
        self.h = L / (nx + 1)  # Spatial step
        self.k = T / nt        # Time step

        # Create grids
        self.x_grid = jnp.linspace(self.h, L - self.h, nx)
        self.t_grid = jnp.linspace(self.k, T, nt)

    def solve(self, force_vec: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
        """Solve the linear system A*u = force."""
        return jnp.linalg.solve(A, force_vec)


class HeatEquationFD(SpaceTimeSolver):
    """Finite difference solver for 1+1D heat equation using Kronecker products."""

    def create_spatial_matrix(self) -> jnp.ndarray:
        """Create 1D Laplacian matrix K_h with homogeneous Dirichlet BCs (positive definite form)."""
        diag_main = 2.0 * jnp.ones(self.nx) / (self.h**2)  # POSITIVE (matches working notebook)
        diag_off = -1.0 * jnp.ones(self.nx-1) / (self.h**2)  # NEGATIVE (matches working notebook)
        K = jnp.diag(diag_main) + jnp.diag(diag_off, 1) + jnp.diag(diag_off, -1)
        return K

    def create_temporal_shift_matrix(self) -> jnp.ndarray:
        """Create temporal shift matrix S_k."""
        S = jnp.diag(jnp.ones(self.nt-1), -1)
        return S

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create space-time matrix A_kh using Kronecker products."""
        K_h = self.create_spatial_matrix()
        S_k = self.create_temporal_shift_matrix()
        I_k = jnp.eye(self.nt)
        I_h = jnp.eye(self.nx)

        # A_kh = I_k � (1/k*I_h + K_h) - 1/k*S_k � I_h
        term1 = jnp.kron(I_k, (1.0/self.k)*I_h + K_h)
        term2 = jnp.kron((1.0/self.k)*S_k, I_h)
        A_kh = term1 - term2
        return A_kh


class HeatEquationFEM(SpaceTimeSolver):
    """Finite element solver for 1+1D heat equation."""

    def create_fem_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create FEM mass and stiffness matrices for linear elements."""
        # Mass matrix (linear elements)
        mass_diag = (self.h / 6.0) * jnp.concatenate([
            jnp.array([2.0, 4.0]),
            jnp.ones(max(0, self.nx-3)) * 4.0,
            jnp.array([2.0]) if self.nx > 2 else jnp.array([])
        ])
        mass_off = (self.h / 6.0) * jnp.ones(self.nx-1)
        M = jnp.diag(mass_diag) + jnp.diag(mass_off, 1) + jnp.diag(mass_off, -1)

        # Stiffness matrix (linear elements)
        stiff_diag = (1.0 / self.h) * jnp.concatenate([
            jnp.array([1.0, 2.0]),
            jnp.ones(max(0, self.nx-3)) * 2.0,
            jnp.array([1.0]) if self.nx > 2 else jnp.array([])
        ])
        stiff_off = -(1.0 / self.h) * jnp.ones(self.nx-1)
        K = jnp.diag(stiff_diag) + jnp.diag(stiff_off, 1) + jnp.diag(stiff_off, -1)

        return M, K

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create space-time FEM system matrix using backward Euler."""
        M, K = self.create_fem_matrices()

        # Time derivative matrix (backward differences)
        T_matrix = (jnp.diag(jnp.ones(self.nt)) -
                   jnp.diag(jnp.ones(self.nt-1), -1)) / self.k

        # Space-time system: T � M + I_t � K
        I_t = jnp.eye(self.nt)
        A_fem = jnp.kron(T_matrix, M) + jnp.kron(I_t, K)
        return A_fem


class HeatEquationCrankNicolson(SpaceTimeSolver):
    """Crank-Nicolson solver for heat equation (more stable than explicit methods)."""

    def create_spatial_matrix(self) -> jnp.ndarray:
        """Create 1D Laplacian matrix K_h."""
        diag_main = -2.0 * jnp.ones(self.nx) / (self.h**2)
        diag_off = jnp.ones(self.nx-1) / (self.h**2)
        K = jnp.diag(diag_main) + jnp.diag(diag_off, 1) + jnp.diag(diag_off, -1)
        return K

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create matrices for Crank-Nicolson scheme.
        Returns (A, B) where A*u^{n+1} = B*u^n + force
        """
        K_h = self.create_spatial_matrix()
        I_h = jnp.eye(self.nx)

        # Crank-Nicolson: (I - k/2*K)*u^{n+1} = (I + k/2*K)*u^n + k*f^{n+1/2}
        A = I_h - 0.5 * self.k * K_h  # Implicit part
        B = I_h + 0.5 * self.k * K_h  # Explicit part

        return A, B

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create full space-time system for Crank-Nicolson."""
        A, B = self.create_system_matrices()

        # Build block tridiagonal system
        n_blocks = self.nt
        size = self.nx * n_blocks

        # Initialize system matrix
        system = jnp.zeros((size, size))

        # Fill blocks
        for i in range(n_blocks):
            row_start = i * self.nx
            row_end = (i + 1) * self.nx

            # Diagonal block (A matrix)
            col_start = row_start
            col_end = row_end
            system = system.at[row_start:row_end, col_start:col_end].set(A)

            # Sub-diagonal block (-B matrix)
            if i > 0:
                col_start = (i - 1) * self.nx
                col_end = i * self.nx
                system = system.at[row_start:row_end, col_start:col_end].set(-B)

        return system


class WaveEquationFD(SpaceTimeSolver):
    """Finite difference solver for 1+1D wave equation."""

    def __init__(self, nx: int, nt: int, c: float = 1.0, L: float = 1.0, T: float = 1.0):
        """
        Initialize wave equation solver.

        Args:
            nx: Number of spatial points
            nt: Number of temporal points
            c: Wave speed
            L: Spatial domain length
            T: Temporal domain length
        """
        super().__init__(nx, nt, L, T)
        self.c = c

    def create_spatial_matrix(self) -> jnp.ndarray:
        """Create 1D Laplacian matrix."""
        diag_main = -2.0 * jnp.ones(self.nx) / (self.h**2)
        diag_off = jnp.ones(self.nx-1) / (self.h**2)
        K = jnp.diag(diag_main) + jnp.diag(diag_off, 1) + jnp.diag(diag_off, -1)
        return self.c**2 * K

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create space-time system for wave equation using central differences in time."""
        K_h = self.create_spatial_matrix()
        I_h = jnp.eye(self.nx)

        # Second order central difference in time
        # u^{n+1} - 2*u^n + u^{n-1} = k^2 * c^2 * K_h * u^n

        # Build block tridiagonal system
        n_blocks = self.nt
        size = self.nx * n_blocks
        system = jnp.zeros((size, size))

        for i in range(n_blocks):
            row_start = i * self.nx
            row_end = (i + 1) * self.nx

            # Main diagonal: I_h
            system = system.at[row_start:row_end, row_start:row_end].set(I_h)

            # Super and sub diagonals for time derivative
            if i > 0:
                col_start = (i - 1) * self.nx
                col_end = i * self.nx
                system = system.at[row_start:row_end, col_start:col_end].set(
                    -2.0 * I_h / self.k**2 - K_h
                )

            if i > 1:
                col_start = (i - 2) * self.nx
                col_end = (i - 1) * self.nx
                system = system.at[row_start:row_end, col_start:col_end].set(
                    I_h / self.k**2
                )

        return system


class Poisson1DFD:
    """Finite difference solver for 1D Poisson equation."""

    def __init__(self, nx: int, L: float = 1.0, **kwargs):
        """
        Initialize 1D Poisson solver.

        Args:
            nx: Number of grid points
            L: Domain length
        """
        self.nx = nx
        self.L = L
        self.h = L / (nx + 1)

        # Create grid
        self.x_grid = jnp.linspace(self.h, L - self.h, nx)

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create 1D Laplacian matrix."""
        diag_main = -2.0 * jnp.ones(self.nx) / (self.h**2)
        diag_off = jnp.ones(self.nx-1) / (self.h**2)
        A = jnp.diag(diag_main) + jnp.diag(diag_off, 1) + jnp.diag(diag_off, -1)
        return A

    def solve(self, force_vec: jnp.ndarray, A: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Solve the 1D Poisson equation."""
        if A is None:
            A = self.create_system_matrix()
        return jnp.linalg.solve(A, force_vec)


class PoissonFD:
    """Finite difference solver for 2D Poisson equation."""

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        """
        Initialize 2D Poisson solver.

        Args:
            nx: Number of grid points in x
            ny: Number of grid points in y
            Lx: Domain length in x
            Ly: Domain length in y
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / (nx + 1)
        self.hy = Ly / (ny + 1)

        # Create grids
        self.x_grid = jnp.linspace(self.hx, Lx - self.hx, nx)
        self.y_grid = jnp.linspace(self.hy, Ly - self.hy, ny)

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create 2D Laplacian using Kronecker products."""
        # 1D Laplacian in x
        Kx = (-2.0 * jnp.eye(self.nx) +
              jnp.diag(jnp.ones(self.nx-1), 1) +
              jnp.diag(jnp.ones(self.nx-1), -1)) / self.hx**2

        # 1D Laplacian in y
        Ky = (-2.0 * jnp.eye(self.ny) +
              jnp.diag(jnp.ones(self.ny-1), 1) +
              jnp.diag(jnp.ones(self.ny-1), -1)) / self.hy**2

        # 2D Laplacian: Kx � I_y + I_x � Ky
        I_x = jnp.eye(self.nx)
        I_y = jnp.eye(self.ny)

        A = jnp.kron(Kx, I_y) + jnp.kron(I_x, Ky)
        return A

    def solve(self, force_vec: jnp.ndarray, A: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Solve the 2D Poisson equation."""
        if A is None:
            A = self.create_system_matrix()
        return jnp.linalg.solve(A, force_vec)


class AdvectionDiffusionFD(SpaceTimeSolver):
    """Finite difference solver for advection-diffusion equation."""

    def __init__(self, nx: int, nt: int, v: float = 1.0, D: float = 0.1,
                 L: float = 1.0, T: float = 1.0):
        """
        Initialize advection-diffusion solver.

        Args:
            nx: Number of spatial points
            nt: Number of temporal points
            v: Advection velocity
            D: Diffusion coefficient
            L: Spatial domain length
            T: Temporal domain length
        """
        super().__init__(nx, nt, L, T)
        self.v = v
        self.D = D

    def create_spatial_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create advection and diffusion matrices."""
        # Diffusion (second derivative)
        diff_matrix = self.D * (-2.0 * jnp.eye(self.nx) +
                                jnp.diag(jnp.ones(self.nx-1), 1) +
                                jnp.diag(jnp.ones(self.nx-1), -1)) / self.h**2

        # Advection (first derivative, upwind scheme)
        if self.v > 0:
            # Backward differences
            adv_matrix = self.v * (jnp.eye(self.nx) -
                                  jnp.diag(jnp.ones(self.nx-1), -1)) / self.h
        else:
            # Forward differences
            adv_matrix = self.v * (jnp.diag(jnp.ones(self.nx-1), 1) -
                                  jnp.eye(self.nx)) / self.h

        return adv_matrix, diff_matrix

    @partial(jax.jit, static_argnums=(0,))
    def create_system_matrix(self) -> jnp.ndarray:
        """Create space-time system for advection-diffusion."""
        adv_matrix, diff_matrix = self.create_spatial_matrices()
        spatial_matrix = -adv_matrix + diff_matrix

        # Time discretization (backward Euler for stability)
        I_h = jnp.eye(self.nx)
        I_t = jnp.eye(self.nt)

        # Time derivative matrix
        T_matrix = (jnp.diag(jnp.ones(self.nt)) -
                   jnp.diag(jnp.ones(self.nt-1), -1)) / self.k

        # Full system
        A = jnp.kron(T_matrix, I_h) - jnp.kron(I_t, spatial_matrix)
        return A


# Factory function to get solver by name
def get_solver(problem_type: str, discretization: str, **kwargs):
    """
    Get solver instance by problem type and discretization method.

    Args:
        problem_type: Type of PDE ('heat', 'wave', 'poisson', 'advection-diffusion')
        discretization: Discretization method ('fd', 'fem', 'crank-nicolson')
        **kwargs: Additional arguments for solver constructor

    Returns:
        Solver instance
    """
    solvers = {
        ('heat', 'fd'): HeatEquationFD,
        ('heat', 'fem'): HeatEquationFEM,
        ('heat', 'crank-nicolson'): HeatEquationCrankNicolson,
        ('wave', 'fd'): WaveEquationFD,
        ('poisson', 'fd'): Poisson1DFD,
        ('poisson', '2d-fd'): PoissonFD,
        ('advection-diffusion', 'fd'): AdvectionDiffusionFD,
    }

    key = (problem_type.lower(), discretization.lower())
    if key not in solvers:
        raise ValueError(f"Unknown solver combination: {problem_type} with {discretization}")

    return solvers[key](**kwargs)