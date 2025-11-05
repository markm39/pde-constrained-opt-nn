"""
Examples module with ground-truth test cases for PDE-constrained optimization.
Implements test cases from the paper: https://arxiv.org/abs/2408.12404
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from pde_opt.solvers import get_solver
from pde_opt.problems import get_problem


@dataclass
class OptimizationExample:
    """Base class for optimization examples."""

    name: str
    problem_name: str
    solver_type: str
    discretization: str
    optimization_type: str  # 'force', 'initial_condition', 'parameter', 'boundary'
    grid_params: Dict[str, int]
    optimizer_config: Dict[str, Any]
    regularization: float = 1e-6


def create_neural_network(hidden_layers: list = [256, 256], activation: str = 'tanh',
                         use_fourier_features: bool = False, fourier_scale: float = 10.0):
    """
    Create a neural network for force/parameter approximation.

    Args:
        hidden_layers: List of hidden layer sizes
        activation: Activation function ('tanh', 'relu', 'sigmoid')
        use_fourier_features: Whether to use Fourier feature encoding for high-frequency learning
        fourier_scale: Scale parameter for random Fourier features (higher = more high-freq)
    """

    class Network(nn.Module):
        layers: list
        activation: str
        use_fourier: bool = False
        fourier_scale: float = 10.0

        @nn.compact
        def __call__(self, x):
            # Apply Fourier features if enabled
            if self.use_fourier:
                # x has shape (batch, input_dim)
                # Create random Fourier feature matrix
                input_dim = x.shape[-1]
                B = self.param('fourier_B',
                              nn.initializers.normal(stddev=self.fourier_scale),
                              (input_dim, 256))
                # Fourier features: [cos(2πBx), sin(2πBx)]
                x_proj = 2 * jnp.pi * x @ B
                x = jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)

            for i, features in enumerate(self.layers):
                x = nn.Dense(features)(x)
                if i < len(self.layers) - 1:
                    if self.activation == 'tanh':
                        x = nn.tanh(x)
                    elif self.activation == 'relu':
                        x = nn.relu(x)
                    elif self.activation == 'sigmoid':
                        x = nn.sigmoid(x)
            # Output layer with ReLU to ensure non-negative forces
            x = nn.Dense(1)(x)
            x = nn.relu(x)  # Apply ReLU to output layer
            return x.squeeze(-1)

    return Network(layers=hidden_layers, activation=activation,
                   use_fourier=use_fourier_features, fourier_scale=fourier_scale)


class Example31_Poisson1D_ScalarForce(OptimizationExample):
    """Example 3.1: 1D Poisson with scalar force estimation."""

    def __init__(self, zero_ic=None, **kwargs):
        super().__init__(
            name="Example 3.1: 1D Poisson Scalar Force",
            problem_name="poisson-1d-scalar",
            solver_type="poisson",
            discretization="fd",
            optimization_type="force",
            grid_params={"nx": 50},
            optimizer_config={"learning_rate": 0.1, "optimizer": "sgd"},
            regularization=0.0
        )

    def run(self, max_iter: int = 100):
        """Run the optimization example."""
        problem = get_problem(self.problem_name)
        solver = get_solver(self.solver_type, self.discretization, nx=self.grid_params['nx'])

        # True force
        f_true = -1.0

        # Initial guess
        f_guess = jnp.array([2.0])

        # Create system matrix
        A = solver.create_system_matrix()

        # True solution
        x_grid = solver.x_grid
        u_true = problem.analytical_solution(x_grid)

        def loss_fn(f_param):
            # Create force vector
            force = f_param[0] * jnp.ones(self.grid_params['nx'])
            # Solve PDE
            u_pred = jnp.linalg.solve(A, force)
            # Compute loss
            return jnp.mean((u_pred - u_true.flatten())**2)

        # Optimization
        optimizer = optax.sgd(self.optimizer_config['learning_rate'])
        opt_state = optimizer.init(f_guess)

        losses = []
        for i in range(max_iter):
            loss, grads = jax.value_and_grad(loss_fn)(f_guess)
            losses.append(float(loss))
            updates, opt_state = optimizer.update(grads, opt_state)
            f_guess = optax.apply_updates(f_guess, updates)

            if i % 20 == 0:
                print(f"Iter {i}: Loss = {loss:.6e}, f_guess = {f_guess[0]:.6f}")

            if loss < 1e-6:
                break

        # Compute final solution
        force_final = f_guess[0] * jnp.ones(self.grid_params['nx'])
        u_final = jnp.linalg.solve(A, force_final)

        return f_guess[0], losses, u_final


class Example32_Poisson1D_VectorForce(OptimizationExample):
    """Example 3.2: 1D Poisson with vector force estimation."""

    def __init__(self, zero_ic=None, **kwargs):
        super().__init__(
            name="Example 3.2: 1D Poisson Vector Force",
            problem_name="poisson-1d-vector",
            solver_type="poisson",
            discretization="fd",
            optimization_type="force",
            grid_params={"nx": 50},
            optimizer_config={"learning_rate": 0.01, "optimizer": "adam"},
            regularization=0.099
        )

    def run(self, max_iter: int = 1000):
        """Run the optimization example."""
        problem = get_problem(self.problem_name)
        solver = get_solver(self.solver_type, self.discretization, nx=self.grid_params['nx'], ny=self.grid_params['nx'])

        # Create system matrix
        A = solver.create_system_matrix()

        # True solution and force
        x_grid = solver.x_grid
        u_true = problem.analytical_solution(x_grid)
        f_true = problem.source_term(x_grid)

        # Initial guess
        f_guess = jnp.zeros_like(f_true)

        def loss_fn(f_param):
            # Solve PDE
            u_pred = jnp.linalg.solve(A, f_param)
            # Data loss + Tikhonov regularization
            data_loss = jnp.mean((u_pred - u_true)**2)
            reg_loss = self.regularization * jnp.mean(f_param**2)
            return data_loss + reg_loss

        # Optimization
        optimizer = optax.adam(self.optimizer_config['learning_rate'])
        opt_state = optimizer.init(f_guess)

        losses = []
        for i in range(max_iter):
            loss, grads = jax.value_and_grad(loss_fn)(f_guess)
            losses.append(float(loss))
            updates, opt_state = optimizer.update(grads, opt_state)
            f_guess = optax.apply_updates(f_guess, updates)

            if i % 200 == 0:
                print(f"Iter {i}: Loss = {loss:.6e}")

        # Compute final solution
        u_final = jnp.linalg.solve(A, f_guess)

        return f_guess, losses, u_final


class Example33_HeatEquation_ForceNN(OptimizationExample):
    """Example 3.3: 1+1D Heat equation with neural network force."""

    def __init__(self, discretization: str = "fd", problem_name: str = "heat-1d", regularization: float = None, **problem_kwargs):
        # Adaptive grid resolution based on oscillations
        n_osc = problem_kwargs.get('n_oscillations', 1)
        nx = max(149, n_osc * 30)  # At least 30 points per wavelength
        nt = 50

        # Adaptive regularization - less for high frequencies (only if not explicitly provided)
        if regularization is None:
            reg = 1e-6 / (n_osc ** 0.5) if n_osc > 1 else 1e-6
        else:
            reg = regularization

        super().__init__(
            name=f"Example 3.3: Heat Equation with NN Force ({discretization.upper()})",
            problem_name=problem_name,
            solver_type="heat",
            discretization=discretization,  # Can be 'fd', 'fem', or 'crank-nicolson'
            optimization_type="force",
            grid_params={"nx": nx, "nt": nt},
            optimizer_config={"learning_rate": 3e-3, "optimizer": "adam"},
            regularization=reg
        )
        self.problem_kwargs = problem_kwargs

    def run(self, max_iter: int = 2000):
        """Run the optimization with neural network using TIME-STEPPING."""
        from jax import lax
        import jax.scipy.linalg as jsp

        problem = get_problem(self.problem_name, **self.problem_kwargs)
        T = self.problem_kwargs.get('T', 1.0)  # Get T from kwargs, default to 1.0
        solver = get_solver(self.solver_type, self.discretization,
                          nx=self.grid_params['nx'], nt=self.grid_params['nt'], T=T)

        # Grid setup
        x_grid = solver.x_grid
        t_grid = solver.t_grid
        k = solver.k  # Time step

        # Create backward Euler matrix for time-stepping
        # A = (1/k)*I - K where K is negative Laplacian (K_h ≈ -Δ)
        K_h = solver.create_spatial_matrix()
        A_be = (1.0/k) * jnp.eye(solver.nx) - K_h
        L_be = jnp.linalg.cholesky(A_be)

        def chol_solve(L, b):
            y = jsp.solve_triangular(L, b, lower=True)
            u = jsp.solve_triangular(L.T, y, lower=False)
            return u

        # Target solution (full trajectory)
        u_target = problem.analytical_solution(x_grid, t_grid)  # (nx, nt)

        # Initial condition (use the problem's IC)
        u0 = problem.initial_condition(x_grid)

        # Initialize neural network with adaptive Fourier features
        n_osc = self.problem_kwargs.get('n_oscillations', 1)
        use_fourier = n_osc >= 4  # Use Fourier features for high-frequency problems
        fourier_scale = float(n_osc) * 2.0 if use_fourier else 1.0

        model = create_neural_network([256, 256], 'tanh',
                                     use_fourier_features=use_fourier,
                                     fourier_scale=fourier_scale)
        key = jax.random.PRNGKey(42)
        dummy = jnp.zeros((1, 2))
        params = model.init(key, dummy)

        if use_fourier:
            print(f"Using Fourier features with scale={fourier_scale:.1f} for k={n_osc} oscillations")

        # Normalize coordinates
        x_norm = 2.0 * x_grid - 1.0
        t_norm = 2.0 * t_grid / solver.T - 1.0

        def forward_with_nn(params):
            """Time-stepping forward pass with NN forcing."""
            def step(u_prev, t_n_norm):
                # Input for NN at this time step
                xt = jnp.stack([x_norm, jnp.full_like(x_norm, t_n_norm)], axis=1)
                f_n = model.apply(params, xt)

                # Backward Euler: A*u_next = u_prev/k + f_n
                rhs = u_prev / k + f_n
                u_next = chol_solve(L_be, rhs)

                return u_next, (u_next, f_n)

            # Scan over time
            _, (U_seq, F_seq) = lax.scan(step, u0, t_norm)
            return U_seq, F_seq  # Both (nt, nx)

        @jax.jit
        def loss_fn(params):
            U_pred, F_pred = forward_with_nn(params)

            # MSE loss (matching working notebook)
            Nu = u_target.size
            data_loss = jnp.sum((U_pred.T - u_target)**2) / Nu  # U_pred is (nt,nx), u_target is (nx,nt)
            reg_loss = self.regularization * jnp.mean(F_pred**2)

            return data_loss + reg_loss, (data_loss, reg_loss)

        # Optimizer (use adam with lr decay like working notebook)
        lr_schedule = optax.exponential_decay(
            init_value=self.optimizer_config['learning_rate'],
            transition_steps=max_iter,
            decay_rate=0.9
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state):
            (loss, (data_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, data_loss, reg_loss

        losses = []
        print(f"Using TIME-STEPPING solver (matches working notebook)")
        for i in range(max_iter):
            params, opt_state, loss, data_loss, reg_loss = train_step(params, opt_state)
            losses.append(float(loss))

            if i % 50 == 0:
                print(f"ep {i:4d} | L={loss:.6f} | mis={data_loss:.6f} | regF={reg_loss:.6f}")

        # Get final predictions
        U_final, F_final = forward_with_nn(params)

        # Convert back to (nx, nt) for compatibility
        u_final = U_final.T  # (nx, nt)
        force_final = F_final.T  # (nx, nt)

        return params, losses, force_final.flatten(), u_final.flatten()


class Example35_LinearHeat2D(OptimizationExample):
    """Example 3.5: 2+1D Linear heat equation with neural network force."""

    def __init__(self, zero_ic=None, regularization: float = 1e-5, prob: str = 'default', **kwargs):
        # Accept zero_ic for compatibility but don't use it
        # Adaptive grid size based on problem type
        if prob == 'cossinsin':
            # Oscillatory problem needs finer grid
            # find_optimal_config() to get < 1% error (needs ~nx=100, ny=100, nt=200)
            nx, ny, nt = 112, 112, 200
        else:
            # Default problem works well with coarser grid (~0.5% solver error)
            nx, ny, nt = 30, 30, 50

        super().__init__(
            name=f"Example 3.5: 2+1D Linear Heat Equation ({prob})",
            problem_name="linear-heat-2d",
            solver_type="heat-2d",
            discretization="crank-nicolson",
            optimization_type="force",
            grid_params={"nx": nx, "ny": ny, "nt": nt},
            optimizer_config={"learning_rate": 1e-3, "optimizer": "adam"},
            regularization=regularization
        )
        self.prob = prob

    def run(self, max_iter: int = 2000, validate: bool = False):
        """Run the optimization with neural network using TIME-STEPPING (Crank-Nicolson extended from 1D to 2D)."""
        from jax import lax
        import jax.scipy.linalg as jsp

        # Validate solver accuracy if requested
        if validate:
            from solver_validation import validate_solver, find_optimal_config
            is_valid, rel_error, details = validate_solver(self, threshold=0.01, verbose=True)
            if not is_valid:
                print(f"⚠ WARNING: Solver error ({rel_error*100:.2f}%) exceeds 1% threshold!")
                print(f"This may prevent accurate recovery of the forcing term.")
                print(f"\nTo find optimal grid size, run:")
                print(f"  from solver_validation import find_optimal_config")
                print(f"  find_optimal_config(ex)")
                print()
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting. Please adjust grid_params and try again.")
                    return None, [], None, None

        problem = get_problem(self.problem_name, prob=self.prob)
        solver = get_solver(self.solver_type, self.discretization,
                          nx=self.grid_params['nx'], ny=self.grid_params['ny'],
                          nt=self.grid_params['nt'])

        # Grid setup
        x_grid = solver.x_grid
        y_grid = solver.y_grid
        t_grid = solver.t_grid
        k = solver.k  # Time step

        # Create Crank-Nicolson matrix for time-stepping
        # Heat equation: du/dt = Δu + f
        # solver.K represents -Δ (negative Laplacian, has negative diagonal)
        # So: du/dt = -K·u + f
        # Crank-Nicolson: (I - k/2 * K) u_{n+1} = (I + k/2 * K) u_n + k * f_n
        A = solver.K  # -Δ (negative Laplacian)
        n_spatial = solver.nx * solver.ny
        A_cn = jnp.eye(n_spatial) - (k/2.0) * A
        L_cn = jnp.linalg.cholesky(A_cn)

        def chol_solve(L, b):
            y = jsp.solve_triangular(L, b, lower=True)
            u = jsp.solve_triangular(L.T, y, lower=False)
            return u

        # Target solution (full trajectory)
        u_target_3d = jnp.stack([problem.analytical_solution(x_grid, y_grid, t)
                                  for t in t_grid], axis=-1)  # (nx, ny, nt)
        u_target = u_target_3d.reshape(n_spatial, solver.nt)  # (nx*ny, nt)

        # Initial condition (use the problem's IC)
        u0_2d = problem.initial_condition(x_grid, y_grid)
        u0 = u0_2d.flatten()  # Shape: (nx*ny,)

        # Initialize neural network (NN input: [x, y, t] → output: f(x,y,t))
        model = create_neural_network([256, 256], 'relu')
        key = jax.random.PRNGKey(42)
        dummy = jnp.zeros((1, 3))  # 3 inputs: x, y, t
        params = model.init(key, dummy)

        # Normalize coordinates (same as 1D)
        x_norm = 2.0 * x_grid / solver.Lx - 1.0
        y_norm = 2.0 * y_grid / solver.Ly - 1.0
        t_norm = 2.0 * t_grid / solver.T - 1.0

        # Create meshgrid for spatial coordinates
        X_norm, Y_norm = jnp.meshgrid(x_norm, y_norm, indexing='ij')
        xy_flat = jnp.stack([X_norm.flatten(), Y_norm.flatten()], axis=1)  # (nx*ny, 2)

        def forward_with_nn(params):
            """Time-stepping forward pass with NN forcing (Crank-Nicolson, same structure as 1D)."""
            def step(u_prev, t_n_norm):
                # Input for NN at this time step (extend 1D pattern to 2D)
                xyt = jnp.stack([xy_flat[:, 0], xy_flat[:, 1], jnp.full(n_spatial, t_n_norm)], axis=1)
                f_n = model.apply(params, xyt)

                # Crank-Nicolson: (I - k/2 * A) u_next = (I + k/2 * A) u_prev + k * f_n
                # where A = solver.K is -Δ (negative Laplacian)
                rhs = (jnp.eye(n_spatial) + (k/2.0) * A) @ u_prev + k * f_n
                u_next = chol_solve(L_cn, rhs)

                return u_next, (u_next, f_n)

            # Scan over time
            _, (U_seq, F_seq) = lax.scan(step, u0, t_norm)
            return U_seq, F_seq  # U_seq: (nt, nx*ny), F_seq: (nt, nx*ny)

        @jax.jit
        def loss_fn(params):
            U_pred, F_pred = forward_with_nn(params)

            # MSE loss
            Nu = u_target.size
            data_loss = jnp.sum((U_pred.T - u_target)**2) / Nu  # U_pred is (nt, nx*ny), u_target is (nx*ny, nt)
            reg_loss = self.regularization * jnp.mean(F_pred**2)

            return data_loss + reg_loss, (data_loss, reg_loss)

        # Optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.optimizer_config['learning_rate'],
            transition_steps=max_iter,
            decay_rate=0.9
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state):
            (loss, (data_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, data_loss, reg_loss

        losses = []
        print(f"Using Crank-Nicolson for linear heat equation ({self.prob})")

        for i in range(max_iter):
            params, opt_state, loss, data_loss, reg_loss = train_step(params, opt_state)
            losses.append(float(loss))

            if i % 50 == 0:
                print(f"ep {i:4d} | L={loss:.6f} | mis={data_loss:.6f} | regF={reg_loss:.6f}")

        # Get final predictions
        U_final, F_final = forward_with_nn(params)

        # Convert back to (nx, ny, nt) for compatibility
        u_final = U_final.T.reshape(solver.nx, solver.ny, solver.nt)  # (nx, ny, nt)
        force_final = F_final.T.reshape(solver.nx, solver.ny, solver.nt)  # (nx, ny, nt)

        return params, losses, force_final.flatten(), u_final.flatten()


class Example36_NonlinearHeat2D(OptimizationExample):
    """Example 3.6: 2+1D Nonlinear heat equation with neural network force."""

    def __init__(self, zero_ic=None, regularization: float = 1e-5, **kwargs):
        # Accept zero_ic for compatibility but don't use it
        super().__init__(
            name="Example 3.6: 2+1D Nonlinear Heat Equation",
            problem_name="nonlinear-heat-2d",
            solver_type="heat-2d",
            discretization="crank-nicolson",
            optimization_type="force",
            grid_params={"nx": 30, "ny": 30, "nt": 50},  # 2D spatial grid + time
            optimizer_config={"learning_rate": 1e-3, "optimizer": "adam"},
            regularization=regularization
        )

    def run(self, max_iter: int = 2000):
        """Run the optimization with neural network using time-stepping for nonlinear PDE."""
        from jax import lax
        import jax.scipy.linalg as jsp

        problem = get_problem(self.problem_name)
        solver = get_solver(self.solver_type, self.discretization,
                          nx=self.grid_params['nx'], ny=self.grid_params['ny'],
                          nt=self.grid_params['nt'])

        # Grid setup
        x_grid = solver.x_grid
        y_grid = solver.y_grid
        t_grid = solver.t_grid
        k = solver.k  # Time step

        # Get the 2D Laplacian matrix
        K = solver.K  # (nx*ny, nx*ny)
        n_spatial = solver.nx * solver.ny

        # Initial condition u0(x,y) on the 2D grid (flattened to 1D vector)
        X, Y = jnp.meshgrid(x_grid, y_grid, indexing='ij')
        u0_2d = problem.initial_condition(x_grid, y_grid)
        u0 = u0_2d.flatten()  # Shape: (nx*ny,)

        # Target solution (full trajectory)
        # u_target[i, j, n] = u(x_i, y_j, t_n)
        u_target_3d = jnp.stack([problem.analytical_solution(x_grid, y_grid, t)
                                  for t in t_grid], axis=-1)  # (nx, ny, nt)
        u_target = u_target_3d.reshape(n_spatial, solver.nt)  # (nx*ny, nt)

        # Initialize neural network (NN input: [x, y, t] → output: f(x,y,t))
        model = create_neural_network([256, 256], 'relu')
        key = jax.random.PRNGKey(42)
        dummy = jnp.zeros((1, 3))  # 3 inputs: x, y, t
        params = model.init(key, dummy)

        # Normalize coordinates
        x_norm = 2.0 * x_grid / solver.Lx - 1.0
        y_norm = 2.0 * y_grid / solver.Ly - 1.0
        t_norm = 2.0 * t_grid / solver.T - 1.0

        # Create meshgrid for spatial coordinates
        X_norm, Y_norm = jnp.meshgrid(x_norm, y_norm, indexing='ij')
        xy_flat = jnp.stack([X_norm.flatten(), Y_norm.flatten()], axis=1)  # (nx*ny, 2)

        def newton_solve(u_guess, rhs, max_iter=10):
            """
            Solve nonlinear system using Newton's method:
            R(u) = (1/k)*u - Δu + u² - rhs = 0
            """
            def residual(u):
                return (1.0/k) * u - K @ u + u**2 - rhs

            def jacobian(u):
                # J = (1/k)*I - K + 2*diag(u)
                return (1.0/k) * jnp.eye(n_spatial) - K + 2.0 * jnp.diag(u)

            def newton_step(i, u):
                J = jacobian(u)
                R = residual(u)
                delta_u = jnp.linalg.solve(J, -R)
                return u + delta_u

            # Run fixed number of Newton iterations (JAX-compatible)
            u = lax.fori_loop(0, max_iter, newton_step, u_guess)
            return u

        def forward_with_nn(params):
            """Time-stepping forward pass with NN forcing and nonlinear solve."""
            def step(u_prev, t_n_norm):
                # Evaluate NN at all spatial points for this time step
                xyt = jnp.concatenate([xy_flat, jnp.full((n_spatial, 1), t_n_norm)], axis=1)
                f_n = model.apply(params, xyt)

                # Crank-Nicolson with nonlinearity:
                # (1/k)*u_{n+1} - Δu_{n+1} + u_{n+1}² = (1/k)*u_n + Δu_n - u_n² + f_{n+1/2}
                # For simplicity, treat nonlinearity implicitly on RHS:
                # RHS = (1/k)*u_n + Δu_n - u_n² + f_n
                rhs = (1.0/k) * u_prev + K @ u_prev - u_prev**2 + f_n

                # Solve nonlinear system with Newton's method
                u_next = newton_solve(u_prev, rhs)

                return u_next, (u_next, f_n)

            # Scan over time
            _, (U_seq, F_seq) = lax.scan(step, u0, t_norm)
            return U_seq, F_seq  # U_seq: (nt, nx*ny), F_seq: (nt, nx*ny)

        @jax.jit
        def loss_fn(params):
            U_pred, F_pred = forward_with_nn(params)

            # MSE loss
            Nu = u_target.size
            data_loss = jnp.sum((U_pred.T - u_target)**2) / Nu  # U_pred is (nt, nx*ny), u_target is (nx*ny, nt)
            reg_loss = self.regularization * jnp.mean(F_pred**2)

            return data_loss + reg_loss, (data_loss, reg_loss)

        # Optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.optimizer_config['learning_rate'],
            transition_steps=max_iter,
            decay_rate=0.9
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state):
            (loss, (data_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, data_loss, reg_loss

        losses = []
        print(f"Using Crank-Nicolson with Newton's method for nonlinear heat equation")
        for i in range(max_iter):
            params, opt_state, loss, data_loss, reg_loss = train_step(params, opt_state)
            losses.append(float(loss))

            if i % 50 == 0:
                print(f"ep {i:4d} | L={loss:.6f} | mis={data_loss:.6f} | regF={reg_loss:.6f}")

        # Get final predictions
        U_final, F_final = forward_with_nn(params)

        # Convert back to (nx, ny, nt) for compatibility
        u_final = U_final.T.reshape(solver.nx, solver.ny, solver.nt)  # (nx, ny, nt)
        force_final = F_final.T.reshape(solver.nx, solver.ny, solver.nt)  # (nx, ny, nt)

        return params, losses, force_final.flatten(), u_final.flatten()


# Factory function to get example by name
def get_example(example_name: str, **kwargs):
    """
    Get example instance by name.

    Args:
        example_name: Name of the example (e.g., 'example-3.1')
        **kwargs: Additional arguments for example constructor

    Returns:
        OptimizationExample instance
    """
    examples = {
        'example-3.1': Example31_Poisson1D_ScalarForce,
        'example-3.2': Example32_Poisson1D_VectorForce,
        'example-3.3': Example33_HeatEquation_ForceNN,
        'example-3.5': Example35_LinearHeat2D,
        'example-3.6': Example36_NonlinearHeat2D,
    }

    if example_name not in examples:
        raise ValueError(f"Unknown example: {example_name}")

    return examples[example_name](**kwargs)


def run_all_examples():
    """Run all available examples."""
    print("Running all examples from the paper...")
    print("=" * 60)

    # Example 3.1
    print("\n" + "="*60)
    print("Example 3.1: 1D Poisson with Scalar Force")
    print("="*60)
    ex1 = get_example('example-3.1')
    f_final, losses = ex1.run(max_iter=100)
    print(f"Final force estimate: {f_final:.6f} (true: -1.0)")

    # Example 3.2
    print("\n" + "="*60)
    print("Example 3.2: 1D Poisson with Vector Force")
    print("="*60)
    ex2 = get_example('example-3.2')
    f_vec, losses = ex2.run(max_iter=500)
    print(f"Final loss: {losses[-1]:.6e}")

    # Example 3.3
    print("\n" + "="*60)
    print("Example 3.3: Heat Equation with Neural Network Force")
    print("="*60)
    ex3 = get_example('example-3.3')
    params, losses, force, solution = ex3.run(max_iter=500)
    print(f"Final loss: {losses[-1]:.6e}")

    print("\n" + "="*60)
    print("All examples completed!")


if __name__ == "__main__":
    # Run a simple test
    print("Testing Example 3.1...")
    ex = get_example('example-3.1')
    f_final, losses = ex.run(max_iter=50)
    print(f"Test completed. Final force: {f_final:.6f}")