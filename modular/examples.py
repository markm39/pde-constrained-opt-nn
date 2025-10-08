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

from solvers import get_solver
from problems import get_problem


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


def create_neural_network(hidden_layers: list = [256, 256], activation: str = 'tanh'):
    """Create a neural network for force/parameter approximation."""

    class Network(nn.Module):
        layers: list
        activation: str

        @nn.compact
        def __call__(self, x):
            for i, features in enumerate(self.layers):
                x = nn.Dense(features)(x)
                if i < len(self.layers) - 1:
                    if self.activation == 'tanh':
                        x = nn.tanh(x)
                    elif self.activation == 'relu':
                        x = nn.relu(x)
                    elif self.activation == 'sigmoid':
                        x = nn.sigmoid(x)
            # Output layer
            x = nn.Dense(1)(x)
            return x.squeeze(-1)

    return Network(layers=hidden_layers, activation=activation)


class Example31_Poisson1D_ScalarForce(OptimizationExample):
    """Example 3.1: 1D Poisson with scalar force estimation."""

    def __init__(self):
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

        return f_guess[0], losses


class Example32_Poisson1D_VectorForce(OptimizationExample):
    """Example 3.2: 1D Poisson with vector force estimation."""

    def __init__(self):
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

        return f_guess, losses


class Example33_HeatEquation_ForceNN(OptimizationExample):
    """Example 3.3: 1+1D Heat equation with neural network force."""

    def __init__(self, zero_ic: bool = True):
        super().__init__(
            name="Example 3.3: Heat Equation with NN Force",
            problem_name="heat-1d",
            solver_type="heat",
            discretization="fd",  # Use FD to get create_spatial_matrix()
            optimization_type="force",
            grid_params={"nx": 149, "nt": 50},  # Match working notebook: nh=150 → nx=149
            optimizer_config={"learning_rate": 3e-3, "optimizer": "adam"},  # Match working notebook
            regularization=1e-5  # Match working notebook
        )
        self.zero_ic = zero_ic

    def run(self, max_iter: int = 2000):
        """Run the optimization with neural network using TIME-STEPPING (matches working notebook!)."""
        from jax import lax
        import jax.scipy.linalg as jsp

        problem = get_problem(self.problem_name, zero_ic=self.zero_ic)
        solver = get_solver(self.solver_type, self.discretization,
                          nx=self.grid_params['nx'], nt=self.grid_params['nt'])

        # Grid setup
        x_grid = solver.x_grid
        t_grid = solver.t_grid
        k = solver.k  # Time step

        # Create backward Euler matrix for time-stepping
        # A = (1/k)*I + K where K is spatial Laplacian
        K_h = solver.create_spatial_matrix()
        A_be = (1.0/k) * jnp.eye(solver.nx) + K_h
        L_be = jnp.linalg.cholesky(A_be)

        def chol_solve(L, b):
            y = jsp.solve_triangular(L, b, lower=True)
            u = jsp.solve_triangular(L.T, y, lower=False)
            return u

        # Target solution (full trajectory)
        u_target = problem.analytical_solution(x_grid, t_grid)  # (nx, nt)

        # Initial condition
        u0 = jnp.zeros(solver.nx) if self.zero_ic else jnp.sin(jnp.pi * x_grid)

        # Initialize neural network
        model = create_neural_network([256, 256], 'tanh')
        key = jax.random.PRNGKey(42)
        dummy = jnp.zeros((1, 2))
        params = model.init(key, dummy)

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


class Example35_ThermalFin_ParameterEstimation(OptimizationExample):
    """Example 3.5: 2D Thermal fin parameter estimation."""

    def __init__(self):
        super().__init__(
            name="Example 3.5: Thermal Fin Parameter Estimation",
            problem_name="thermal-fin-2d",
            solver_type="poisson",
            discretization="fem",
            optimization_type="parameter",
            grid_params={"nx": 60, "ny": 41},
            optimizer_config={"learning_rate": 0.01, "optimizer": "rprop"},
            regularization=0.1
        )

    def run(self, max_iter: int = 100):
        """Run parameter estimation for thermal fin."""
        # This is a simplified version - full implementation would require
        # more complex subdomain handling
        print("Thermal fin example - simplified implementation")

        # True parameters from paper
        mu_true = jnp.array([0.1, 8.37317, 6.57228, 0.466517, 1.88354, 0.01])
        mu_ref = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1])
        mu_guess = 0.5 * jnp.ones(6)

        def loss_fn(mu):
            # Simplified loss - actual would solve thermal fin PDE
            diff = mu - mu_true
            data_loss = jnp.sum(diff**2)
            reg_loss = 0.1 * jnp.sum(((mu - mu_ref) / mu_ref)**2)
            return data_loss + reg_loss

        # Optimization with constraints
        optimizer = optax.adam(self.optimizer_config['learning_rate'])
        opt_state = optimizer.init(mu_guess)

        losses = []
        for i in range(max_iter):
            loss, grads = jax.value_and_grad(loss_fn)(mu_guess)
            losses.append(float(loss))
            updates, opt_state = optimizer.update(grads, opt_state)
            mu_guess = optax.apply_updates(mu_guess, updates)

            # Apply constraints
            mu_guess = jnp.clip(mu_guess.at[:5].set(jnp.clip(mu_guess[:5], 0.1, 10.0)))
            mu_guess = mu_guess.at[5].set(jnp.clip(mu_guess[5], 0.01, 1.0))

            if i % 20 == 0:
                print(f"Iter {i}: Loss = {loss:.6e}")

        return mu_guess, losses


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
        'example-3.5': Example35_ThermalFin_ParameterEstimation,
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
    ex3 = get_example('example-3.3', zero_ic=True)
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