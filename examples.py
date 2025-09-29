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
            discretization="fem",
            optimization_type="force",
            grid_params={"nx": 32, "nt": 32},
            optimizer_config={"learning_rate": 1e-3, "optimizer": "adamw"},
            regularization=1e-6
        )
        self.zero_ic = zero_ic

    def run(self, max_iter: int = 2000):
        """Run the optimization with neural network."""
        problem = get_problem(self.problem_name, zero_ic=self.zero_ic)
        solver = get_solver(self.solver_type, self.discretization,
                          nx=self.grid_params['nx'], nt=self.grid_params['nt'])

        # Create system matrix
        A_fem = solver.create_system_matrix()

        # Target solution
        x_grid = solver.x_grid
        t_grid = solver.t_grid
        u_target = problem.analytical_solution(x_grid, t_grid)
        u_target_vec = u_target.flatten()

        # Create input coordinates for neural network
        coords = []
        for i in range(len(x_grid)):
            for j in range(len(t_grid)):
                coords.append([x_grid[i], t_grid[j]])
        input_coords = jnp.array(coords)

        # Initialize neural network
        model = create_neural_network([256, 256], 'tanh')
        key = jax.random.PRNGKey(42)
        params = model.init(key, input_coords)

        @jax.jit
        def loss_fn(params):
            # Get force from neural network
            force_pred = model.apply(params, input_coords)
            # Solve PDE
            u_pred = jnp.linalg.solve(A_fem, force_pred)
            # Losses
            data_loss = jnp.mean((u_pred - u_target_vec)**2)
            reg_loss = self.regularization * jnp.mean(force_pred**2)
            return data_loss + reg_loss, (data_loss, reg_loss)

        # Optimization
        optimizer = optax.adamw(self.optimizer_config['learning_rate'])
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state):
            (loss, (data_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, data_loss, reg_loss

        losses = []
        for i in range(max_iter):
            params, opt_state, loss, data_loss, reg_loss = train_step(params, opt_state)
            losses.append(float(loss))

            if i % 200 == 0:
                print(f"Iter {i}: Loss = {loss:.6f}, Data = {data_loss:.6f}, Reg = {reg_loss:.6f}")

        # Get final force
        force_final = model.apply(params, input_coords)
        u_final = jnp.linalg.solve(A_fem, force_final)

        return params, losses, force_final, u_final


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