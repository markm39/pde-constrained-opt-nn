"""Vlasov-Poisson optimization example for plasma instability suppression.

Optimizes an external electric field H(x) to suppress plasma instabilities
in the 1D-1D Vlasov-Poisson system. H(x) is parameterized as a truncated
Fourier series and optimized via gradient-based methods using JAX autodiff
through the semi-Lagrangian forward solver.

Reference: arXiv:2504.10435
Uses: https://github.com/maguerrap/vlasov-poisson
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import optax

from pde_opt.examples.examples import OptimizationExample
from pde_opt.problems.vlasov_poisson import VPProblemConfig


@dataclass
class VPResult:
    """Structured results from VP optimization.

    The VP equivalent of the (params, losses, force, solution) tuple
    returned by heat equation examples, with VP-specific fields.
    """
    ak_coefficients: jnp.ndarray    # (2, n_modes) optimized Fourier coefficients
    H_field: jnp.ndarray            # (nx,) optimized external field H(x)
    losses: list                     # Objective values per iteration
    f_final: jnp.ndarray            # (nx, nv) final distribution at t=T
    f_eq: jnp.ndarray               # (nx, nv) equilibrium distribution
    E_array: jnp.ndarray            # (num_steps, nx) total electric field history
    ee_array: jnp.ndarray           # (num_steps,) electric energy over time
    ee_baseline: jnp.ndarray        # (num_steps,) uncontrolled (H=0) electric energy
    mesh: Any                        # vp_solver.Mesh object
    config: VPProblemConfig          # Problem configuration
    cost_type: str                   # Which cost function was used
    f_total: Optional[jnp.ndarray] = None  # (num_steps, nx, nv) full trajectory if requested


class ExampleVP_FourierControl(OptimizationExample):
    """Vlasov-Poisson instability suppression with Fourier-parameterized H(x).

    Optimizes Fourier coefficients (a_k, b_k) of the external electric field
        H(x) = sum_k [a_k cos(k*k_0*x) + b_k sin(k*k_0*x)]
    to suppress plasma instabilities measured by one of:
        - 'kl': KL divergence between f(T) and f_eq
        - 'ee': Electric energy at final time T
        - 'eet': Time-integrated electric energy over [0, T]
    """

    def __init__(self, problem_name: str = 'vp-two-stream',
                 cost_function: str = 'ee',
                 n_fourier_modes: int = None,
                 regularization: float = 0.0,
                 **problem_overrides):
        from pde_opt.problems.vlasov_poisson import TwoStreamConfig, BumpOnTailConfig

        vp_configs = {
            'vp-two-stream': TwoStreamConfig,
            'vp-bump-on-tail': BumpOnTailConfig,
        }
        if problem_name not in vp_configs:
            raise ValueError(f"Unknown VP problem: {problem_name}. "
                             f"Available: {list(vp_configs.keys())}")

        config = vp_configs[problem_name](**problem_overrides)
        n_modes = n_fourier_modes if n_fourier_modes is not None else config.n_fourier_modes

        super().__init__(
            name=f"VP: {config.name} ({cost_function.upper()}, {n_modes} modes)",
            problem_name=problem_name,
            solver_type="vlasov-poisson",
            discretization="semi-lagrangian",
            optimization_type="external_field",
            grid_params={"nx": config.nx, "nv": config.nv},
            optimizer_config={
                "optimizer": "sgd_linesearch",
                "learning_rate": 1.0,
                "max_linesearch_steps": 15,
            },
            regularization=regularization,
        )
        self.config = config
        self.cost_type = cost_function
        self.n_fourier_modes = n_modes
        self.reg_weight = regularization
        self.problem_kwargs = problem_overrides

    def run(self, max_iter: int = 50, seed: int = 888,
            optimizer: str = 'linesearch', learning_rate: float = None,
            store_trajectory: bool = False) -> VPResult:
        """Run the VP optimization.

        Args:
            max_iter: Number of optimization iterations
            seed: Random seed for initial Fourier coefficients
            optimizer: 'linesearch' (default, paper's method) or 'adam'
            learning_rate: Override LR (default: 1e-4 for adam)
            store_trajectory: If True, store full f(t,x,v) trajectory (memory-heavy)

        Returns:
            VPResult with all optimization outputs
        """
        # VP solver requires float64
        jax.config.update("jax_enable_x64", True)

        from vp_solver.jax_vp_solver import Mesh, make_mesh, VlasovPoissonSolver
        from vp_solver.utils import (
            external_electric_field,
            make_cost_function_kl,
            make_cost_function_ee,
            make_cost_function_eet,
        )

        config = self.config

        # 1. Build mesh and solver
        mesh = make_mesh(float(config.length_x), float(config.length_v),
                         config.nx, config.nv)
        f_eq = config.make_f_eq(mesh)
        f_iv = config.make_f_iv(mesh, f_eq)
        solver = VlasovPoissonSolver(mesh=mesh, dt=config.dt, f_eq=f_eq)
        solver_jit = jax.jit(solver.run_forward_jax_scan,
                             static_argnames=('t_final',))

        t_final = float(config.t_final)
        k_0 = float(config.k_0)

        print(f"VP problem: {config.name}")
        print(f"  Domain: x in [0, {float(config.length_x):.2f}], "
              f"v in [-{float(config.length_v)}, {float(config.length_v)}]")
        print(f"  Grid: {config.nx} x {config.nv}, dt={config.dt}, T={t_final}")
        print(f"  Cost function: {self.cost_type.upper()}")
        print(f"  Fourier modes: {self.n_fourier_modes}")
        if self.reg_weight > 0:
            print(f"  Regularization: {self.reg_weight:.2e}")

        # 2. Build cost function
        if self.cost_type == 'kl':
            base_cost_fn = make_cost_function_kl(solver, solver_jit, f_iv, k_0, t_final)
        elif self.cost_type == 'ee':
            base_cost_fn = make_cost_function_ee(solver, solver_jit, f_iv, k_0, t_final)
        elif self.cost_type == 'eet':
            base_cost_fn = make_cost_function_eet(solver, solver_jit, f_iv, k_0, t_final)
        else:
            raise ValueError(f"Unknown cost function: {self.cost_type}. "
                             f"Available: kl, ee, eet")

        # Wrap with regularization: J(ak) = base_cost(ak) + lambda * ||ak||^2
        if self.reg_weight > 0:
            reg_w = self.reg_weight
            def cost_fn(ak):
                return base_cost_fn(ak) + reg_w * jnp.sum(ak ** 2)
        else:
            cost_fn = base_cost_fn

        # 3. Initialize Fourier coefficients
        key = jax.random.PRNGKey(seed)
        ak = jax.random.uniform(key, (2, self.n_fourier_modes),
                                minval=-0.003, maxval=0.001)

        # 4. Run baseline (H=0) before optimization
        print("\nRunning baseline (H=0)...")
        H_zero = jnp.zeros(mesh.nx)
        _, _, _, ee_baseline = solver_jit(f_iv, H_zero, t_final)
        print(f"  Baseline final electric energy: {float(ee_baseline[-1]):.6e}")

        # 5. Optimize
        print(f"\nOptimizing with {optimizer} for {max_iter} iterations...")
        initial_cost = float(cost_fn(ak))
        print(f"  Initial cost: {initial_cost:.6e}")

        if optimizer == 'linesearch':
            if not hasattr(optax, 'scale_by_zoom_linesearch'):
                print("  Warning: optax.scale_by_zoom_linesearch not available "
                      "(requires optax>=0.2.0). Falling back to Adam.")
                optimizer = 'adam'

        if optimizer == 'linesearch':
            ak, losses = self._optimize_linesearch(
                ak, cost_fn, max_iter)
        elif optimizer == 'adam':
            lr = learning_rate if learning_rate is not None else 1e-4
            ak, losses = self._optimize_adam(
                ak, cost_fn, max_iter, lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. "
                             f"Available: adam, linesearch")

        print(f"  Final cost: {float(losses[-1]):.6e}")

        # 6. Final forward pass with optimized H
        H_opt = external_electric_field(ak, mesh, k_0)
        f_array, f_total_data, E_array, ee_array = solver_jit(f_iv, H_opt, t_final)

        print(f"\n  Final electric energy: {float(ee_array[-1]):.6e}")
        print(f"  Baseline electric energy: {float(ee_baseline[-1]):.6e}")
        suppression = float(ee_array[-1]) / float(ee_baseline[-1])
        print(f"  Suppression ratio: {suppression:.4f}")

        return VPResult(
            ak_coefficients=ak,
            H_field=H_opt,
            losses=[float(x) for x in losses],
            f_final=f_array,
            f_eq=f_eq,
            E_array=E_array,
            ee_array=ee_array,
            ee_baseline=ee_baseline,
            mesh=mesh,
            config=config,
            cost_type=self.cost_type,
            f_total=f_total_data if store_trajectory else None,
        )

    def _optimize_linesearch(self, ak: jnp.ndarray,
                             cost_fn, max_iter: int):
        """Optimize using SGD with zoom line search (paper's method)."""
        opt = optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_zoom_linesearch(max_linesearch_steps=15),
        )
        opt_state = opt.init(ak)

        value_and_grad_fn = jax.jit(
            optax.value_and_grad_from_state(cost_fn))

        def scan_fn(carry, _):
            params, opt_state = carry
            value, grad = value_and_grad_fn(params, state=opt_state)
            updates, opt_state = opt.update(
                grad, opt_state, params,
                value=value, grad=grad, value_fn=cost_fn)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), value

        (ak_opt, _), losses = jax.lax.scan(
            scan_fn, (ak, opt_state), None, length=max_iter)

        # Print progress for a few checkpoints
        losses_list = [float(x) for x in losses]
        for i in [0, max_iter // 4, max_iter // 2, 3 * max_iter // 4, max_iter - 1]:
            if i < len(losses_list):
                print(f"  iter {i:4d} | cost={losses_list[i]:.6e}")

        return ak_opt, losses_list

    def _optimize_adam(self, ak: jnp.ndarray,
                       cost_fn, max_iter: int, lr: float):
        """Optimize using Adam (alternative to line search)."""
        opt = optax.adam(lr)
        opt_state = opt.init(ak)

        @jax.jit
        def step(ak, opt_state):
            value, grad = jax.value_and_grad(cost_fn)(ak)
            updates, opt_state = opt.update(grad, opt_state)
            ak = optax.apply_updates(ak, updates)
            return ak, opt_state, value

        losses = []
        for i in range(max_iter):
            ak, opt_state, value = step(ak, opt_state)
            losses.append(float(value))
            if i % max(1, max_iter // 20) == 0:
                print(f"  iter {i:4d} | cost={losses[-1]:.6e}")

        return ak, losses
