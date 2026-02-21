# PDE-Constrained Optimization with Neural Network Surrogates

Implementation of PDE-constrained optimization using neural networks for force/parameter estimation, based on [arxiv.org/abs/2408.12404](https://arxiv.org/abs/2408.12404).

## Structure

### Original Implementation
- **`legacy/1d_heat_force_vector.ipynb`**: Original working implementation using time-stepping for the 1+1D heat equation with neural network force estimation.

### Modular Implementation
Refactored into reusable components for scalability and testing:

- **`solvers.py`**: PDE discretization (finite difference, finite element, Crank-Nicolson)
- **`problems.py`**: Problem definitions and analytical solutions
- **`examples.py`**: Example implementations (3.1, 3.2, 3.3 from paper)
- **`run_examples.py`**: Command-line interface to run examples
- **`run_modular_examples.ipynb`**: Jupyter notebook interface
- **`plotting.py`** Boilerplate code for plotting in the notebook based on problem/example

## Examples

- **Example 3.1**: 1D Poisson equation with scalar force estimation
- **Example 3.2**: 1D Poisson equation with vector force and Tikhonov regularization
- **Example 3.3**: 1+1D heat equation with neural network force (time-stepping solver)
- **Example 3.6**: 2+1D nonlinear heat equation with neural network force

## Reference

```bibtex
@article{pde-nn-2024,
  title={Optimal control of partial differential equations in PyTorch using
automatic differentiation and neural network surrogates},
  url={https://arxiv.org/abs/2408.12404},
  year={2024}
}
```

## Architecture Study Scripts

- Baseline architecture study:
  - `python scripts/nn_architecture_study.py --dry-run`
  - `python scripts/nn_architecture_study.py --max-iter 3000`
- Fourier-space architecture study:
  - `python scripts/nn_fourier_study.py --dry-run`
  - `python scripts/nn_fourier_study.py --input-schemes state_time,state_only,time_only --mode-budgets 8,16,32,full`
