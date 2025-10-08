# PDE-Constrained Optimization with Neural Network Surrogates

Implementation of PDE-constrained optimization using neural networks for force/parameter estimation, based on [arxiv.org/abs/2408.12404](https://arxiv.org/abs/2408.12404).

## Structure

### Original Implementation
- **`1d_heat_force_vector.ipynb`**: Original working implementation using time-stepping for the 1+1D heat equation with neural network force estimation.

### Modular Implementation
Refactored into reusable components for scalability and testing:

- **`modular/solvers.py`**: PDE discretization (finite difference, finite element, Crank-Nicolson)
- **`modular/problems.py`**: Problem definitions and analytical solutions
- **`modular/examples.py`**: Example implementations (3.1, 3.2, 3.3 from paper)
- **`modular/run_examples.py`**: Command-line interface to run examples
- **`modular/run_modular_examples.ipynb`**: Jupyter notebook interface

## Quick Start

### Setup (WSL/Linux with NVIDIA GPU)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install JAX with CUDA support
pip install --upgrade pip
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax matplotlib jupyter
```

### Run Examples

```bash
source venv/bin/activate

# Run individual examples
python modular/run_examples.py 3.1  # Poisson scalar force (~5s)
python modular/run_examples.py 3.2  # Poisson vector force (~30s)
python modular/run_examples.py 3.3  # Heat equation with NN (~5min on GPU)

# Or run all examples
python modular/run_examples.py all
```

### Using Jupyter

```bash
source venv/bin/activate
jupyter notebook modular/run_modular_examples.ipynb
```

## Examples

- **Example 3.1**: 1D Poisson equation with scalar force estimation
- **Example 3.2**: 1D Poisson equation with vector force and Tikhonov regularization
- **Example 3.3**: 1+1D heat equation with neural network force (time-stepping solver)

## Key Features

- Time-stepping solver using `jax.lax.scan` for stability
- GPU acceleration with JAX
- Modular architecture for easy extension to new PDEs
- Matches paper results: final loss ~0.0003 for Example 3.3

## Reference

```bibtex
@article{pde-nn-2024,
  title={PDE-Constrained Optimization with Neural Network Surrogates},
  url={https://arxiv.org/abs/2408.12404},
  year={2024}
}
```