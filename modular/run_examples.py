#!/usr/bin/env python3
"""
Simple runner script for PDE optimization examples.
Usage: python run_examples.py [example_number]
Examples: python run_examples.py 3.1
          python run_examples.py all
"""

import sys
import argparse
from examples import get_example


def run_example_31():
    """Run Example 3.1: 1D Poisson with scalar force."""
    print("\n" + "="*60)
    print("Example 3.1: 1D Poisson Scalar Force Estimation")
    print("="*60)
    print("Problem: -u''(x) = f, x ∈ (0,1), u(0) = u(1) = 0")
    print("Goal: Estimate constant force f from target solution")
    print("-"*60)

    ex = get_example('example-3.1')
    f_final, losses = ex.run(max_iter=100)

    print(f"\nResults:")
    print(f"  Final force estimate: {f_final:.6f}")
    print(f"  True force value:     -1.0")
    print(f"  Error:                {abs(f_final - (-1.0)):.6e}")
    print(f"  Final loss:           {losses[-1]:.6e}")
    print(f"  Converged in {len(losses)} iterations")
    return True


def run_example_32():
    """Run Example 3.2: 1D Poisson with vector force."""
    print("\n" + "="*60)
    print("Example 3.2: 1D Poisson Vector Force Estimation")
    print("="*60)
    print("Problem: -u''(x) = f(x), x ∈ (0,1)")
    print("Goal: Estimate spatially-varying force f(x)")
    print("Regularization: Tikhonov (α = 0.099)")
    print("-"*60)

    ex = get_example('example-3.2')
    f_vec, losses = ex.run(max_iter=500)

    print(f"\nResults:")
    print(f"  Final loss:           {losses[-1]:.6e}")
    print(f"  Force vector shape:   {f_vec.shape}")
    print(f"  Force range:          [{f_vec.min():.4f}, {f_vec.max():.4f}]")
    print(f"  Converged in {len(losses)} iterations")
    return True


def run_example_33():
    """Run Example 3.3: Heat equation with neural network force."""
    print("\n" + "="*60)
    print("Example 3.3: 1+1D Heat Equation with Neural Network Force")
    print("="*60)
    print("Problem: ∂u/∂t - ∂²u/∂x² = f(x,t)")
    print("Domain: (x,t) ∈ (0,1) × (0,1)")
    print("Network: 2 hidden layers with 256 neurons each")
    print("Discretization: Finite Element Method")
    print("-"*60)

    ex = get_example('example-3.3', zero_ic=True)
    params, losses, force, solution = ex.run(max_iter=500)

    print(f"\nResults:")
    print(f"  Final loss:           {losses[-1]:.6e}")
    print(f"  Solution shape:       {solution.shape}")
    print(f"  Force shape:          {force.shape}")
    print(f"  Force range:          [{force.min():.4f}, {force.max():.4f}]")
    print(f"  Converged in {len(losses)} iterations")

    # Count NN parameters
    import jax
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  NN parameters:        {n_params}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run PDE optimization examples from the paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py 3.1         # Run Example 3.1
  python run_examples.py 3.2         # Run Example 3.2
  python run_examples.py 3.3         # Run Example 3.3
  python run_examples.py all         # Run all examples
        """
    )
    parser.add_argument('example', nargs='?', default='all',
                      help='Example number to run (3.1, 3.2, 3.3, or all)')

    args = parser.parse_args()

    # Check JAX GPU availability
    try:
        import jax
        print("\n" + "="*60)
        print("JAX Configuration")
        print("="*60)
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")

        if jax.default_backend() == 'gpu':
            print("✓ GPU acceleration enabled!")
        else:
            print("⚠ Running on CPU (GPU not detected)")
        print("="*60)
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: JAX not installed!")
        print("="*60)
        print("\nTo install JAX with CUDA support (for GPU):")
        print("  pip install --upgrade pip")
        print("  pip install --upgrade jax[cuda12]")
        print("\nOr for CPU-only:")
        print("  pip install --upgrade jax")
        print("="*60)
        return 1

    # Map example names to runner functions
    examples = {
        '3.1': run_example_31,
        'example-3.1': run_example_31,
        '3.2': run_example_32,
        'example-3.2': run_example_32,
        '3.3': run_example_33,
        'example-3.3': run_example_33,
    }

    success = True

    if args.example.lower() == 'all':
        print("\nRunning all available examples...")
        for name, runner in [('3.1', run_example_31),
                            ('3.2', run_example_32),
                            ('3.3', run_example_33)]:
            try:
                runner()
            except Exception as e:
                print(f"\n✗ Example {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                success = False

        if success:
            print("\n" + "="*60)
            print("✓ All examples completed successfully!")
            print("="*60)
    else:
        example_key = args.example.lower()
        if example_key not in examples:
            print(f"\nError: Unknown example '{args.example}'")
            print(f"Available examples: {', '.join(sorted(set(examples.keys())))}")
            return 1

        try:
            examples[example_key]()
            print("\n" + "="*60)
            print(f"✓ Example {args.example} completed successfully!")
            print("="*60)
        except Exception as e:
            print(f"\n✗ Example {args.example} FAILED: {e}")
            import traceback
            traceback.print_exc()
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())