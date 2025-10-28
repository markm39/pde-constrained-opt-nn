"""
Standalone script to validate solver accuracy for all examples.

This script checks that the forward solver has acceptable error (< 1%)
before attempting neural network training. This is a quick sanity check
to catch configuration issues.

Usage:
    python validate_examples.py                    # Validate all examples
    python validate_examples.py --example 3.5      # Validate specific example
    python validate_examples.py --find-config 3.5  # Find optimal config for example
"""

import argparse
from pde_opt.examples import get_example
from pde_opt.utils.solver_validation import validate_solver, find_optimal_config


def validate_all_examples():
    """Validate all available examples with their default configurations."""

    examples_to_test = [
        ('example-3.3', {'problem_name': 'heat-1d'}),
        ('example-3.3', {'problem_name': 'heat-1d-oscillating', 'n_oscillations': 10}),
        ('example-3.5', {'prob': 'default'}),
        ('example-3.5', {'prob': 'cossinsin'}),
        ('example-3.6', {}),
    ]

    print("\n" + "="*80)
    print("VALIDATING ALL EXAMPLES")
    print("="*80)

    results = []
    for ex_name, kwargs in examples_to_test:
        print(f"\nTesting {ex_name} with {kwargs}")
        print("-" * 80)

        try:
            ex = get_example(ex_name, **kwargs)
            is_valid, rel_error, details = validate_solver(ex, threshold=0.01, verbose=True)
            results.append((ex.name, is_valid, rel_error))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((ex_name, False, None))

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Example':<50} {'Status':<10} {'Error (%)'}")
    print("-" * 80)

    for name, is_valid, error in results:
        status = "✓ PASS" if is_valid else "✗ FAIL"
        error_str = f"{error*100:.2f}" if error is not None else "ERROR"
        print(f"{name:<50} {status:<10} {error_str}")

    print("="*80)

    pass_count = sum(1 for _, is_valid, _ in results if is_valid)
    total = len(results)
    print(f"\nResults: {pass_count}/{total} passed")

    if pass_count < total:
        print("\n⚠ Some examples failed validation!")
        print("Use --find-config <example> to find optimal grid sizes")


def validate_specific_example(example_name: str):
    """Validate a specific example."""

    # Map example names to kwargs
    example_configs = {
        '3.3': {'problem_name': 'heat-1d'},
        '3.3-osc': {'problem_name': 'heat-1d-oscillating', 'n_oscillations': 10},
        '3.5': {'prob': 'default'},
        '3.5-cos': {'prob': 'cossinsin'},
        '3.6': {},
    }

    if example_name not in example_configs:
        print(f"Unknown example: {example_name}")
        print(f"Available: {list(example_configs.keys())}")
        return

    kwargs = example_configs[example_name]
    ex = get_example(f"example-{example_name.split('-')[0]}", **kwargs)

    is_valid, rel_error, details = validate_solver(ex, threshold=0.01, verbose=True)

    if not is_valid:
        print(f"\n💡 Tip: Run with --find-config {example_name} to find optimal grid size")


def find_config_for_example(example_name: str):
    """Find optimal configuration for a specific example."""

    example_configs = {
        '3.3': {'problem_name': 'heat-1d'},
        '3.3-osc': {'problem_name': 'heat-1d-oscillating', 'n_oscillations': 10},
        '3.5': {'prob': 'default'},
        '3.5-cos': {'prob': 'cossinsin'},
        '3.6': {},
    }

    if example_name not in example_configs:
        print(f"Unknown example: {example_name}")
        print(f"Available: {list(example_configs.keys())}")
        return

    kwargs = example_configs[example_name]
    ex = get_example(f"example-{example_name.split('-')[0]}", **kwargs)

    optimal_config = find_optimal_config(ex, threshold=0.01, max_nx=200)

    if optimal_config:
        print("\n💡 To use this configuration, update your notebook:")
        print(f"\nex = get_example('example-{example_name.split('-')[0]}', ", end="")
        for key, val in kwargs.items():
            if isinstance(val, str):
                print(f"{key}='{val}', ", end="")
            else:
                print(f"{key}={val}, ", end="")
        print(")")
        print(f"ex.grid_params = {optimal_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate solver accuracy for examples")
    parser.add_argument('--example', type=str, help='Specific example to validate (e.g., 3.5, 3.5-cos)')
    parser.add_argument('--find-config', type=str, help='Find optimal config for example')

    args = parser.parse_args()

    if args.find_config:
        find_config_for_example(args.find_config)
    elif args.example:
        validate_specific_example(args.example)
    else:
        validate_all_examples()
