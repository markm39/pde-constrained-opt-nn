#!/usr/bin/env python3
"""
Standalone script to benchmark solver accuracy across heat equation problems.

Usage:
    python scripts/benchmark_solvers.py [options]

Options:
    --output-dir DIR      Output directory for results (default: results/benchmarks/)
    --target-error ERROR  Target relative L2 error (default: 0.01 = 1%)
    --max-nx SIZE         Maximum grid size to try (default: 500)
    --formats FORMATS     Comma-separated formats: md,csv,latex,json (default: all)
    --help                Show this help message
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pde_opt.utils.benchmarking import benchmark_heat_equations
from pde_opt.utils.benchmark_export import export_all_formats
from pde_opt.utils.convergence_plots import plot_all_convergence


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark solver accuracy for heat equation problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmarks',
        help='Output directory for results (default: results/benchmarks/)'
    )

    parser.add_argument(
        '--target-error',
        type=float,
        default=0.01,
        help='Target relative L2 error as fraction (default: 0.01 = 1%%)'
    )

    parser.add_argument(
        '--max-nx',
        type=int,
        default=500,
        help='Maximum grid size to try (default: 500)'
    )

    parser.add_argument(
        '--formats',
        type=str,
        default='md,csv,latex,json',
        help='Comma-separated output formats: md,csv,latex,json (default: all)'
    )

    parser.add_argument(
        '--convergence',
        action='store_true',
        help='Run convergence study with predefined grids (both 1D and 2D problems)'
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    if args.convergence:
        print("CONVERGENCE STUDY - HEAT EQUATIONS")
    else:
        print("SOLVER BENCHMARKING - HEAT EQUATIONS")
    print("=" * 70)
    print(f"Configuration:")
    if args.convergence:
        print(f"  Mode:          Convergence Study")
        print(f"  1D Problems:")
        print(f"    Spatial grids:  nx=[50,100,150,200,250], nt=200")
        print(f"    Temporal grids: nx=150, nt=[50,100,150,200,250]")
        print(f"  2D Problems:")
        print(f"    Spatial grids:  nx=ny=[20,40,60,80,100,120], nt=50")
        print(f"    Temporal grids: nx=ny=60, nt=[25,50,75,100,125,150]")
    else:
        print(f"  Target Error:  {args.target_error * 100:.1f}%")
        print(f"  Max Grid Size: {args.max_nx}")
    print(f"  Output Dir:    {args.output_dir}")
    print(f"  Formats:       {args.formats}")
    print("=" * 70)

    # Run benchmarks
    print("\nRunning benchmarks...\n")
    try:
        results = benchmark_heat_equations(
            target_error=args.target_error,
            max_nx=args.max_nx,
            convergence_study=args.convergence
        )
    except Exception as e:
        print(f"\n✗ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export results
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    try:
        files = export_all_formats(results, args.output_dir)

        print("\n✓ Data tables exported!")
        print(f"\nTables saved to {args.output_dir}/:")
        for format_name, filepath in files.items():
            print(f"  - {format_name}: {Path(filepath).name}")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate convergence plots if in convergence study mode
    if args.convergence:
        print("\n" + "=" * 70)
        print("GENERATING CONVERGENCE PLOTS")
        print("=" * 70)

        try:
            plot_dir = str(Path(args.output_dir) / 'convergence_plots')
            plot_files = plot_all_convergence(results, plot_dir)

            print(f"\n✓ Convergence plots generated!")
            print(f"\nPlots saved to {plot_dir}/:")
            for problem, files_list in plot_files.items():
                for filepath in files_list:
                    print(f"  - {Path(filepath).name}")

        except Exception as e:
            print(f"\n✗ Plotting failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if args.convergence:
        # Count convergence study configurations
        total_configs = 0
        total_passing = 0
        for problem_results in results.values():
            if 'spatial' in problem_results and 'temporal' in problem_results:
                for solver_results in problem_results['spatial'].values():
                    total_configs += len(solver_results)
                    total_passing += sum(1 for r in solver_results if r['metrics']['is_valid'])
                for solver_results in problem_results['temporal'].values():
                    total_configs += len(solver_results)
                    total_passing += sum(1 for r in solver_results if r['metrics']['is_valid'])

        print(f"Total configurations tested: {total_configs}")
        print(f"Configurations passing (<1.0% error): {total_passing}/{total_configs}")
        print(f"1D Problems tested: {len([k for k in results.keys() if 'spatial' in results[k]])}")
    else:
        # Regular benchmark mode
        total_configs = sum(
            len(solver_results)
            for problem_results in results.values()
            for solver_results in problem_results.values()
        )

        total_passing = sum(
            1
            for problem_results in results.values()
            for solver_results in problem_results.values()
            for result in solver_results
            if result['metrics']['is_valid']
        )

        print(f"Total configurations tested: {total_configs}")
        print(f"Configurations passing (<{args.target_error*100:.1f}% error): {total_passing}/{total_configs}")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
