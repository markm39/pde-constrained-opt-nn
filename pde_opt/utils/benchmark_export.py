"""
Export utilities for benchmark results.

Supports multiple formats: Markdown, CSV, LaTeX, and JSON.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import jax.numpy as jnp


def _format_params(params: Dict[str, Any]) -> str:
    """Format problem parameters as a compact string."""
    if not params:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _convert_to_native_types(obj):
    """
    Recursively convert JAX arrays to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, or JAX array)

    Returns:
        Object with JAX arrays converted to Python types
    """
    if isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (jnp.ndarray, jnp.number)):
        # Convert JAX array/scalar to native Python type
        return float(obj) if obj.size == 1 else obj.tolist()
    else:
        return obj


def _flatten_convergence_results(results: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Flatten convergence study results to standard format for export.

    Convergence format: {problem: {'spatial': {solver: [...]}, 'temporal': {solver: [...]}}}
    Standard format: {problem: {solver: [...]}}

    Args:
        results: Convergence study results or standard results

    Returns:
        Flattened results in standard format
    """
    flattened = {}

    for problem_key, problem_results in results.items():
        # Check if this is convergence format
        if isinstance(problem_results, dict) and 'spatial' in problem_results and 'temporal' in problem_results:
            # Convergence format - combine spatial and temporal
            combined = {}
            for solver, spatial_results in problem_results.get('spatial', {}).items():
                combined[f"{solver}_spatial"] = spatial_results
            for solver, temporal_results in problem_results.get('temporal', {}).items():
                combined[f"{solver}_temporal"] = temporal_results
            flattened[problem_key] = combined
        else:
            # Standard format - pass through
            flattened[problem_key] = problem_results

    return flattened


def export_to_markdown(results: Dict[str, Dict[str, List[Dict[str, Any]]]], filename: str) -> None:
    """
    Export benchmark results to Markdown format.

    Args:
        results: Nested dict {problem: {discretization: [results]}}
        filename: Output file path
    """
    # Flatten convergence results if needed
    results = _flatten_convergence_results(results)

    with open(filename, 'w') as f:
        f.write("# Solver Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for problem_key, solver_results in results.items():
            f.write(f"## {problem_key}\n\n")

            # Determine if 2D
            first_result = next(iter(solver_results.values()))[0] if solver_results else None
            is_2d = first_result and 'ny' in first_result['grid_params']

            # Table header
            if is_2d:
                f.write("| Discretization | nx | ny | nt | DOFs | Rel Error (%) | MSE | Status |\n")
                f.write("|----------------|----|----|-------|------|---------------|-----|--------|\n")
            else:
                f.write("| Discretization | nx | nt | DOFs | Rel Error (%) | MSE | Status |\n")
                f.write("|----------------|----|----|------|---------------|-----|--------|\n")

            # Table rows
            for discretization, bench_results in solver_results.items():
                for result in bench_results:
                    gp = result['grid_params']
                    m = result['metrics']
                    status = "✓" if m['is_valid'] else "✗"

                    if is_2d:
                        f.write(f"| {discretization} | {gp['nx']} | {gp['ny']} | {gp['nt']} | "
                                f"{result['total_dofs']:,} | {m['rel_error_pct']:.2f} | "
                                f"{m['mse']:.2e} | {status} |\n")
                    else:
                        f.write(f"| {discretization} | {gp['nx']} | {gp['nt']} | "
                                f"{result['total_dofs']:,} | {m['rel_error_pct']:.2f} | "
                                f"{m['mse']:.2e} | {status} |\n")

            f.write("\n")

    print(f"✓ Markdown table saved to: {filename}")


def export_to_csv(results: Dict[str, Dict[str, List[Dict[str, Any]]]], filename: str) -> None:
    """
    Export benchmark results to CSV format.

    Args:
        results: Nested dict {problem: {discretization: [results]}}
        filename: Output file path
    """
    # Flatten convergence results if needed
    results = _flatten_convergence_results(results)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Problem', 'Parameters', 'Discretization',
            'nx', 'ny', 'nt', 'Total_DOFs',
            'Rel_Error_%', 'MSE', 'Status'
        ])

        # Data rows
        for problem_key, solver_results in results.items():
            # Split problem name and params
            if '[' in problem_key:
                problem_name, params_str = problem_key.split('[')
                params_str = params_str.rstrip(']')
            else:
                problem_name = problem_key
                params_str = '-'

            for discretization, bench_results in solver_results.items():
                for result in bench_results:
                    gp = result['grid_params']
                    m = result['metrics']
                    status = 'PASS' if m['is_valid'] else 'FAIL'

                    ny = gp.get('ny', '')  # Empty if not 2D

                    writer.writerow([
                        problem_name,
                        params_str,
                        discretization,
                        gp['nx'],
                        ny,
                        gp['nt'],
                        result['total_dofs'],
                        f"{m['rel_error_pct']:.2f}",
                        f"{m['mse']:.2e}",
                        status
                    ])

    print(f"✓ CSV table saved to: {filename}")


def export_to_latex(results: Dict[str, Dict[str, List[Dict[str, Any]]]], filename: str) -> None:
    """
    Export benchmark results to LaTeX format.

    Args:
        results: Nested dict {problem: {discretization: [results]}}
        filename: Output file path
    """
    # Flatten convergence results if needed
    results = _flatten_convergence_results(results)

    with open(filename, 'w') as f:
        f.write("% Solver Benchmark Results\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\begin{document}\n\n")

        for problem_key, solver_results in results.items():
            # Escape underscores for LaTeX
            problem_latex = problem_key.replace('_', '\\_')
            f.write(f"\\section*{{{problem_latex}}}\n\n")

            # Determine if 2D
            first_result = next(iter(solver_results.values()))[0] if solver_results else None
            is_2d = first_result and 'ny' in first_result['grid_params']

            # Table
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")

            if is_2d:
                f.write("\\begin{tabular}{lSSSSSl}\n")
                f.write("\\toprule\n")
                f.write("Discretization & {nx} & {ny} & {nt} & {DOFs} & {Error (\\%)} & Status \\\\\n")
            else:
                f.write("\\begin{tabular}{lSSSSl}\n")
                f.write("\\toprule\n")
                f.write("Discretization & {nx} & {nt} & {DOFs} & {Error (\\%)} & Status \\\\\n")

            f.write("\\midrule\n")

            for discretization, bench_results in solver_results.items():
                disc_latex = discretization.replace('_', '\\_')
                for result in bench_results:
                    gp = result['grid_params']
                    m = result['metrics']
                    status = "\\checkmark" if m['is_valid'] else "\\times"

                    if is_2d:
                        f.write(f"{disc_latex} & {gp['nx']} & {gp['ny']} & {gp['nt']} & "
                                f"{result['total_dofs']} & {m['rel_error_pct']:.2f} & {status} \\\\\n")
                    else:
                        f.write(f"{disc_latex} & {gp['nx']} & {gp['nt']} & "
                                f"{result['total_dofs']} & {m['rel_error_pct']:.2f} & {status} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Benchmark results for {problem_latex}}}\n")
            f.write("\\end{table}\n\n")

        f.write("\\end{document}\n")

    print(f"✓ LaTeX table saved to: {filename}")


def export_to_json(results: Dict[str, Dict[str, List[Dict[str, Any]]]], filename: str) -> None:
    """
    Export benchmark results to JSON format (for programmatic use).

    Args:
        results: Nested dict {problem: {discretization: [results]}}
        filename: Output file path
    """
    # Convert JAX arrays to native Python types for JSON serialization
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': _convert_to_native_types(results)
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ JSON data saved to: {filename}")


def export_all_formats(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    base_name: str = None
) -> Dict[str, str]:
    """
    Export benchmark results to all supported formats.

    Args:
        results: Benchmark results
        output_dir: Output directory
        base_name: Base filename (default: benchmark_TIMESTAMP)

    Returns:
        Dictionary mapping format to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if base_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"benchmark_{timestamp}"

    files = {}

    # Export to each format
    files['markdown'] = str(output_path / f"{base_name}.md")
    export_to_markdown(results, files['markdown'])

    files['csv'] = str(output_path / f"{base_name}.csv")
    export_to_csv(results, files['csv'])

    files['latex'] = str(output_path / f"{base_name}.tex")
    export_to_latex(results, files['latex'])

    files['json'] = str(output_path / f"{base_name}.json")
    export_to_json(results, files['json'])

    return files
