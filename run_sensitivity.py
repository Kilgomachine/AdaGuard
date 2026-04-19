#!/usr/bin/env python3
"""AdaGuard Sensitivity & Viability Study Runner.

Case 1 (Sensitivity): Sweep one variable at a time to understand its effect.
Case 2 (Viability): Head-to-head comparisons proving AdaGuard works vs competitors.

Usage:
    # List all scenarios
    python run_sensitivity.py --list

    # Run all sensitivity scenarios
    python run_sensitivity.py --case sensitivity --artifacts-dir /path/to/artifacts

    # Run a specific sensitivity group
    python run_sensitivity.py --case sensitivity --group S1 --artifacts-dir /path/to/artifacts

    # Run specific viability scenarios
    python run_sensitivity.py --case viability --scenario V4,V5 --artifacts-dir /path/to/artifacts

    # Run a single scenario by ID (works for both cases)
    python run_sensitivity.py --scenario S3.2 --artifacts-dir /path/to/artifacts

    # Slurm array job mode: run scenario at SLURM_ARRAY_TASK_ID index
    python run_sensitivity.py --slurm-index 0 --artifacts-dir /path/to/artifacts

Artifact Reuse Strategy:
    Most scenarios replay Phase 1 artifacts with different encryption/scoring —
    no retraining needed. The runner auto-detects when metrics need recomputation
    (e.g. S1 changes entropy_bins → recomputes entropy scores from saved gradients).

    Scenarios that REQUIRE separate Phase 1 training runs:
    - S2  (grad_accum_K):  K changes gradient computation at training time
    - S8  (batch_size):    Different batches = different raw gradients
    - V10 (100 clients):   Different FL topology
    - V11 (1000 clients):  Different FL topology
    - V12 (50% participation): Different client selection
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='AdaGuard Sensitivity & Viability Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--list', action='store_true',
                        help='List all available scenarios and exit')
    parser.add_argument('--case', choices=['sensitivity', 'viability', 'all'],
                        help='Run all scenarios in a case')
    parser.add_argument('--group', type=str,
                        help='Run a specific sensitivity group (e.g. S1, S5)')
    parser.add_argument('--scenario', type=str,
                        help='Comma-separated scenario IDs (e.g. V4,V5 or S3.2)')
    parser.add_argument('--slurm-index', type=int, default=None,
                        help='Slurm array task index (0-based)')
    parser.add_argument('--artifacts-dir', type=str,
                        help='Path to Phase 1 training artifacts')
    parser.add_argument('--config', type=str, default=None,
                        help='Attack config YAML override')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity',
                        help='Directory for output JSON and CSV')
    parser.add_argument('--tb-dir', type=str, default=None,
                        help='Tensorboard log directory')
    parser.add_argument('--rounds', type=str, default=None,
                        help='Comma-separated round indices to evaluate (default: all)')
    parser.add_argument('--no-attacks', action='store_true',
                        help='Skip full attacks (metrics only)')
    return parser.parse_args()


def get_scenario_list(args):
    """Resolve which scenarios to run based on CLI args."""
    from adaguard.scenarios.registry import (
        ALL_SCENARIOS, SENSITIVITY_SCENARIOS, VIABILITY_SCENARIOS,
        SENSITIVITY_GROUPS, get_sensitivity_group,
    )

    if args.slurm_index is not None:
        # Slurm array mode: map index to sorted scenario list
        all_ids = sorted(ALL_SCENARIOS.keys(), key=_scenario_sort_key)
        idx = args.slurm_index
        if idx >= len(all_ids):
            print(f"Slurm index {idx} out of range (max {len(all_ids) - 1})")
            sys.exit(1)
        return [ALL_SCENARIOS[all_ids[idx]]]

    if args.scenario:
        ids = [s.strip() for s in args.scenario.split(',')]
        scenarios = []
        for sid in ids:
            if sid not in ALL_SCENARIOS:
                print(f"Unknown scenario: {sid}")
                sys.exit(1)
            scenarios.append(ALL_SCENARIOS[sid])
        return scenarios

    if args.group:
        groups = [g.strip() for g in args.group.split(',')]
        scenarios = []
        for g in groups:
            scenarios.extend(get_sensitivity_group(g))
        return scenarios

    if args.case == 'sensitivity':
        return [SENSITIVITY_SCENARIOS[k] for k in sorted(SENSITIVITY_SCENARIOS.keys(), key=_scenario_sort_key)]
    elif args.case == 'viability':
        return [VIABILITY_SCENARIOS[k] for k in sorted(VIABILITY_SCENARIOS.keys(), key=_scenario_sort_key)]
    elif args.case == 'all':
        return [ALL_SCENARIOS[k] for k in sorted(ALL_SCENARIOS.keys(), key=_scenario_sort_key)]

    print("Specify --case, --group, --scenario, or --slurm-index. Use --list to see options.")
    sys.exit(1)


def _scenario_sort_key(sid):
    """Sort scenario IDs: S1.1 < S1.2 < ... < S11.3 < V1 < V12."""
    if sid.startswith('S'):
        parts = sid[1:].split('.')
        return (0, int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    elif sid.startswith('V'):
        return (1, int(sid[1:]), 0)
    return (2, 0, 0)


def run_scenario(scenario, args):
    """Run a single scenario and return results."""
    from adaguard.scenarios.runner import ScenarioRunner
    import yaml

    sid = scenario['id']
    print(f"\n{'#' * 70}")
    print(f"  Running scenario {sid}: {scenario.get('description', '')}")
    print(f"  Overrides: {scenario.get('config_overrides', {})}")
    print(f"{'#' * 70}")

    # Load just the yaml overrides — NOT DEFAULT_CONFIG + yaml, which is what
    # adaguard.config.load_config returns. A merged DEFAULT_CONFIG would stomp
    # the Phase-1 training config (model=resnet18 → smallcnn) when the runner
    # layers attack_config on top of training_config.
    attack_config = {}
    if args.config:
        with open(args.config) as f:
            attack_config = yaml.safe_load(f) or {}

    # Initialize runner
    runner = ScenarioRunner(
        artifacts_dir=args.artifacts_dir,
        scenario_config=scenario,
        attack_config=attack_config,
    )

    # Determine which rounds to evaluate
    available_rounds = runner.get_saved_rounds()
    if args.rounds:
        target_rounds = [int(r) for r in args.rounds.split(',')]
        target_rounds = [r for r in target_rounds if r in available_rounds]
    else:
        # Default: evaluate a sample of rounds (first, middle, last)
        if len(available_rounds) <= 5:
            target_rounds = available_rounds
        else:
            n = len(available_rounds)
            target_rounds = [
                available_rounds[0],
                available_rounds[n // 4],
                available_rounds[n // 2],
                available_rounds[3 * n // 4],
                available_rounds[-1],
            ]

    if not target_rounds:
        print(f"  No rounds to evaluate (available: {available_rounds})")
        return {'scenario_id': sid, 'rounds': [], 'error': 'no rounds'}

    print(f"  Evaluating rounds: {target_rounds} (of {len(available_rounds)} available)")

    # TB logger
    tb_logger = None
    if args.tb_dir:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from run_headless import TBLogger
        case = scenario.get('case', 'unknown')
        group = scenario.get('group', '')
        tb_subdir = f"{case}/{group}/{sid}"
        overrides = scenario.get('config_overrides', {})
        strategy_label = overrides.get('encryption', group or sid)
        tb_logger = TBLogger(
            os.path.join(args.tb_dir, tb_subdir),
            runner.config,
            strategy_label,
            run_name=sid,
        )

    # Run each round
    round_results = []
    t0 = time.time()
    for rnd in target_rounds:
        try:
            result = runner.run_round(
                rnd, run_attacks=not args.no_attacks, tb_logger=tb_logger,
            )
            round_results.append(result)
        except Exception as e:
            print(f"  Round {rnd} failed: {e}")
            round_results.append({'round': rnd, 'error': str(e)})

    total_time = time.time() - t0

    if tb_logger:
        tb_logger.close()

    # Compile results
    scenario_result = {
        'scenario_id': sid,
        'case': scenario.get('case', ''),
        'group': scenario.get('group', ''),
        'description': scenario.get('description', ''),
        'config_overrides': scenario.get('config_overrides', {}),
        'variable': scenario.get('variable', ''),
        'value': scenario.get('value', ''),
        'rounds': round_results,
        'total_time_s': total_time,
    }

    # Save per-scenario JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{sid}.json'
    with open(out_path, 'w') as f:
        json.dump(scenario_result, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")

    return scenario_result


def write_sensitivity_csv(results, output_dir):
    """Write a CSV summary for sensitivity study results, grouped by sweep variable."""
    groups = {}
    for r in results:
        if r.get('case') != 'sensitivity':
            continue
        grp = r.get('group', '')
        if grp not in groups:
            groups[grp] = []
        groups[grp].append(r)

    out_dir = Path(output_dir)
    for grp, grp_results in groups.items():
        csv_path = out_dir / f'sensitivity_{grp}.csv'
        if not grp_results:
            continue

        # Collect metric keys from round results
        metric_keys = set()
        for r in grp_results:
            for rnd in r.get('rounds', []):
                if isinstance(rnd, dict):
                    for k, v in rnd.items():
                        if isinstance(v, (int, float)):
                            metric_keys.add(k)
        metric_keys = sorted(metric_keys)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['scenario_id', 'variable', 'value'] + metric_keys
            writer.writerow(header)

            for r in sorted(grp_results, key=lambda x: str(x.get('value', ''))):
                # Average metrics across rounds
                avgs = {}
                for k in metric_keys:
                    vals = [rnd[k] for rnd in r.get('rounds', [])
                            if isinstance(rnd, dict) and k in rnd
                            and isinstance(rnd[k], (int, float))]
                    avgs[k] = sum(vals) / len(vals) if vals else ''

                row = [r['scenario_id'], r.get('variable', ''), r.get('value', '')]
                row.extend([avgs.get(k, '') for k in metric_keys])
                writer.writerow(row)

        print(f"  Sensitivity CSV: {csv_path}")


def write_viability_csv(results, output_dir):
    """Write a CSV summary for viability study results."""
    viability = [r for r in results if r.get('case') == 'viability']
    if not viability:
        return

    out_dir = Path(output_dir)
    csv_path = out_dir / 'viability_summary.csv'

    # Collect all metric keys
    metric_keys = set()
    for r in viability:
        for rnd in r.get('rounds', []):
            if isinstance(rnd, dict):
                for k, v in rnd.items():
                    if isinstance(v, (int, float)):
                        metric_keys.add(k)
                # Also grab attack summary metrics
                attacks = rnd.get('attacks', {})
                summary = attacks.get('summary', {})
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        metric_keys.add(f'attack_{k}')
    metric_keys = sorted(metric_keys)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['scenario_id', 'name', 'description'] + metric_keys
        writer.writerow(header)

        for r in viability:
            avgs = {}
            for k in metric_keys:
                raw_k = k[7:] if k.startswith('attack_') else k
                vals = []
                for rnd in r.get('rounds', []):
                    if not isinstance(rnd, dict):
                        continue
                    if k.startswith('attack_'):
                        summary = rnd.get('attacks', {}).get('summary', {})
                        if raw_k in summary and isinstance(summary[raw_k], (int, float)):
                            vals.append(summary[raw_k])
                    elif raw_k in rnd and isinstance(rnd[raw_k], (int, float)):
                        vals.append(rnd[raw_k])
                avgs[k] = sum(vals) / len(vals) if vals else ''

            row = [r['scenario_id'], r.get('description', '')[:40], r.get('description', '')]
            row.extend([avgs.get(k, '') for k in metric_keys])
            writer.writerow(row)

    print(f"  Viability CSV: {csv_path}")


def main():
    args = parse_args()

    if args.list:
        from adaguard.scenarios.registry import list_scenarios
        list_scenarios()
        return

    if not args.artifacts_dir and not args.list:
        print("Error: --artifacts-dir is required (or use --list)")
        sys.exit(1)

    scenarios = get_scenario_list(args)
    print(f"\n{'=' * 70}")
    print(f"  AdaGuard Study: {len(scenarios)} scenario(s) to run")
    print(f"{'=' * 70}")

    all_results = []
    for i, scenario in enumerate(scenarios):
        print(f"\n  [{i+1}/{len(scenarios)}]")
        result = run_scenario(scenario, args)
        all_results.append(result)

    # Write summary CSVs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_sensitivity_csv(all_results, args.output_dir)
    write_viability_csv(all_results, args.output_dir)

    # Save combined results
    combined_path = out_dir / 'all_results.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(all_results)} scenarios completed")
    print(f"  Results: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
