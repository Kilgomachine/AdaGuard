#!/usr/bin/env python3
"""AdaGuard Phase 2: Post-Training Scenario Evaluation.

Replays saved training artifacts under different encryption scenarios,
runs full paper-matched gradient inversion attacks, and computes all
reconstruction quality metrics.

Each scenario is independent and can be run separately.

Usage:
    # List all scenarios
    python run_scenario.py --list-scenarios

    # Run a single scenario
    python run_scenario.py --artifacts-dir /scratch/.../artifacts --scenario adaguard_high --tb-dir /scratch/.../tb_logs --output results.json

    # Run on specific rounds only
    python run_scenario.py --artifacts-dir /scratch/.../artifacts --scenario no_encryption --rounds 0,50,100,200,249 --output results.json

    # Run all scenarios (use with slurm_scenarios.sh)
    python run_scenario.py --artifacts-dir /scratch/.../artifacts --scenario all --output results_dir/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Force line-buffered stdout
if not os.environ.get('PYTHONUNBUFFERED'):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


def main():
    parser = argparse.ArgumentParser(
        description='AdaGuard Phase 2: Post-Training Scenario Evaluation',
    )
    parser.add_argument('--artifacts-dir', type=str, required=False,
                        help='Path to training artifacts from Phase 1')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Scenario name (or "all" to run all). Use --list-scenarios to see options.')
    parser.add_argument('--rounds', type=str, default=None,
                        help='Comma-separated round indices to evaluate (default: all saved rounds)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--tb-dir', type=str, default=None,
                        help='Tensorboard log directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional attack config YAML (overrides artifact metadata)')
    parser.add_argument('--no-attacks', action='store_true',
                        help='Skip full attacks (only compute encryption + accuracy)')
    parser.add_argument('--max-attack-clients', type=int, default=5,
                        help='Max clients to attack per round (default: 5)')
    parser.add_argument('--list-scenarios', action='store_true',
                        help='List all available scenarios and exit')
    parser.add_argument('--log', type=str, default=None,
                        help='Persistent log file')

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        from adaguard.scenarios.registry import list_scenarios
        list_scenarios()
        return

    if not args.artifacts_dir:
        parser.error("--artifacts-dir is required (unless --list-scenarios)")
    if not args.scenario:
        parser.error("--scenario is required (use --list-scenarios to see options)")

    # Set up logging
    if args.log:
        from run_headless import TeeLogger
        sys.stdout = TeeLogger(args.log)
        sys.stderr = sys.stdout

    # Load attack config
    attack_config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            attack_config = yaml.safe_load(f) or {}
    attack_config['max_attack_clients'] = args.max_attack_clients

    from adaguard.scenarios.registry import SCENARIOS, get_scenario

    # Determine which scenarios to run
    if args.scenario == 'all':
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    for scenario_name in scenario_names:
        print(f"\n{'#'*70}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'#'*70}")

        scenario_config = get_scenario(scenario_name)

        from adaguard.scenarios.runner import ScenarioRunner
        runner = ScenarioRunner(
            artifacts_dir=args.artifacts_dir,
            scenario_config=scenario_config,
            attack_config=attack_config,
        )

        # Determine rounds to evaluate
        saved_rounds = runner.get_saved_rounds()
        if args.rounds:
            rounds = [int(r.strip()) for r in args.rounds.split(',')]
            rounds = [r for r in rounds if r in saved_rounds]
        else:
            rounds = saved_rounds

        if not rounds:
            print(f"  No saved rounds found in {args.artifacts_dir}")
            continue

        print(f"  Evaluating rounds: {rounds}")

        # Set up Tensorboard
        tb_logger = None
        if args.tb_dir:
            sys.path.insert(0, str(Path(__file__).parent))
            from run_headless import TBLogger
            tb_logger = TBLogger(
                args.tb_dir,
                runner.config,
                scenario_config.get('encryption', 'unknown'),
                run_name=f"scenario_{scenario_name}",
            )

        # Run each round
        all_results = []
        total_start = time.time()

        for rnd in rounds:
            try:
                result = runner.run_round(
                    rnd,
                    run_attacks=not args.no_attacks,
                    tb_logger=tb_logger,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] Round {rnd}: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - total_start

        if tb_logger:
            tb_logger.close()

        # Save results
        if args.output:
            output = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'scenario': scenario_name,
                    'scenario_config': scenario_config,
                    'artifacts_dir': str(args.artifacts_dir),
                    'rounds_evaluated': rounds,
                    'total_time_s': total_time,
                },
                'rounds': all_results,
            }

            out_path = Path(args.output)
            if args.scenario == 'all':
                out_path = out_path / f'{scenario_name}.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Make serializable
            def _serialize(obj):
                if isinstance(obj, dict):
                    return {k: _serialize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_serialize(v) for v in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return obj

            with open(out_path, 'w') as f:
                json.dump(_serialize(output), f, indent=2, default=str)
            print(f"\n  Results saved to: {out_path}")

        print(f"\n  Scenario '{scenario_name}' complete in {total_time:.1f}s")


if __name__ == '__main__':
    main()
