"""AdaGuard Simulator — CLI entry point.

Usage:
    python main.py --experiment fl_rounds
    python main.py --experiment comparison
    python main.py --experiment weight_study
    python main.py --experiment batch_size
    python main.py --experiment noise
    python main.py --experiment all
    python main.py --config path/to/config.yaml --experiment fl_rounds
"""

import argparse
import sys

from adaguard.experiments.runner import ExperimentRunner
from adaguard.visualization.plots import AdaGuardPlotter


def main():
    parser = argparse.ArgumentParser(
        description='AdaGuard: Leakage-Aware Adaptive Encryption for FL',
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['fl_rounds', 'comparison', 'weight_study', 'batch_size', 'noise', 'all'],
        default='fl_rounds',
        help='Experiment to run (default: fl_rounds)',
    )
    parser.add_argument(
        '--config', '-c',
        type=str, default=None,
        help='Path to YAML config file (optional)',
    )
    parser.add_argument(
        '--strategy', '-s',
        choices=['fisher', 'maskcrypt', 'full', 'none'],
        default='fisher',
        help='Encryption strategy for fl_rounds (default: fisher)',
    )
    parser.add_argument(
        '--skip-glmip', action='store_true',
        help='Skip GLMIP computation (faster)',
    )
    parser.add_argument(
        '--skip-empirical', action='store_true',
        help='Skip empirical attack scoring (faster)',
    )
    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)
    plotter = AdaGuardPlotter()

    if args.experiment == 'fl_rounds':
        results = runner.run_fl_rounds(
            encryption_strategy=args.strategy,
            skip_glmip=args.skip_glmip,
            skip_empirical=args.skip_empirical,
        )
        plotter.plot_fl_rounds(results, title=f'FL Rounds ({args.strategy})')

    elif args.experiment == 'comparison':
        all_results = runner.run_encryption_comparison()
        plotter.plot_encryption_comparison(all_results)

    elif args.experiment == 'weight_study':
        all_results = runner.run_weight_study()
        plotter.plot_weight_study(all_results)

    elif args.experiment == 'batch_size':
        results = runner.run_batch_size_experiment()
        plotter.plot_batch_size_experiment(results)
        plotter.plot_heatmaps(results, 'Batch Size')

    elif args.experiment == 'noise':
        results = runner.run_noise_experiment()
        plotter.plot_noise_experiment(results)
        plotter.plot_heatmaps(results, 'Noise Level')

    elif args.experiment == 'all':
        print("\n=== FL Rounds ===")
        fl_results = runner.run_fl_rounds(
            encryption_strategy='fisher',
            skip_glmip=args.skip_glmip,
            skip_empirical=args.skip_empirical,
        )
        plotter.plot_fl_rounds(fl_results, title='FL Rounds (Fisher)')

        print("\n=== Encryption Comparison ===")
        comp_results = runner.run_encryption_comparison()
        plotter.plot_encryption_comparison(comp_results)

        print("\n=== Batch Size Experiment ===")
        bs_results = runner.run_batch_size_experiment()
        plotter.plot_batch_size_experiment(bs_results)

        print("\n=== Noise Experiment ===")
        noise_results = runner.run_noise_experiment()
        plotter.plot_noise_experiment(noise_results)

        print("\n=== Weight Study ===")
        weight_results = runner.run_weight_study()
        plotter.plot_weight_study(weight_results)

    print("\nDone.")


if __name__ == '__main__':
    main()
