"""
main.py
────────
Entry point for the Credit Risk Factory.

Usage:
  python main.py                                  # runs on default dev dataset
  python main.py --dataset /path/to/dataset2.csv  # plug-and-play Dataset 2
  python main.py --auto                            # auto-approve all checkpoints (demo mode)
  python main.py --trials 50                       # more Optuna trials
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import CreditRiskOrchestrator


DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Loan_Data_Development.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic Credit Risk Factory")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to input CSV file")
    parser.add_argument("--name", default="",
                        help="Dataset label for the report")
    parser.add_argument("--output", default="outputs",
                        help="Output directory for reports and artefacts")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-approve all human checkpoints (demo/CI mode)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna hyperparameter trials")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose agent logs")
    parser.add_argument("--target-col", default="",
                        help="Override target column name (skips auto-detection)")
    parser.add_argument("--hyperparam-override", action="store_true",
                        help="Apply an active hyperparameter_override from outputs/checkpoints.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at {args.dataset}")
        sys.exit(1)

    orchestrator = CreditRiskOrchestrator(
        output_dir    = args.output,
        optuna_trials = args.trials,
        auto_approve  = args.auto,
        verbose       = not args.quiet,
        hyperparam_override = args.hyperparam_override,
    )

    state = orchestrator.run(
        dataset_path = args.dataset,
        dataset_name = args.name or os.path.basename(args.dataset),
        target_col   = args.target_col,
    )

    return 0 if state.validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
