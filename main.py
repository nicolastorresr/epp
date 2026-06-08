"""
main.py
Entry point for the EPP experiment.

Usage:
    python main.py                        # full experiment
    python main.py --analyze-only         # skip experiment, just analyze existing results
    python main.py --models-config config/models.json
    python main.py --seed 123

Environment variable required:
    OPENROUTER_API_KEY
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── make src/ importable ──────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))
# ─────────────────────────────────────────────────────────────────────────────

from experiment_runner import ExperimentRunner, ExperimentConfig
from analysis import run_analysis, load_results_csv
from visualization import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).parent
CONFIG_PATH    = BASE_DIR / "config" / "models.json"
QUESTIONS_PATH = BASE_DIR / "data"  / "questions.json"
TEMPLATES_PATH = BASE_DIR / "data"  / "pressure_templates.json"
RESULTS_DIR    = BASE_DIR / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
ANALYSIS_DIR   = RESULTS_DIR / "analysis"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(cfg: dict, seed: int) -> None:
    """Run the full multi-model experiment."""
    exp_cfg = ExperimentConfig(
        models=cfg["models"],
        questions_path=str(QUESTIONS_PATH),
        templates_path=str(TEMPLATES_PATH),
        results_dir=str(RESULTS_DIR),
        seed=seed,
        max_retries=cfg["experiment"].get("max_retries", 3),
        backoff_base=cfg["experiment"].get("retry_backoff_base", 2.0),
        request_timeout=cfg["experiment"].get("request_timeout", 30),
    )
    runner = ExperimentRunner(exp_cfg)
    runner.run()


def run_analysis_and_viz() -> None:
    """Load results and run analysis + visualization."""
    results_csv = RESULTS_DIR / "results.csv"
    if not results_csv.exists():
        logger.error("results.csv not found at %s. Run the experiment first.", results_csv)
        sys.exit(1)

    logger.info("Running analysis...")
    analysis_results = run_analysis(str(results_csv), str(ANALYSIS_DIR))

    logger.info("Generating figures...")
    raw_rows = load_results_csv(results_csv)
    generate_all_figures(analysis_results, raw_rows, str(FIGURES_DIR))

    # Print summary table
    _print_summary(analysis_results["model_metrics"])


def _print_summary(model_metrics: dict) -> None:
    """Pretty-print a summary of per-model EPP metrics."""
    header = f"{'Model':<45} {'Stability':>10} {'Compliance':>12} {'CD Mean':>10} {'CD Std':>10}"
    print("\n" + "=" * len(header))
    print("EPISTEMIC PERSONALITY PROFILE SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, m in sorted(model_metrics.items()):
        s   = m.get("stability", float("nan"))
        c   = m.get("compliance", float("nan"))
        cdm = m.get("confidence_drift_mean", float("nan"))
        cds = m.get("confidence_drift_std", float("nan"))

        def fmt(v): return f"{v:.3f}" if v == v else "N/A"  # nan check

        print(f"{model:<45} {fmt(s):>10} {fmt(c):>12} {fmt(cdm):>10} {fmt(cds):>10}")
    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EPP Experiment — Epistemic Personality Profiles for LLMs"
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to models.json configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip experiment, only run analysis on existing results.csv.",
    )
    parser.add_argument(
        "--experiment-only",
        action="store_true",
        help="Run experiment only, skip analysis and visualization.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.models_config))
    logger.info("Loaded config from %s", args.models_config)

    if not args.analyze_only:
        logger.info("=== PHASE 1: Running Experiment ===")
        run_experiment(cfg, seed=args.seed)

    if not args.experiment_only:
        logger.info("=== PHASE 2: Analysis & Visualization ===")
        run_analysis_and_viz()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
