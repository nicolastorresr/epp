"""
visualization.py
Generates all publication-quality figures for the EPP experiment.

Figures:
  1. Stability by model (bar chart)
  2. Compliance by model (bar chart)
  3. Confidence Drift by model (boxplot)
  4. Pressure susceptibility heatmap (compliance)
  5. Epistemic Personality Profile radar chart
"""

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (8, 5)
FIGURE_SIZE_WIDE = (10, 6)
FIGURE_SIZE_SQUARE = (7, 7)

PALETTE = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # amber
    "#00BCD4",  # cyan
]

FONT_TITLE = {"fontsize": 14, "fontweight": "bold"}
FONT_LABEL = {"fontsize": 11}
FONT_TICK  = {"fontsize": 9}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _short_model_name(model_id: str) -> str:
    """Return a short display name from a full model ID string."""
    name_map = {
        "openai/gpt-4o": "GPT-4o",
        "anthropic/claude-sonnet-4-5": "Claude\nSonnet 4.5",
        "anthropic/claude-sonnet-4.5": "Claude\nSonnet 4.5",
        "google/gemini-2.0-flash": "Gemini\n2.0 Flash",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3\n70B",
    }
    return name_map.get(model_id, model_id.split("/")[-1])


def _short_pressure_name(pt: str) -> str:
    label_map = {
        "authority": "Authority",
        "majority": "Majority",
        "emotional": "Emotional",
        "iterative_contradiction": "Iterative\nContradiction",
    }
    return label_map.get(pt, pt)


# ---------------------------------------------------------------------------
# Figure 1: Stability by model
# ---------------------------------------------------------------------------

def plot_stability_by_model(
    model_metrics: dict[str, dict],
    output_path: str | Path,
    bootstrap_ci: Optional[dict[str, dict]] = None,
) -> None:
    """Bar chart: Stability by model with optional bootstrap CI error bars."""
    models = sorted(model_metrics.keys())
    labels = [_short_model_name(m) for m in models]
    values = [model_metrics[m].get("stability", 0.0) for m in models]
    values = [v if not math.isnan(v) else 0.0 for v in values]

    yerr = None
    if bootstrap_ci:
        lo = [bootstrap_ci.get(m, {}).get("ci_lower", v) for m, v in zip(models, values)]
        hi = [bootstrap_ci.get(m, {}).get("ci_upper", v) for m, v in zip(models, values)]
        yerr_lo = [max(0, v - l) for v, l in zip(values, lo)]
        yerr_hi = [max(0, h - v) for v, h in zip(values, hi)]
        yerr = [yerr_lo, yerr_hi]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
    x = np.arange(len(models))
    bars = ax.bar(
        x, values, color=PALETTE[:len(models)],
        edgecolor="white", linewidth=0.8, width=0.55,
        yerr=yerr, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT_TICK)
    ax.set_ylabel("Stability (proportion)", **FONT_LABEL)
    ax.set_title("Figure 1 — Epistemic Stability by Model", **FONT_TITLE)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=9, color="#333333",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 1 to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 2: Compliance by model
# ---------------------------------------------------------------------------

def plot_compliance_by_model(
    model_metrics: dict[str, dict],
    output_path: str | Path,
) -> None:
    """Bar chart: Compliance by model."""
    models = sorted(model_metrics.keys())
    labels = [_short_model_name(m) for m in models]
    values = [model_metrics[m].get("compliance", 0.0) for m in models]
    values = [v if not math.isnan(v) else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
    x = np.arange(len(models))
    bars = ax.bar(
        x, values, color=PALETTE[:len(models)],
        edgecolor="white", linewidth=0.8, width=0.55,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT_TICK)
    ax.set_ylabel("Compliance (proportion)", **FONT_LABEL)
    ax.set_title("Figure 2 — Epistemic Compliance by Model", **FONT_TITLE)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=9, color="#333333",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 2 to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 3: Confidence Drift by model (boxplot)
# ---------------------------------------------------------------------------

def plot_confidence_drift_boxplot(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """Boxplot: Confidence Drift distribution per model."""
    from itertools import groupby

    # Build per-model drift lists from raw rows
    models = sorted(set(r["model"] for r in rows))

    def get_drifts(model_rows):
        from collections import defaultdict
        by_trial = defaultdict(list)
        for r in model_rows:
            by_trial[r["experiment_id"]].append(r)
        drifts = []
        for turns in by_trial.values():
            turns_sorted = sorted(turns, key=lambda t: t["turn"])
            t0 = next((t for t in turns_sorted if t["turn"] == 0), None)
            t3 = next((t for t in turns_sorted if t["turn"] == 3), None)
            if t0 and t3:
                drifts.append(abs(int(t3["confidence"]) - int(t0["confidence"])))
        return drifts

    data = []
    labels = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        drifts = get_drifts(model_rows)
        data.append(drifts if drifts else [0])
        labels.append(_short_model_name(model))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
    bp = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, **FONT_TICK)
    ax.set_ylabel("Confidence Drift |Δ confidence| (points)", **FONT_LABEL)
    ax.set_title("Figure 3 — Confidence Drift Distribution by Model", **FONT_TITLE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 3 to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 4: Pressure susceptibility heatmap
# ---------------------------------------------------------------------------

def plot_pressure_heatmap(
    heatmap_data: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Heatmap: Compliance rate for each (model, pressure_type) cell."""
    models = sorted(heatmap_data.keys())
    if not models:
        logger.warning("No data for heatmap; skipping.")
        return

    pts = sorted(next(iter(heatmap_data.values())).keys())
    matrix = np.array([
        [heatmap_data[m].get(pt, 0.0) for pt in pts]
        for m in models
    ], dtype=float)
    # Replace nan with 0
    matrix = np.nan_to_num(matrix, nan=0.0)

    row_labels = [_short_model_name(m) for m in models]
    col_labels = [_short_pressure_name(pt) for pt in pts]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Compliance Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    ax.set_xticks(np.arange(len(pts)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(col_labels, **FONT_TICK)
    ax.set_yticklabels(row_labels, **FONT_TICK)
    ax.set_title("Figure 4 — Pressure Susceptibility Heatmap (Compliance)", **FONT_TITLE)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(pts)):
            val = matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 4 to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 5: EPP Radar chart
# ---------------------------------------------------------------------------

def plot_epp_radar(
    normalized_profiles: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Radar chart: Epistemic Personality Profiles for all models."""
    dimensions = ["stability", "compliance", "confidence_drift_mean"]
    dim_labels = ["Stability", "Compliance", "Confidence\nDrift"]

    models = sorted(normalized_profiles.keys())
    n_dims = len(dimensions)

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE_SQUARE,
        subplot_kw={"projection": "polar"},
    )

    legend_handles = []
    for idx, model in enumerate(models):
        color = PALETTE[idx % len(PALETTE)]
        profile = normalized_profiles[model]
        values = [profile.get(dim, 0.0) for dim in dimensions]
        values = [v if not math.isnan(v) else 0.0 for v in values]
        values += values[:1]  # close

        ax.plot(angles, values, color=color, linewidth=2.0, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.15)
        legend_handles.append(
            mpatches.Patch(color=color, label=_short_model_name(model).replace("\n", " "))
        )

    # Grid and labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), dim_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_title(
        "Figure 5 — Epistemic Personality Profiles",
        pad=20, **FONT_TITLE,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 5 to %s", output_path)


# ---------------------------------------------------------------------------
# Convenience: generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    analysis_results: dict,
    raw_rows: list[dict],
    output_dir: str | Path,
) -> None:
    """
    Generate all 5 EPP figures.

    Args:
        analysis_results: Output dict from analysis.run_analysis().
        raw_rows: Raw list of row dicts from results.csv.
        output_dir: Directory to save figures.
    """
    import matplotlib.ticker  # ensure available

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_metrics = analysis_results["model_metrics"]
    heatmap_data = analysis_results["heatmap_data"]
    bootstrap_ci = analysis_results["statistical_tests"].get("bootstrap_stability", {})

    # Compute normalized profiles for radar
    from metrics import normalize_profile
    raw_profiles = {
        m: {
            "stability": model_metrics[m].get("stability", 0.0),
            "compliance": model_metrics[m].get("compliance", 0.0),
            "confidence_drift_mean": model_metrics[m].get("confidence_drift_mean", 0.0),
        }
        for m in model_metrics
    }
    normalized = normalize_profile(raw_profiles)

    plot_stability_by_model(model_metrics, out / "fig1_stability_by_model.png", bootstrap_ci)
    plot_compliance_by_model(model_metrics, out / "fig2_compliance_by_model.png")
    plot_confidence_drift_boxplot(raw_rows, out / "fig3_confidence_drift_boxplot.png")
    plot_pressure_heatmap(heatmap_data, out / "fig4_pressure_heatmap.png")
    plot_epp_radar(normalized, out / "fig5_epp_radar.png")

    logger.info("All 5 figures saved to %s", out)
