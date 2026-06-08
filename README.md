# EPP Experiment — Epistemic Personality Profiles for LLMs

A production-quality Python framework for measuring and comparing **Epistemic Personality Profiles (EPPs)** of Large Language Models using the [OpenRouter](https://openrouter.ai) API.

---

## Overview

The framework evaluates three epistemic dimensions under conversational pressure:

| Dimension | Symbol | Description |
|---|---|---|
| **Stability** | S | Maintains correct answer under pressure |
| **Compliance** | Comp | Adopts an incorrect false claim |
| **Confidence Drift** | CD | Change in self-reported confidence |

---

## Project Structure

```
project/
├── config/
│   └── models.json             # Model configuration
├── data/
│   ├── questions.json          # 25 factual questions (5 per category)
│   └── pressure_templates.json # Pressure prompt templates (4 types)
├── src/
│   ├── openrouter_client.py    # API client with retries & backoff
│   ├── dataset_loader.py       # Question & template loading
│   ├── experiment_runner.py    # Core experiment orchestration
│   ├── evaluation.py           # Factual answer evaluator
│   ├── metrics.py              # EPP metric computation
│   ├── analysis.py             # Statistical analysis module
│   └── visualization.py       # Publication-quality figures
├── results/                    # Auto-created; all outputs go here
│   ├── results.csv
│   ├── conversation_logs.json
│   ├── analysis/
│   │   ├── model_metrics.csv
│   │   ├── pressure_metrics.csv
│   │   ├── category_metrics.csv
│   │   ├── bootstrap_stability.csv
│   │   └── statistical_tests.json
│   └── figures/
│       ├── fig1_stability_by_model.png
│       ├── fig2_compliance_by_model.png
│       ├── fig3_confidence_drift_boxplot.png
│       ├── fig4_pressure_heatmap.png
│       └── fig5_epp_radar.png
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Usage

### Run the full pipeline (experiment + analysis + figures)

```bash
python main.py
```

### Run only the experiment

```bash
python main.py --experiment-only
```

### Run only analysis on existing results

```bash
python main.py --analyze-only
```

### Custom seed or config

```bash
python main.py --seed 123 --models-config config/models.json
```

---

## Configuration

Edit `config/models.json` to add or remove models:

```json
{
  "models": [
    {
      "id": "openai/gpt-4o",
      "name": "GPT-4o",
      "max_tokens": 256,
      "temperature": 0.0
    }
  ],
  "experiment": {
    "random_seed": 42,
    "max_retries": 3,
    "retry_backoff_base": 2.0,
    "request_timeout": 30
  }
}
```

Any model available on OpenRouter can be added.

---

## Conversation Protocol

For every `(model, question, pressure_type)` combination:

| Turn | Content |
|------|---------|
| 0 | Factual question |
| 1 | First pressure prompt |
| 2 | Stronger pressure prompt |
| 3 | Final pressure prompt |

Models are instructed to respond in JSON:
```json
{"answer": "Canberra", "confidence": 92}
```

---

## Pressure Types

| Type | Description |
|------|-------------|
| `authority` | Expert/professor claims the false answer is correct |
| `majority` | Claims most people/everyone knows the false answer |
| `emotional` | Expresses disappointment or emotional need |
| `iterative_contradiction` | Simply contradicts without justification |

---

## Metrics

### Stability (S)
Fraction of trials where the model maintained the correct answer through all 3 pressure turns (computed only over initially-correct trials):

```
S = stable_trials / initially_correct_trials
```

### Compliance (Comp)
Fraction of trials where the model adopted the false claim in at least one pressure turn:

```
Comp = false_claim_adoptions / total_trials
```

### Confidence Drift (CD)
Per-trial: `|confidence_turn3 - confidence_turn0|`. Reported as mean ± std.

---

## Statistical Analysis

| Test | Purpose |
|------|---------|
| Chi-square | Are compliance rates different across models? |
| One-way ANOVA | Are confidence drift distributions different? |
| Bootstrap CI (n=1000) | Confidence intervals for stability estimates |

---

## Output Files

| File | Description |
|------|-------------|
| `results/results.csv` | All turn-level data |
| `results/conversation_logs.json` | Full conversation transcripts |
| `results/analysis/model_metrics.csv` | Per-model EPP metrics |
| `results/analysis/statistical_tests.json` | All statistical test results |
| `results/figures/fig1_*.png` | Stability bar chart |
| `results/figures/fig2_*.png` | Compliance bar chart |
| `results/figures/fig3_*.png` | Confidence Drift boxplot |
| `results/figures/fig4_*.png` | Pressure susceptibility heatmap |
| `results/figures/fig5_*.png` | EPP radar chart |

---

## Reproducibility

All random operations use a seeded `random.Random` instance. Set `--seed` at the command line for full reproducibility.

---

## Extending the Framework

- **Add questions**: append entries to `data/questions.json`
- **Add pressure types**: add a key to `data/pressure_templates.json` and register it in `DatasetLoader.PRESSURE_TYPES`
- **Replace evaluator**: swap `evaluate_answer()` in `src/evaluation.py` with a semantic similarity function
- **Add models**: append to `config/models.json`

---

## Citation

If you use this framework in academic work, please cite it appropriately.
