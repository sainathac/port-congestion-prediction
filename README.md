# Port Congestion Forecast — Aiqon Take-Home

A six-hour-ahead port-level congestion forecasting project for the Los Angeles / Long Beach (LA/LB) port region. The system estimates the likelihood of congestion, reports a calibrated event probability, and explains the main factors behind each LightGBM prediction using SHAP.

## Project summary

- **Data:** Public LA/LB AIS dataset from Kaggle, published by `nguyenductuandat` under CC BY 4.0. The source data is preprocessed by the dataset publisher; this project performs the hourly aggregation, labeling, feature engineering, model training and evaluation.
- **Modeling table:** 22,378 labeled port-hours from June 2023 to December 2025, with 67 engineered features.
- **Target:** Whether the port congestion index is above the training-period 75th-percentile threshold six hours after the feature timestamp.
- **Models:** Persistence and class-balanced logistic regression baselines, compared with constrained LightGBM.
- **Probability calibration:** Isotonic regression, fitted on validation-period predictions.
- **Explainability:** SHAP values for the raw LightGBM model output.
- **Held-out test (July–December 2025):** LightGBM ROC-AUC 0.747 and PR-AUC 0.386; calibrated Brier score 0.132. Persistence achieved the higher F1 score (0.492 vs. 0.375), which is discussed in the report.

The system forecasts congestion at the port-region level. It does **not** currently predict an individual shipment's ETA or shipment-specific delay.

## Repository layout

```text
src/                 Dataset construction, labeling, training, evaluation and SHAP CLI
tests/                Automated tests
models/               Trained LightGBM, logistic-regression and calibration artifacts
data/processed/       Included port-hour modeling table
reports/              Three-page report, metrics, explanations and supporting figures
docs/                 Congestion-label definition and assumptions
README.md              Setup and usage instructions
pyproject.toml         Python dependencies
uv.lock                Locked dependency versions
Makefile               Common project commands
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

Install dependencies from the repository root:

```bash
uv sync
```

## Run the project

The processed modeling table is included, so model training, evaluation, explanation generation and tests can be run without downloading the large raw AIS file:

```bash
uv run python -m src.train
uv run python -m src.evaluate
uv run python -m src.explain --report
uv run pytest
uv run ruff check .
```

To rebuild the processed table from the original AIS data, download and extract the Kaggle dataset so the source file is available at:

```text
archive/ais_2023_2025_clean.parquet
```

Then run:

```bash
uv run python -m src.build_dataset
```

The raw source file is intentionally not included in this repository because it is approximately 1.28 GB. The processed modeling table is included to make the modeling and evaluation steps accessible without that download.

## Makefile shortcuts

```bash
make train
make evaluate
make explain
make test
make lint
```

Use `make all` only after placing the raw AIS file at the path above; that target rebuilds the processed data before running the rest of the pipeline.

## Generate a prediction and explanation

Choose an hour that exists in `data/processed/port_hours.parquet`:

```bash
uv run python -m src.explain --hour 2025-07-15T14:00
```

The command displays:

1. The raw LightGBM probability (the output explained by SHAP).
2. The isotonic-calibrated congestion probability for the six-hour horizon.
3. A decision flag using the calibrated model's validation-selected threshold.
4. The main SHAP drivers, which explain the raw LightGBM output.

The raw-model and calibrated-model thresholds are different and are reported separately. They must not be interchanged. The report's false-positive and false-negative analysis uses the raw-model threshold to remain consistent with its evaluation results.

## Reports and reproducibility

- `reports/report.pdf` is the submission-ready technical report (3 A4 pages, within the assignment's 2–3 page requirement).
- `reports/report.md` is the editable Markdown copy of the report content.
- `reports/metrics_final.json` contains the final validation/test metrics and model-specific thresholds.
- `reports/explanations.md` contains the SHAP ranking and selected prediction/failure examples.
- `docs/label_decision.md` documents the congestion-index definition, label threshold and data-handling assumptions.

The source AIS dataset is available from [Kaggle — LA/LB AIS Port Congestion, 2023–2025](https://www.kaggle.com/datasets/nguyenductuandat/lalb-ais-port-congestion-62023-122025). Please retain the dataset publisher's CC BY 4.0 attribution when redistributing that source data.
