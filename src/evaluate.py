"""Test-set evaluation, F1-threshold rule, comparison table (Task 5).

Scope: ``classification_metrics(y_true, y_prob, *, threshold=None)``,
``render_comparison(metrics, out_path=None)``, and the run that scores the
Task 4 artifacts on the Task 3 splits and writes
``reports/metrics_final.json`` + ``reports/figures/comparison_table.png``.

Binding evaluation rules (controller):

1. **Test labels never influence any choice.** The F1-maximizing threshold
   is selected on VAL 2025-H1 predictions only
   (``threshold_selection.per_model``) and applied unchanged to test.
   ``threshold=None`` (the brief's two-argument call form) selects the
   threshold from the arrays passed in — correct for val, never call it
   that way on test; ``evaluate_splits`` always passes the val-selected
   threshold through for the test rows.
2. **Nothing is refit.** Predictions are recomputed by loading
   ``models/lightgbm.txt`` (Booster), ``models/logistic.pkl``, and
   ``models/calibrator.pkl`` (isotonic) and applying them to the
   ``split_train_val_test`` windows. All fitting happened in Task 4.
3. **Per-split base rates ride with every metrics table** (regime shift:
   train 24.98% / val 43.64% / test 13.49%) — PR-AUC is never compared
   across splits as if base rates matched (``notes`` in the output).
4. **Val-calibrated ECE is in-sample** (the isotonic calibrator was fit
   on val 2025-H1) — flagged in ``notes``, never presented as
   calibration quality; the test-split ECE is the honest number.

Persistence reuses ``src.train.persistence_scores`` (forecast for hour h =
congestion observed AT h = the label of the row whose ``label_hour == h``);
NaN scores (unobserved hours) are dropped for scoring, matching Task 4's
``n_scored`` semantics.

ECE reuses ``src.train.expected_calibration_error`` — 10 equal-width bins,
ECE = sum_b (n_b / n) * |mean_prob_b - frac_positive_b|, empty bins skipped.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.train import (
    FEATURES,
    LOGISTIC_FEATURES,
    expected_calibration_error,
    persistence_scores,
    split_train_val_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
METRICS_FINAL_PATH = REPORTS_DIR / "metrics_final.json"


def _repo_path(path: Path) -> str:
    """Return a stable repository-relative path for saved metadata."""
    return path.relative_to(PROJECT_ROOT).as_posix()


COMPARISON_FIGURE_PATH = FIGURES_DIR / "comparison_table.png"

# Exact key set required by the brief's interface.
METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "f1@best-threshold-on-val",
    "brier",
    "ece",
)

# Model rows in the comparison, in report order.
MODEL_KEYS = (
    "persistence",
    "logistic",
    "lightgbm_raw",
    "lightgbm_calibrated",
)

_NOTES = [
    (
        "Base rates differ sharply across splits (train 0.2498 / val 0.4364 / "
        "test 0.1349): PR-AUC must not be compared across splits as if base "
        "rates matched."
    ),
    (
        "f1@best-threshold-on-val: threshold maximizing F1 on val 2025-H1 "
        "predictions only, applied unchanged to test; test labels never "
        "influence threshold selection."
    ),
    (
        "val_2025_h1 lightgbm_calibrated ECE is in-sample (the isotonic "
        "calibrator was fit on val predictions) — not calibration quality; "
        "the test 2025-H2 ECE is the honest number."
    ),
]

_F1_COLUMN_LABEL = "F1@val-thr"


def best_f1_threshold(y_true, y_prob) -> float:
    """F1-maximizing threshold over the scores present in ``(y_true, y_prob)``.

    Candidates are the unique finite scores; a row predicts positive when
    ``p >= threshold``. Ties keep the smallest threshold (first max over the
    ascending candidate list). Callers scoring the test split must pass the
    threshold selected on val instead of calling this on test labels.
    """
    y, p = _finite(y_true, y_prob)
    candidates = np.unique(p)
    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        f1 = f1_score(y, p >= t, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def classification_metrics(y_true, y_prob, *, threshold: float | None = None) -> dict:
    """roc_auc / pr_auc / f1@best-threshold-on-val / brier / ece.

    Exactly the five brief-specified keys, all Python floats, scored on
    rows where ``y_prob`` is finite (persistence NaNs dropped).

    ``threshold=None`` selects the F1-maximizing threshold from the arrays
    passed in — correct for the val split, where the threshold is by
    definition chosen. For the test split pass the val-selected threshold
    (keyword-only) so test labels never influence any choice.
    """
    y, p = _finite(y_true, y_prob)
    if threshold is None:
        threshold = best_f1_threshold(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1@best-threshold-on-val": float(
            f1_score(y, p >= threshold, zero_division=0)
        ),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
    }


def _finite(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    return y[mask], p[mask]


def render_comparison(metrics: dict, out_path: Path | None = None) -> Path:
    """Render the test-set comparison table (persistence vs logistic vs
    LightGBM) to a PNG and return its path.

    ``metrics`` is the ``evaluate_splits`` structure: rows come from
    ``metrics["test_2025_h2"]`` (model blocks with the five metric keys,
    plus ``n`` and ``base_rate``, both printed alongside the table).
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    test = metrics["test_2025_h2"]
    models = [m for m in MODEL_KEYS if m in test]
    if out_path is None:
        out_path = COMPARISON_FIGURE_PATH

    headers = [
        "model",
        "ROC-AUC",
        "PR-AUC",
        _F1_COLUMN_LABEL,
        "Brier",
        "ECE",
    ]
    row_labels = {
        "persistence": "Persistence",
        "logistic": "Logistic",
        "lightgbm_raw": "LightGBM (raw)",
        "lightgbm_calibrated": "LightGBM (isotonic)",
    }
    cell_text = [
        [row_labels[m]] + [f"{test[m][k]:.4f}" for k in METRIC_KEYS]
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(models) + 1.4))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.26, 0.14, 0.14, 0.16, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold")
        elif _col == 0:
            # Model names read left-aligned inside their wider first column.
            cell._text.set_x(0.06)
            cell._text.set_ha("left")

    fig.suptitle(
        f"Model comparison — test 2025-H2 (n={test['n']:,}, "
        f"base rate {test['base_rate']:.1%})",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.03,
        f"F1 at threshold chosen on val 2025-H1 only · val base rate "
        f"{metrics['val_2025_h1']['base_rate']:.1%} · "
        f"PR-AUC not comparable across splits",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_splits(df: pd.DataFrame) -> dict:
    """Score every model on val 2025-H1 and test 2025-H2. No fitting.

    Loads the Task 4 artifacts, recomputes prediction arrays on the
    ``split_train_val_test`` windows (persistence travels with each row
    via ``persistence_scores`` before the split, exactly as in
    ``train_all``), selects each model's F1 threshold on val only, and
    returns the metrics structure written to ``reports/metrics_final.json``.
    """
    df = df.copy()
    df["_persistence"] = persistence_scores(df)
    train, val, test = split_train_val_test(df)

    logistic = joblib.load(MODELS_DIR / "logistic.pkl")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))
    calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")

    def predict_block(part: pd.DataFrame) -> dict:
        raw = booster.predict(part[FEATURES])
        return {
            "persistence": part["_persistence"].to_numpy(dtype=float),
            "logistic": logistic.predict_proba(part[LOGISTIC_FEATURES])[:, 1],
            "lightgbm_raw": raw,
            "lightgbm_calibrated": calibrator.transform(raw),
        }

    val_preds = predict_block(val)
    test_preds = predict_block(test)
    val_y = val["label"].to_numpy()
    test_y = test["label"].to_numpy()

    # Binding rule: thresholds come from VAL predictions only.
    thresholds = {
        m: best_f1_threshold(val_y, val_preds[m]) for m in MODEL_KEYS
    }

    def score_block(y, preds: dict) -> dict:
        return {
            m: classification_metrics(y, preds[m], threshold=thresholds[m])
            for m in MODEL_KEYS
        }

    splits = {
        "train": {
            "n": len(train),
            "base_rate": float(train["label"].mean()),
        },
        "val_2025_h1": {
            "n": len(val),
            "base_rate": float(val_y.mean()),
        },
        "test_2025_h2": {
            "n": len(test),
            "base_rate": float(test_y.mean()),
        },
    }
    return {
        "splits": splits,
        "threshold_selection": {
            "rule": (
                "F1-maximizing threshold selected on val 2025-H1 predictions "
                "only; applied unchanged to test 2025-H2"
            ),
            "per_model": thresholds,
        },
        "val_2025_h1": {"n": len(val), "base_rate": float(val_y.mean()),
                        **score_block(val_y, val_preds)},
        "test_2025_h2": {"n": len(test), "base_rate": float(test_y.mean()),
                         **score_block(test_y, test_preds)},
        "notes": list(_NOTES),
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    metrics = evaluate_splits(df)
    figure_path = render_comparison(metrics)
    metrics["sources"] = {
        "data": _repo_path(DATA_PATH),
        "task4_metrics_json": _repo_path(METRICS_PATH),
        "lightgbm": _repo_path(MODELS_DIR / "lightgbm.txt"),
        "logistic": _repo_path(MODELS_DIR / "logistic.pkl"),
        "calibrator": _repo_path(MODELS_DIR / "calibrator.pkl"),
        "figure": _repo_path(figure_path),
    }

    METRICS_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_FINAL_PATH.open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=False)

    test = metrics["test_2025_h2"]
    val = metrics["val_2025_h1"]
    print(
        "per-split base rates: train "
        f"{metrics['splits']['train']['base_rate']:.4f} / val "
        f"{val['base_rate']:.4f} / test {test['base_rate']:.4f} "
        "(PR-AUC not comparable across splits)"
    )
    print(
        "val 2025-H1 F1 thresholds: "
        + ", ".join(
            f"{m}={t:.4f}"
            for m, t in metrics["threshold_selection"]["per_model"].items()
        )
    )
    print(
        f"test 2025-H2 (n={test['n']:,}, base rate {test['base_rate']:.4f}):"
    )
    for model in MODEL_KEYS:
        b = test[model]
        print(
            f"  {model:>20}: roc_auc={b['roc_auc']:.4f} pr_auc={b['pr_auc']:.4f} "
            f"f1@val-thr={b['f1@best-threshold-on-val']:.4f} "
            f"brier={b['brier']:.4f} ece={b['ece']:.4f}"
        )
    print(f"saved: {METRICS_FINAL_PATH}")
    print(f"saved: {figure_path}")


if __name__ == "__main__":
    main()
