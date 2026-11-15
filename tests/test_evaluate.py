"""Tests for src/evaluate.py (Task 5): metrics, threshold rule, comparison table.

Covers the brief's Step 1 sanity bounds (perfect predictor roc_auc 1.0,
random labels within [0.35, 0.65], exact key set, ECE in [0,1]) plus the
controller's binding additions:

- an exact ECE value pinning the 10-bin equal-width formula,
- the F1-threshold rule: the threshold maximizing F1 is selected on VAL
  predictions only and applied unchanged to TEST (test labels must never
  influence any choice),
- NaN persistence scores dropped (scored-row parity with Task 4),
- ``render_comparison`` writes a real PNG,
- an integration test recomputing predictions from ``models/*`` that
  reproduces ``reports/metrics.json`` test metrics and the val-selected
  thresholds, plus a label-flip probe proving thresholds ignore test labels.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluate import (
    best_f1_threshold,
    classification_metrics,
    evaluate_splits,
    render_comparison,
)
from src.train import (
    FEATURES,
    LOGISTIC_FEATURES,
    TEST_START,
    persistence_scores,
    split_train_val_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"
METRICS_JSON = PROJECT_ROOT / "reports" / "metrics.json"
MODELS_DIR = PROJECT_ROOT / "models"

EXPECTED_KEYS = {
    "roc_auc",
    "pr_auc",
    "f1@best-threshold-on-val",
    "brier",
    "ece",
}


# --- brief Step 1: sanity bounds --------------------------------------------


def test_classification_metrics_returns_exact_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    result = classification_metrics(y, p)
    assert set(result.keys()) == EXPECTED_KEYS
    assert len(result) == 5
    for key, value in result.items():
        assert isinstance(value, float), key


def test_perfect_predictor_roc_auc_one():
    y = np.array([0] * 10 + [1] * 10)
    p = y.astype(float)
    result = classification_metrics(y, p)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0)
    assert result["ece"] == pytest.approx(0.0)
    # Best-F1 threshold on perfect 0/1 scores is exactly 1.0 -> all right.
    assert result["f1@best-threshold-on-val"] == pytest.approx(1.0)


def test_random_labels_roc_auc_within_bounds():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 1_000)
    p = rng.random(1_000)
    result = classification_metrics(y, p)
    assert 0.35 <= result["roc_auc"] <= 0.65
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert 0.0 <= result["brier"] <= 1.0
    assert 0.0 <= result["f1@best-threshold-on-val"] <= 1.0


# --- ECE: pinned bin count + formula (exact expected value) ------------------


def test_ece_exact_value_equal_width_10_bins():
    """Hand-computed ECE with 10 equal-width bins.

    8 rows, one per occupied bin: bin b holds one row with confidence p_b
    and accuracy 1 for the four high rows, 0 for the four low rows.
    ECE = sum_b (n_b/n) * |mean_prob_b - frac_positive_b|
        = (0.05 + 0.15 + 0.25 + 0.45 + 0.45 + 0.25 + 0.15 + 0.05) / 8
        = 1.8 / 8 = 0.225
    Edges are np.linspace(0, 1, 11); np.digitize(p, edges[1:-1], right=False).
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.85, 0.95])
    result = classification_metrics(y, p)
    assert result["ece"] == pytest.approx(0.225)
    # Perfect calibration -> exactly 0.
    perfect = classification_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert perfect["ece"] == pytest.approx(0.0)


# --- F1 threshold rule: selected on val, applied to test --------------------


def test_f1_threshold_selected_on_val_applied_to_test():
    # Val: threshold 0.75 gives F1 = 1.0 (its own best) ...
    y_val = np.array([0, 1, 1, 0])
    p_val = np.array([0.30, 0.75, 0.90, 0.40])
    assert best_f1_threshold(y_val, p_val) == pytest.approx(0.75)
    val_result = classification_metrics(y_val, p_val)
    assert val_result["f1@best-threshold-on-val"] == pytest.approx(1.0)

    # ... test: its OWN best threshold would be 0.70 (F1 = 1.0), but the
    # binding rule forbids that — the val-chosen 0.75 is applied instead,
    # where p >= 0.75 predicts [0, 0, 1, 0] -> F1 = 2/3.
    y_test = np.array([0, 1, 1, 0])
    p_test = np.array([0.60, 0.70, 0.80, 0.40])
    assert best_f1_threshold(y_test, p_test) == pytest.approx(0.70)
    test_result = classification_metrics(y_test, p_test, threshold=0.75)
    assert test_result["f1@best-threshold-on-val"] == pytest.approx(2 / 3)
    # The hand-computed value at the val threshold (independent of any
    # threshold search on the test labels).
    manual = f1_score(y_test, p_test >= 0.75)
    assert test_result["f1@best-threshold-on-val"] == pytest.approx(manual)
    # The val-threshold result differs from what test-label selection
    # would have produced (1.0) — proving test labels did not choose.
    assert test_result["f1@best-threshold-on-val"] < 1.0


def test_nan_probs_dropped_for_scoring():
    # Persistence maps unobserved hours to NaN (Task 4 semantics): those
    # rows are dropped, the rest scored exactly.
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, np.nan, 0.9, 0.2])
    result = classification_metrics(y, p)
    # After dropping the NaN row: y=[0,1,0], p=[0.1,0.9,0.2] — the single
    # positive scores above both negatives -> roc 1.0; brier =
    # (0.01 + 0.01 + 0.04)/3 = 0.02.
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.02)
    # Best F1 at threshold 0.9 -> predicts [0,1,0] exactly.
    assert result["f1@best-threshold-on-val"] == pytest.approx(1.0)


# --- comparison figure ------------------------------------------------------


def test_render_comparison_writes_png(tmp_path):
    def block(roc, pr, f1, brier, ece):
        return {
            "roc_auc": roc,
            "pr_auc": pr,
            "f1@best-threshold-on-val": f1,
            "brier": brier,
            "ece": ece,
        }

    metrics = {
        "val_2025_h1": {"n": 100, "base_rate": 0.44},
        "test_2025_h2": {
            "n": 100,
            "base_rate": 0.13,
            "persistence": block(0.70, 0.31, 0.30, 0.14, 0.14),
            "logistic": block(0.73, 0.33, 0.32, 0.17, 0.24),
            "lightgbm_raw": block(0.75, 0.39, 0.36, 0.17, 0.23),
            "lightgbm_calibrated": block(0.74, 0.37, 0.35, 0.13, 0.16),
        }
    }
    out = render_comparison(metrics, out_path=tmp_path / "comparison_table.png")
    assert isinstance(out, Path)
    assert out.exists()
    payload = out.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 5_000


# --- integration: real artifacts, val thresholds, Task 4 parity -------------


def _independent_val_test_predictions(df: pd.DataFrame) -> tuple:
    """Recompute val/test predictions from models/* without evaluate.py."""
    df = df.copy()
    df["_persistence"] = persistence_scores(df)
    _, val, test = split_train_val_test(df)
    logistic = joblib.load(MODELS_DIR / "logistic.pkl")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))
    calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")

    def block(part: pd.DataFrame) -> dict:
        raw = booster.predict(part[FEATURES])
        return {
            "persistence": part["_persistence"].to_numpy(dtype=float),
            "logistic": logistic.predict_proba(part[LOGISTIC_FEATURES])[:, 1],
            "lightgbm_raw": raw,
            "lightgbm_calibrated": calibrator.transform(raw),
        }

    return val, test, block(val), block(test)


def test_evaluate_splits_val_threshold_rule_and_task4_parity():
    df = pd.read_parquet(PARQUET_PATH)
    result = evaluate_splits(df)

    # Per-split base rates alongside the metrics (regime shift).
    assert result["splits"]["train"]["base_rate"] == pytest.approx(0.2498, abs=1e-3)
    assert result["splits"]["val_2025_h1"]["base_rate"] == pytest.approx(0.4364, abs=1e-3)
    assert result["splits"]["test_2025_h2"]["base_rate"] == pytest.approx(0.1349, abs=1e-3)
    assert result["test_2025_h2"]["n"] == 4_307
    assert result["val_2025_h1"]["n"] == 4_315

    val, test, val_preds, test_preds = _independent_val_test_predictions(df)
    val_y = val["label"].to_numpy()
    test_y = test["label"].to_numpy()

    # Thresholds equal an independent best-F1 search on VAL predictions only.
    expected_thr = {
        m: best_f1_threshold(val_y, val_p) for m, val_p in val_preds.items()
    }
    for model, thr in expected_thr.items():
        got = result["threshold_selection"]["per_model"][model]
        assert got == pytest.approx(thr, abs=1e-12), model

    # Test F1 rows use the val threshold, recomputed by hand.
    for model, thr in expected_thr.items():
        manual = f1_score(test_y, test_preds[model] >= thr, zero_division=0)
        got = result["test_2025_h2"][model]["f1@best-threshold-on-val"]
        assert got == pytest.approx(manual, abs=1e-12), model

    # Parity with Task 4's metrics.json (same artifacts, same formulas).
    t4 = json.loads(METRICS_JSON.read_text())
    for model in ("persistence", "logistic", "lightgbm_raw", "lightgbm_calibrated"):
        for key in ("roc_auc", "pr_auc", "brier", "ece"):
            assert result["test_2025_h2"][model][key] == pytest.approx(
                t4["test_2025_h2"][model][key], abs=1e-9
            ), f"{model}.{key}"
            assert result["val_2025_h1"][model][key] == pytest.approx(
                t4["val_2025_h1"][model][key], abs=1e-9
            ), f"val {model}.{key}"

    # Presentation flags carried inside the final metrics.
    assert any("PR-AUC" in note for note in result["notes"])
    assert any("in-sample" in note for note in result["notes"])


def test_threshold_selection_ignores_test_labels():
    df = pd.read_parquet(PARQUET_PATH)
    base = evaluate_splits(df)

    flipped = df.copy()
    mask = flipped["hour_key"] >= TEST_START
    flipped.loc[mask, "label"] = (1 - flipped.loc[mask, "label"]).astype(
        flipped["label"].dtype
    )
    alt = evaluate_splits(flipped)

    # The flip really landed (test base rate inverted) ...
    assert alt["test_2025_h2"]["base_rate"] == pytest.approx(
        1.0 - base["test_2025_h2"]["base_rate"], abs=1e-9
    )
    # ... yet thresholds and every val metric are bit-identical: only val
    # predictions choose thresholds, and val labels never changed.
    assert alt["threshold_selection"]["per_model"] == pytest.approx(
        base["threshold_selection"]["per_model"], abs=1e-15
    )
    assert alt["val_2025_h1"] == base["val_2025_h1"]
    # Test metrics did react to the flipped labels (sanity, not leakage).
    assert (
        alt["test_2025_h2"]["logistic"]["roc_auc"]
        != base["test_2025_h2"]["logistic"]["roc_auc"]
    )
