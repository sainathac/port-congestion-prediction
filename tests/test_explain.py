"""Tests for src/explain.py (Task 6): SHAP explain_hour, risk band, failures.

Covers the brief's Step 1 sanity contract (probability in [0,1], top_drivers
sorted by |SHAP| desc, narrative names the top driver) plus the controller's
binding additions:

- ``probability`` is the RAW model probability: sigmoid(expected_value +
  sum(SHAP)) must match ``booster.predict`` (the calibrator is never fed
  to SHAP),
- the risk band uses the val-selected F1 threshold for ``lightgbm_raw``
  from ``reports/metrics_final.json`` (pinned literal),
- the exact high-confidence-misclassification rule on the test split with
  pinned counts (1,246 = 1,039 FP + 207 FN) and the selected case-study
  hours,
- the CLI prints raw and calibrated probabilities, the calibrated decision flag, and SHAP drivers,
- figures and ``reports/explanations.md`` are actually written.
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap

from src.explain import (
    build_report,
    calibrated_probability,
    calibrated_threshold,
    explain_hour,
    high_confidence_errors,
    load_booster,
    main,
    plot_importance,
    plot_waterfall,
    risk_band,
    val_threshold,
)
from src.train import FEATURES, split_train_val_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FINAL = PROJECT_ROOT / "reports" / "metrics_final.json"

# Val-selected F1 threshold for lightgbm_raw (reports/metrics_final.json ->
# threshold_selection.per_model.lightgbm_raw), pinned verbatim.
EXPECTED_THR = 0.44174124132672526

CLI_HOUR = pd.Timestamp("2025-07-15 14:00")
CONGESTED_HOUR = pd.Timestamp("2025-07-10 23:00")  # label=1, model right, p~0.8978
CLEAR_HOUR = pd.Timestamp("2025-12-24 21:00")  # label=0, model right, p~0.0471
FP_HOUR = pd.Timestamp("2025-07-16 23:00")  # most confident FP (tie -> earliest)
FN_HOUR = pd.Timestamp("2025-12-19 04:00")  # most confident FN

EXPECTED_KEYS = {"probability", "top_drivers", "narrative"}


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    _, _, test = split_train_val_test(df)
    return test


@pytest.fixture(scope="module")
def booster() -> lgb.Booster:
    return load_booster()


def _row(df: pd.DataFrame, hour: pd.Timestamp) -> pd.DataFrame:
    rows = df.loc[df["hour_key"] == hour]
    assert len(rows) == 1, f"expected exactly one row for {hour}, got {len(rows)}"
    return rows[FEATURES]


# --- brief Step 1: explain_hour sanity contract --------------------------------


def test_explain_hour_returns_exact_keys_and_types(booster, test_df):
    result = explain_hour(booster, _row(test_df, CLI_HOUR))
    assert set(result.keys()) == EXPECTED_KEYS
    assert len(result) == 3
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["top_drivers"], list)
    assert len(result["top_drivers"]) == 10
    for name, value in result["top_drivers"]:
        assert isinstance(name, str)
        assert isinstance(value, float)
        assert name in FEATURES
    assert isinstance(result["narrative"], str)
    assert result["narrative"]


def test_top_drivers_sorted_by_abs_shap_desc(booster, test_df):
    row = _row(test_df, CONGESTED_HOUR)
    result = explain_hour(booster, row)
    drivers = result["top_drivers"]
    magnitudes = [abs(v) for _, v in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)

    # Independent oracle: TreeExplainer computed here, not via explain_hour.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="LightGBM binary classifier with TreeExplainer.*"
        )
        sv = np.asarray(shap.TreeExplainer(booster).shap_values(row)).reshape(-1)
    top_idx = np.argsort(-np.abs(sv))[:10]
    assert [n for n, _ in drivers] == [FEATURES[i] for i in top_idx]
    for (name, value), i in zip(drivers, top_idx):
        assert value == pytest.approx(float(sv[i]), abs=1e-9), name

    # Concrete pin: the congested example hour is queue-driven.
    assert drivers[0][0] == "waiting_count"
    assert drivers[0][1] == pytest.approx(0.3893, abs=1e-3)


def test_narrative_mentions_top_driver(booster, test_df):
    result = explain_hour(booster, _row(test_df, CLI_HOUR))
    top_name = result["top_drivers"][0][0]
    assert top_name in result["narrative"]
    # Concrete: the CLI hour is July -> month_cos dominates.
    assert top_name == "month_cos"


def test_probability_is_raw_model_probability_matching_predict(booster, test_df):
    # Binding resolution: SHAP explains the booster's raw output; probability
    # = sigmoid(expected_value + sum SHAP) must equal booster.predict.
    for hour, expected in (
        (CLI_HOUR, 0.6689429223295367),
        (CONGESTED_HOUR, 0.8977842175380909),
        (CLEAR_HOUR, 0.04706355422704792),
    ):
        row = _row(test_df, hour)
        result = explain_hour(booster, row)
        assert result["probability"] == pytest.approx(expected, abs=1e-9), hour
        assert result["probability"] == pytest.approx(
            float(booster.predict(row)[0]), abs=1e-9
        ), hour


# --- risk band: val threshold from metrics_final.json --------------------------


def test_val_threshold_is_pinned_lightgbm_raw_value():
    assert val_threshold() == EXPECTED_THR


def test_calibrated_threshold_is_loaded_from_final_metrics():
    assert calibrated_threshold() == pytest.approx(1 / 3)


def test_calibrator_returns_a_valid_event_probability():
    result = calibrated_probability(0.6689429223295367)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_risk_band_rule_uses_val_threshold():
    thr = val_threshold()
    assert risk_band(0.6689, thr) == "above"
    assert risk_band(0.05, thr) == "below"
    # Boundary is inclusive (p >= thr -> above).
    assert risk_band(thr, thr) == "above"
    assert risk_band(np.nextafter(thr, 0.0), thr) == "below"


def test_cli_prints_probability_drivers_band_and_rule(capsys):
    rc = main(["--hour", "2025-07-15T14:00"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0.6689" in out  # raw probability, 4dp
    assert "congestion probability at t+6 (isotonic-calibrated)" in out
    assert "decision flag (calibrated)" in out
    assert "0.3333" in out  # calibrated threshold
    assert "0.4417" in out  # separate raw-model analysis threshold
    assert "month_cos" in out  # top driver
    assert "rule" in out.lower()  # the rule is stated in the output
    assert "false positive" in out  # raw-threshold historical outcome


def test_cli_rejects_unknown_hour(capsys):
    rc = main(["--hour", "1999-01-01T00:00"])
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()


# --- failure analysis: exact rule + pinned counts ------------------------------


def test_high_confidence_errors_rule_and_pinned_counts(booster, test_df):
    thr = val_threshold()
    probs = booster.predict(test_df[FEATURES])
    errors = high_confidence_errors(test_df, probs, thr)

    # Pool = rows wrong at the val threshold whose wrong-class confidence
    # (p for FP, 1-p for FN) is >= thr. At thr < 0.5 the confidence leg is
    # implied, so the pool equals every misclassified test hour.
    y = test_df["label"].to_numpy()
    pred = probs >= thr
    wrong = pred != (y == 1)
    conf = np.where(pred, probs, 1.0 - probs)
    expected_mask = wrong & (conf >= thr)
    assert len(test_df) == 4_307
    assert test_df["label"].mean() == pytest.approx(0.1349, abs=1e-3)
    assert int(wrong.sum()) == 1_246
    assert int(((y == 0) & pred).sum()) == 1_039  # FP
    assert int(((y == 1) & ~pred).sum()) == 207  # FN
    assert len(errors) == 1_246
    assert set(errors["hour_key"]) == set(
        test_df.loc[expected_mask, "hour_key"]
    )
    assert (errors["error_type"] == "false_positive").sum() == 1_039
    assert (errors["error_type"] == "false_negative").sum() == 207
    assert (errors["wrong_class_confidence"] >= thr - 1e-12).all()

    # Selection: most confident FP / FN (confidence desc, ties -> earliest).
    fps = errors.loc[errors["error_type"] == "false_positive"]
    fns = errors.loc[errors["error_type"] == "false_negative"]
    assert fps.iloc[0]["hour_key"] == FP_HOUR
    assert fps.iloc[0]["probability"] == pytest.approx(0.8990606555314379, abs=1e-9)
    assert fns.iloc[0]["hour_key"] == FN_HOUR
    assert fns.iloc[0]["probability"] == pytest.approx(0.05017800507455917, abs=1e-9)
    # The FP tie at p~0.89906 resolves to the earlier hour (bit-identical
    # doubles, so exact equality is the right filter here).
    tied = fps.loc[fps["probability"] == 0.8990606555314379]
    assert len(tied) == 2
    assert tied.iloc[0]["hour_key"] == FP_HOUR
    assert tied.iloc[1]["hour_key"] == pd.Timestamp("2025-07-17 01:00")


# --- figures + explanations.md --------------------------------------------------


def test_plot_importance_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    importance = rng.random(20)
    names = [f"feature_{i}" for i in range(20)]
    out = plot_importance(
        importance,
        names,
        tmp_path / "shap_importance.png",
        title="Global feature importance",
        subtitle="test 2025-H2 (n=4,307)",
    )
    payload = out.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 5_000


def test_plot_waterfall_writes_png(tmp_path):
    n = 6
    out = plot_waterfall(
        hour_key=pd.Timestamp("2025-01-01 00:00"),
        feature_names=[f"feature_{i}" for i in range(n)],
        feature_values=np.linspace(0, 10, n),
        shap_values=np.array([0.4, -0.3, 0.2, -0.1, 0.05, -0.05]),
        base_value=-0.4,
        probability=0.7,
        label=1,
        out_path=tmp_path / "shap_waterfall.png",
    )
    payload = out.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 5_000


def test_build_report_writes_all_figures_and_markdown(tmp_path):
    fig_dir = tmp_path / "figures"
    out_path = tmp_path / "explanations.md"
    result = build_report(figure_dir=fig_dir, out_path=out_path)
    assert result == out_path

    expected_figures = [
        "shap_importance.png",
        "shap_waterfall_congested.png",
        "shap_waterfall_clear.png",
        "shap_waterfall_wrong.png",
        "shap_waterfall_wrong2.png",
    ]
    for name in expected_figures:
        payload = (fig_dir / name).read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n", name
        assert len(payload) > 5_000, name

    md = out_path.read_text()
    # The >=2 high-confidence wrong hours, each with its waterfall.
    assert "2025-07-16 23:00" in md
    assert "2025-12-19 04:00" in md
    assert "shap_waterfall_wrong.png" in md
    assert "shap_waterfall_wrong2.png" in md
    # Example hours + global figure referenced.
    assert "shap_waterfall_congested.png" in md
    assert "shap_waterfall_clear.png" in md
    assert "shap_importance.png" in md
    # Rule and numbers stated.
    assert "0.4417" in md
    assert "1,246" in md
    assert "1,039" in md
    assert "207" in md
    assert "confidence in the wrongly predicted class" in md
    assert "month_cos" in md
