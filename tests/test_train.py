"""Tests for src/train.py (Task 4): splits, baselines, constraints, calibration.

Covers the brief's three named tests plus the controller's binding
additions: the label-horizon purge at every split/fold boundary, concrete
split membership on the real port-hour table, the logistic baseline, and
an exact ECE value. Synthetic fixtures are used for fold logic and
persistence; the real parquet (committed, 4.8 MB) is read only for the
split-membership test.

Persistence semantics under test: the forecast for hour h is the observed
congestion AT h (``label`` of the row whose ``label_hour == h``, i.e. the
row at h-6) — data at or before h, never the row's own label (which is
congestion at h+6) and never anything after h.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.train import (
    FEATURES,
    LOGISTIC_FEATURES,
    MONOTONIC_CONSTRAINTS,
    expanding_window_splits,
    expected_calibration_error,
    fit_lgbm,
    make_logistic_pipeline,
    persistence_scores,
    split_train_val_test,
)

_LGBM_TEST_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "n_estimators": 50,
}

PARQUET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "port_hours.parquet"
)

# The four feature families that carry a +1 monotonic constraint
# (brief: "+1 for queue/waiting/density/arrivals, 0 elsewhere").
EXPECTED_CONSTRAINED = {
    "waiting_count",
    *(f"waiting_count_lag{k}" for k in range(1, 7)),
    "queue_mean_hours",
    *(f"queue_mean_hours_lag{k}" for k in range(1, 7)),
    "ship_density_mean",
    *(f"ship_density_mean_lag{k}" for k in range(1, 7)),
    "arrivals",
    "arrivals_roll3h",
    "arrivals_roll12h",
    "arrivals_roll24h",
}


def _hourly_frame(n_hours: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Contiguous hourly frame with label_hour = hour_key + 6h."""
    hour_key = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "hour_key": hour_key,
            "label_hour": hour_key + pd.Timedelta(hours=6),
            "label": (np.arange(n_hours) % 7 == 0).astype("int8"),
        }
    )


# --- brief Step 1, test 1 ---------------------------------------------------


def test_monotonic_constraints_match_features():
    assert len(MONOTONIC_CONSTRAINTS) == len(FEATURES)
    assert set(MONOTONIC_CONSTRAINTS) <= {-1, 0, 1}  # LightGBM requirement
    constrained = {f for f, c in zip(FEATURES, MONOTONIC_CONSTRAINTS) if c == 1}
    assert constrained == EXPECTED_CONSTRAINED
    # Every +1 index maps to a queue/pressure column name.
    for f in constrained:
        assert f.startswith(
            ("waiting_count", "queue_mean_hours", "ship_density_mean", "arrivals")
        )
    # Explicit zeros: as-of delay deviation (controller: no constraint),
    # vessel count, weather x traffic interactions (constraining a product
    # is ill-defined), and everything not in the four families.
    for f in (
        "delay_dev_samedow",
        "vessel_count",
        "wind_x_queue",
        "gusts_x_waiting",
        "port_throughput_lag1",
        "wind_speed_10m",
        "imputed_fraction",
        "hour_sin",
    ):
        assert MONOTONIC_CONSTRAINTS[FEATURES.index(f)] == 0


# --- brief Step 1, test 2 ---------------------------------------------------


def test_expanding_window_no_time_overlap():
    # 150 days hourly; min_train 30d -> first valid at day 30, three 40d
    # valid windows (span 120d / 3), hour-aligned.
    df = _hourly_frame(150 * 24)
    folds = list(expanding_window_splits(df, n_splits=3, min_train_days=30))
    assert len(folds) == 3

    train_sizes = [len(tr) for tr, _ in folds]
    valid_sizes = [len(va) for _, va in folds]
    # Concrete counts: 30d/40d/40d windows minus the 6h label-horizon purge.
    assert train_sizes == [714, 1674, 2634]
    assert valid_sizes == [960, 960, 960]
    # Windows grow: each train is a strict time-prefix of the next.
    assert train_sizes == sorted(train_sizes)
    for (tr_a, va_a), (tr_b, _) in pairwise(folds):
        assert tr_a["hour_key"].max() < tr_b["hour_key"].max()
        assert tr_a["hour_key"].max() < va_a["hour_key"].min()
    # Brief condition: no hour_key overlap, valid windows ordered/disjoint.
    for tr, va in folds:
        assert tr["hour_key"].max() < va["hour_key"].min()
    for (_, va_a), (_, va_b) in pairwise(folds):
        assert va_a["hour_key"].max() < va_b["hour_key"].min()


# --- controller binding: purge the label horizon per fold -------------------


def test_expanding_window_purges_label_horizon():
    df = _hourly_frame(150 * 24)
    folds = list(expanding_window_splits(df, n_splits=3, min_train_days=30))
    for i, (tr, va) in enumerate(folds):
        # Binding condition, strictly stronger than hour_key overlap:
        # no training row's label may land on/after the valid window start.
        assert tr["label_hour"].max() < va["hour_key"].min(), f"fold {i}"
        # Exactly 6 contiguous hourly rows purged per fold (label = +6h).
        prefix = df[df["hour_key"] < va["hour_key"].min()]
        assert len(prefix) - len(tr) == 6, f"fold {i}"
        # The purged rows are precisely those whose label_hour reaches in.
        purged = prefix[prefix["label_hour"] >= va["hour_key"].min()]
        assert len(purged) == 6
        assert purged["label_hour"].min() == va["hour_key"].min()


# --- brief Step 1, test 3 ---------------------------------------------------


def _perfect_persistence_frame(n_hours: int = 240) -> pd.DataFrame:
    """Congestion periodic with period 6 -> label(t+6) == label(t) always.

    Congestion f(h) = 1 for h mod 6 in {0,1,2} else 0. Row at hour offset
    i carries label = f(i+6) = f(i); persistence at h reads f(h) via the
    row with label_hour == h (row h-6), so score == label exactly.
    """
    hour_key = pd.date_range("2024-01-01", periods=n_hours + 6, freq="h")
    offsets = np.arange(n_hours + 6)
    congestion = np.where(offsets % 6 < 3, 1, 0)
    # Labeled rows i = 0..n_hours-1 predict congestion at i+6.
    return pd.DataFrame(
        {
            "hour_key": hour_key[:n_hours],
            "label_hour": hour_key[6 : n_hours + 6],
            "label": congestion[6 : n_hours + 6].astype("int8"),
        }
    )


def test_persistence_uses_only_t():
    df = _perfect_persistence_frame()
    scores = persistence_scores(df)
    # First 6 rows have no observed congestion at their hour (no row with
    # label_hour == hour_key); everything after is scored.
    assert scores.isna().sum() == 6
    scored = scores.notna()
    y = df.loc[scored, "label"].to_numpy()
    p = scores[scored].to_numpy()
    # Perfect persistence -> AUC exactly 1.0 (both classes present).
    assert set(np.unique(y)) == {0, 1}
    assert roc_auc_score(y, p) == pytest.approx(1.0)

    # The score at t equals the label AT t (row with label_hour == t),
    # not the row's own label (which is congestion at t+6). Congestion
    # steps 0->1 at offset 120, so row i carries label = 1{i+6 >= 120} =
    # 1{i >= 114}; at row 116 the own label is 1 but congestion(116) = 0.
    step = pd.DataFrame(
        {
            "hour_key": pd.date_range("2024-01-01", periods=240, freq="h"),
            "label": (np.arange(240) >= 114).astype("int8"),
            "label_hour": pd.date_range("2024-01-01", periods=240, freq="h")
            + pd.Timedelta(hours=6),
        }
    )
    step_scores = persistence_scores(step)
    # At offset 116: congestion(116) = 0 (step at 120) but the row's own
    # label is congestion(122) = 1.
    assert step_scores.iloc[116] == 0.0
    assert step.loc[116, "label"] == 1

    # Uses no data after t: perturbing every label after t0 leaves the
    # score at t0 bit-identical (a forward-looking implementation would
    # move). t0 = the hour at row offset 100 (2024-01-05 04:00); its
    # persistence comes from row 94 (before t0).
    t0 = df.loc[100, "hour_key"]
    base = persistence_scores(df)
    perturbed = df.copy()
    after = perturbed["hour_key"] > t0
    perturbed.loc[after, "label"] = 1 - perturbed.loc[after, "label"]
    assert persistence_scores(perturbed).loc[100] == base.loc[100]
    # And the perturbation actually flipped later labels (the check bit).
    assert after.sum() == 139  # rows 101..239
    assert not perturbed.loc[after, "label"].equals(df.loc[after, "label"])


# --- controller: concrete split membership on the real table ----------------


def test_split_membership_purges_and_rates():
    df = pd.read_parquet(PARQUET_PATH)
    train, val, test = split_train_val_test(df)

    # Calendar partition by hour_key is disjoint and exhaustive ...
    raw_train = df.loc[df["hour_key"] < pd.Timestamp("2025-01-01")]
    raw_val = df.loc[
        (df["hour_key"] >= pd.Timestamp("2025-01-01"))
        & (df["hour_key"] < pd.Timestamp("2025-07-01"))
    ]
    raw_test = df.loc[df["hour_key"] >= pd.Timestamp("2025-07-01")]
    assert len(raw_train) + len(raw_val) + len(raw_test) == len(df) == 22_378
    # ... but the horizon purge intentionally drops the 6 train rows whose
    # labels land in 2025-H1: they are in no returned window at all.
    assert len(train) + len(val) + len(test) == 22_378 - 6
    assert train["hour_key"].max() < pd.Timestamp("2025-01-01")
    assert val["hour_key"].min() == pd.Timestamp("2025-01-01 00:00")
    assert val["hour_key"].max() < pd.Timestamp("2025-07-01")
    assert test["hour_key"].min() == pd.Timestamp("2025-07-02 00:00")

    # Concrete purge: the last 6 train hours (18:00-23:00 on 2024-12-31)
    # have label_hour inside 2025-H1; val's final labels all land before
    # 2025-07-01 (its last observed hour is 2025-06-30 17:00).
    assert len(df[df["hour_key"] < pd.Timestamp("2025-01-01")]) == 13_756
    assert len(train) == 13_750
    assert len(val) == 4_315
    assert len(test) == 4_307

    # Binding purge conditions across the boundaries.
    assert train["label_hour"].max() < val["hour_key"].min()
    assert val["label_hour"].max() < test["hour_key"].min()
    assert train["label_hour"].max() == pd.Timestamp("2024-12-31 23:00")
    assert val["label_hour"].max() == pd.Timestamp("2025-06-30 23:00")

    # Per-split base rates (measured regime shift; report must not treat
    # PR-AUCs across splits as base-rate-comparable).
    assert train["label"].mean() == pytest.approx(0.2498, abs=1e-3)
    assert val["label"].mean() == pytest.approx(0.4364, abs=1e-3)
    assert test["label"].mean() == pytest.approx(0.1349, abs=1e-3)


def test_cv_folds_stay_inside_train_period():
    """Optuna/CV never see the val or test window (controller resolution)."""
    df = pd.read_parquet(PARQUET_PATH)
    train, _, _ = split_train_val_test(df)
    folds = list(expanding_window_splits(train, n_splits=3, min_train_days=365))
    assert len(folds) == 3
    expected_train_n = [8_689, 10_400, 12_084]
    expected_valid_n = [1_711, 1_684, 1_660]
    for i, (tr, va) in enumerate(folds):
        assert len(tr) == expected_train_n[i]
        assert len(va) == expected_valid_n[i]
        prefix = train[train["hour_key"] < va["hour_key"].min()]
        assert len(prefix) - len(tr) == 6
        assert tr["label_hour"].max() < va["hour_key"].min()
        assert tr["hour_key"].max() < va["hour_key"].min()
        # Nothing at or after 2025 ever enters a fold.
        assert va["hour_key"].max() < pd.Timestamp("2025-01-01")
        assert tr["hour_key"].max() < pd.Timestamp("2025-01-01")
    # Windows grow across folds.
    assert expected_train_n == sorted(expected_train_n)


# --- baselines --------------------------------------------------------------


def test_logistic_baseline_separates_synthetic():
    rng = np.random.default_rng(0)
    n = 2_000
    waiting = rng.normal(0.0, 1.0, n)
    label = (waiting + rng.normal(0.0, 0.3, n) > 0).astype("int8")
    data = {f: rng.normal(0.0, 1.0, n) for f in LOGISTIC_FEATURES}
    data["waiting_count"] = waiting
    df = pd.DataFrame(data)
    df.insert(0, "hour_key", pd.date_range("2024-01-01", periods=n, freq="h"))
    df["label"] = label

    pipe = make_logistic_pipeline()
    pipe.fit(df[LOGISTIC_FEATURES], df["label"])
    proba = pipe.predict_proba(df[LOGISTIC_FEATURES])[:, 1]
    # The driver is waiting_count with positive weight; near-perfect AUC.
    assert roc_auc_score(df["label"], proba) >= 0.95
    assert pipe.named_steps["clf"].coef_[0][LOGISTIC_FEATURES.index("waiting_count")] > 0
    # Hand-picked subset only: no label columns, all real features.
    assert set(LOGISTIC_FEATURES) <= set(FEATURES)
    assert not ({"label", "label_hour", "hour_key"} & set(LOGISTIC_FEATURES))


# --- LightGBM (Step 4) ------------------------------------------------------


def _lgbm_fit_frame(n: int = 1_500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {f: rng.normal(0.0, 1.0, n) for f in FEATURES}
    waiting = rng.normal(0.0, 1.0, n)
    data["waiting_count"] = waiting
    frame = pd.DataFrame(data)
    frame["label"] = (waiting + rng.normal(0.0, 0.5, n) > 0).astype("int8")
    return frame


def test_lgbm_monotonic_constraint_holds_in_prediction():
    df = _lgbm_fit_frame()
    model = fit_lgbm(df[FEATURES], df["label"], dict(_LGBM_TEST_PARAMS))
    # Sweep the constrained driver alone across one fixed row: predictions
    # must never decrease (the +1 constraint's contract), and must
    # actually rise because the label was driven by waiting_count.
    grid = pd.concat([df[FEATURES].iloc[[0]]] * 21, ignore_index=True)
    grid["waiting_count"] = np.linspace(0.0, 40.0, 21)
    proba = model.predict_proba(grid)[:, 1]
    assert np.all(np.diff(proba) >= -1e-12)
    assert proba[-1] > proba[0]


def test_lgbm_monotonic_constraints_passed_to_model():
    """The constraint must actually reach the model (review fix).

    The empirical sweep above alone is non-discriminating: it passes
    against an unconstrained fit on a monotone-driven fixture. This
    asserts the parameter is wired through ``fit_lgbm`` — checked on the
    estimator params AND on the booster dump (what a shipped
    ``models/lightgbm.txt`` records), so a missing-requirement defect
    cannot go green again.
    """
    df = _lgbm_fit_frame()
    model = fit_lgbm(df[FEATURES], df["label"], dict(_LGBM_TEST_PARAMS))
    assert model.get_params()["monotone_constraints"] == list(MONOTONIC_CONSTRAINTS)
    dumped = model.booster_.dump_model()["monotone_constraints"]
    assert dumped == list(MONOTONIC_CONSTRAINTS)


def test_lgbm_scale_pos_weight_matches_ratio():
    df = _lgbm_fit_frame()
    model = fit_lgbm(df[FEATURES], df["label"], dict(_LGBM_TEST_PARAMS))
    expected = float((df["label"] == 0).sum() / (df["label"] == 1).sum())
    assert model.scale_pos_weight == pytest.approx(expected)
    assert model.objective == "binary"
    assert model.random_state == 42


# --- calibration metric -----------------------------------------------------


def test_expected_calibration_error_exact_value():
    # Hand-computed: 8 rows, 10 equal-width bins, one row per occupied bin
    # -> ECE = mean(|conf - acc|) = 1.8 / 8 = 0.225.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.85, 0.95])
    assert expected_calibration_error(y, p) == pytest.approx(0.225)
    # Perfect calibration -> 0.
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
