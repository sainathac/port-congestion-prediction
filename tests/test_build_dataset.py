"""Tests for src/build_dataset.py (Task 3): port-hour table with t+6 labels.

Covers Review Focus items 1-4: label leakage via rolling/lag features,
unsorted output, null labels, and baseline fit-period contamination.
All fixtures are synthetic vessel rows — the 575 MB raw parquet is never
read in unit tests.

Fixture layout (unperturbed): 240 contiguous hours from 2024-06-01 00:00
(a Saturday), one row per hour — normally one distinct vessel visit per
hour, except hours 170..190 which share a single multi-hour visit
(mmsi=2000, visit_id=42, visit_end=190) so visit-level leakage can be
tested (test_delay_dev_samedow_completed_visits_only). Per-hour
aggregates are hand-computable:

    i = hour offset; hour = 2024-06-01 00:00 + i h
    vessel_count      = 1
    waiting_count     = 1 if i even else 0
    queue_mean_hours  = 5.0 + i
    delay_p50         = 100.0 + 10*i
    ship_density_mean = 50 + i
    arrivals          = 1
    port_throughput   = i % 8
    avg_port_speed    = 1.0 + 0.01*i
    wind_speed_10m    = 5.0 + i
    wind_gusts_10m    = 10.0 + 2*i
    precipitation     = 1.5 if i % 3 == 0 else 0.0
    wave_height       = 1.0 + 0.05*i
    imputed_fraction  = 1.0 if i % 4 == 0 else 0.0

Label ground truth: with FIT_BASELINE literals, waiting (<=1) and queue
(<=244) stay below their baselines so those index components are 0, and
index = 0.4 * clip((delay_p50 - 235.108333)/235.108333, 0, 5)/5.
index(t+6) >= CONGESTION_THRESHOLD (0.016981388565773952) holds iff
delay_p50(t+6) >= 290 iff i+6 >= 19 iff row index i >= 13.
So labels for the 234 kept rows are [0]*13 + [1]*221 — these exact
values pin that build() uses FIT_BASELINE + CONGESTION_THRESHOLD.
"""

import math

import pandas as pd
import pytest

from src.build_dataset import FEATURE_COLS, RAW_COLUMNS, build
from src.eda_label import (
    CONGESTION_THRESHOLD,
    FIT_BASELINE,
    FIT_CUTOFF,
    fit_baseline,
)

HORIZON = 6
N_HOURS = 240
START = pd.Timestamp("2024-06-01 00:00")
KEPT = N_HOURS - HORIZON  # 234 rows: tail shift(-6) has no label


def _write_raw(path, perturb_from=None, delay_overrides=None) -> None:
    """Write a synthetic raw-AIS parquet with all RAW_COLUMNS.

    perturb_from: if set, rows i >= perturb_from get different delay,
    wind, waiting flag, draft flag, throughput, speed and density so a
    future-data leak into earlier rows becomes visible.
    delay_overrides: optional {hour index -> new delay_minutes}, applied
    last — used to alter a visit's delay stamp on all of its rows.
    """
    rows = []
    for i in range(N_HOURS):
        ts = START + pd.Timedelta(hours=i)
        if perturb_from is not None and i >= perturb_from:
            delay, zone = 50.0, 9.0 + i  # delay 50 -> future index 0 -> label flips
            wind = 100.0 + i
            draft = 0 if i % 4 == 0 else 1
            waiting = i % 2 != 0
            throughput, density = (i + 3) % 8, 60 + i
            speed = 2.0 + 0.01 * i
        else:
            delay, zone = 100.0 + 10.0 * i, 5.0 + i
            wind = 5.0 + i
            draft = 1 if i % 4 == 0 else 0
            waiting = i % 2 == 0
            throughput, density = i % 8, 50 + i
            speed = 1.0 + 0.01 * i
        if delay_overrides is not None and i in delay_overrides:
            delay = float(delay_overrides[i])
        if 170 <= i <= 190:
            mmsi, visit_id = 2000, 42  # multi-hour visit, visit_end=190 > t0=179
        else:
            mmsi, visit_id = 1000 + i, 1
        rows.append(
            {
                "hour_key": ts,
                "mmsi": mmsi,
                "visit_id": visit_id,
                "is_in_waiting_area": waiting,
                "time_in_zone_hours": zone,
                "delay_minutes": delay,
                "ship_density": density,
                "port_throughput": throughput,
                "avg_port_speed": speed,
                "wind_speed_10m": wind,
                "wind_gusts_10m": 10.0 + 2 * i,
                "precipitation": 1.5 if i % 3 == 0 else 0.0,
                "wave_height": 1.0 + 0.05 * i,
                "draft_imputed_flag": draft,
                "length_imputed_flag": 0,
                "width_imputed_flag": 0,
                "sog_imputed_flag": 0,
                "cog_imputed_flag": 0,
            }
        )
    df = pd.DataFrame(rows)
    assert list(df.columns) == RAW_COLUMNS
    df.to_parquet(path, index=False)


def _build(tmp_path, name="out.parquet", perturb_from=None, delay_overrides=None):
    raw = tmp_path / f"{name}.raw.parquet"
    _write_raw(raw, perturb_from=perturb_from, delay_overrides=delay_overrides)
    out = tmp_path / name
    df = build(raw, out, horizon_hours=HORIZON)
    return df, out


def test_features_precede_label(tmp_path):
    """label at t+6; every feature window ends at or before t."""
    df, _ = _build(tmp_path)
    assert len(df) == KEPT

    # label_hour is exactly t+6 for every row
    assert (df["label_hour"] == df["hour_key"] + pd.Timedelta(hours=HORIZON)).all()
    # hand-computed labels from FIT_BASELINE + CONGESTION_THRESHOLD (see module docstring)
    assert df["label"].tolist() == [0] * 13 + [1] * (KEPT - 13)
    assert df["label"].dtype == "int8"

    # lags: lag-k at row t is the value at t-k (exact fixture values)
    assert df.loc[5, "queue_mean_hours_lag1"] == pytest.approx(9.0)   # queue at i=4
    assert df.loc[11, "queue_mean_hours_lag6"] == pytest.approx(10.0)  # queue at i=5
    assert df.loc[7, "waiting_count_lag6"] == 0   # i=1 odd
    assert df.loc[8, "waiting_count_lag6"] == 1   # i=2 even
    assert df.loc[5, "port_throughput_lag1"] == pytest.approx(4.0)  # i=4: 4%8
    assert df.loc[5, "avg_port_speed_lag1"] == pytest.approx(1.04)
    assert df.loc[5, "ship_density_mean_lag2"] == pytest.approx(53.0)  # i=3: 50+3

    # rollings use shift(1) first: window at t covers [t-w, t-1], never t or later
    # (without shift(1), wind_roll3h at row 5 would be mean(8,9,10)=9.0)
    assert df.loc[5, "wind_speed_10m_roll3h"] == pytest.approx(8.0)   # mean(7,8,9) = i 2..4
    assert df.loc[5, "wind_speed_10m_roll12h"] == pytest.approx(7.0)  # partial: i 0..4
    assert df.loc[5, "arrivals_roll3h"] == pytest.approx(1.0)
    assert pd.isna(df.loc[0, "wind_speed_10m_roll3h"])  # no pre-history at series start

    # temporal features describe the predicted hour t+6 (calendar only, no observed data)
    assert df.loc[0, "hour_sin"] == pytest.approx(1.0)       # pred 06:00 -> sin(pi/2)
    assert df.loc[0, "hour_cos"] == pytest.approx(0.0)
    assert df.loc[0, "month_sin"] == pytest.approx(0.0)       # June -> sin(pi)
    assert df.loc[0, "is_gate_hours"] == 0                    # pred 06:00 not in 7..16
    assert df.loc[1, "is_gate_hours"] == 1                    # pred 07:00
    assert df.loc[42, "is_night_shift"] == 1                  # pred 00:00
    assert df.loc[0, "is_night_shift"] == 0                   # pred 06:00
    assert df.loc[0, "is_weekend"] == 1                       # 2024-06-01 is Saturday
    assert df.loc[48, "is_weekend"] == 0                      # pred 2024-06-03 Monday
    assert df.loc[0, "dow_sin"] == pytest.approx(math.sin(2 * math.pi * 6 / 7))

    # interactions: both inputs observed at t
    assert df.loc[4, "wind_x_queue"] == pytest.approx(9.0 * 9.0)    # wind i=4, queue i=4
    assert df.loc[5, "wind_x_queue"] == pytest.approx(10.0 * 10.0)
    assert df.loc[4, "gusts_x_waiting"] == pytest.approx(18.0 * 1.0)
    assert df.loc[5, "gusts_x_waiting"] == pytest.approx(0.0)

    # baseline deviation: past-only same-hour-of-week median of the as-of
    # completed-visit delay (visit-final stamps of completed visits only)
    assert df.loc[10, "delay_dev_samedow"] == 0.0  # first (Sat,10:00) -> no history -> 0
    # i=178 is the next Saturday 10:00; as_of(178) = median of completions
    # in (154h, 178h] = single-hour visits 155..169 -> 1720 (the fixture's
    # multi-hour visit 170..190 is still open); baseline = as_of(10) = 150
    assert df.loc[178, "delay_dev_samedow"] == pytest.approx(1720.0 - 150.0)

    # data quality
    assert df.loc[0, "imputed_fraction"] == 1.0
    assert df.loc[1, "imputed_fraction"] == 0.0

    # future-perturbation invariance: changing every row after t0 must not
    # change any feature at or before t0 (and labels only through t+6)
    t0 = START + pd.Timedelta(hours=119)
    df_b, _ = _build(tmp_path, name="perturbed.parquet", perturb_from=120)
    a = df.loc[df["hour_key"] <= t0].drop(columns=["label", "label_hour"])
    b = df_b.loc[df_b["hour_key"] <= t0].drop(columns=["label", "label_hour"])
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # labels sourced at or before t0 are unchanged; some later label flipped
    la = df.loc[df["label_hour"] <= t0, "label"]
    lb = df_b.loc[df_b["label_hour"] <= t0, "label"]
    assert (la.to_numpy() == lb.to_numpy()).all()
    assert (df["label"].to_numpy() != df_b["label"].to_numpy()).any()


def test_delay_dev_samedow_completed_visits_only(tmp_path):
    """R6: delay_dev_samedow must not embed a visit's outcome before it ends.

    delay_minutes is the dataset's target variable — the Kaggle data
    dictionary defines it as the visit's total dwell time, stamped as a
    whole-visit outcome onto every hourly row of that visit, so the stamp
    at hour t is not knowable while the visit is open.

    Part 1 (leak): alter the delay stamp on ALL rows of the fixture's
    multi-hour visit (hours 170..190, visit_end=190 > t0=179). Under a
    snapshot source this changes delay_dev at hours 170..179 (RED);
    under a completed-visit-only source it cannot — the visit has not
    completed by t0, so delay_dev at every hour <= t0 is byte-identical.

    Part 2 (signal): alter the stamp of a visit that COMPLETED before
    hour 196 (the single-hour visit at 193) — delay_dev at 196 MUST
    change, pinning that the feature still carries real delay signal
    rather than collapsing to a constant zero.

    Only delay_dev_samedow is asserted: the label may legitimately flip
    under these perturbations (delay is a label component at t+6).
    """
    t0 = START + pd.Timedelta(hours=179)

    # --- part 1: a future visit's stamp must not reach hours <= t0 ---
    df, _ = _build(tmp_path)
    df_b, _ = _build(
        tmp_path,
        name="future_visit.parquet",
        delay_overrides={i: 777.0 for i in range(170, 191)},
    )
    a = df.loc[df["hour_key"] <= t0, "delay_dev_samedow"].reset_index(drop=True)
    b = df_b.loc[df_b["hour_key"] <= t0, "delay_dev_samedow"].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b, check_exact=True)

    # --- part 2: a completed visit's stamp IS read ---
    df_c, _ = _build(
        tmp_path,
        name="done_visit.parquet",
        delay_overrides={193: 9999.0},
    )
    h196 = START + pd.Timedelta(hours=196)
    base = df.loc[df["hour_key"] == h196, "delay_dev_samedow"].iloc[0]
    alt = df_c.loc[df_c["hour_key"] == h196, "delay_dev_samedow"].iloc[0]
    # hand-derived (fixture, completed-visit-only source):
    #   as_of(196)  = median of completions in (172h, 196h] = the fixture's
    #                 multi-hour visit (v_end=190, d=2000) plus single-hour
    #                 visits 191..196 -> [2000..2060] -> 2030; past same-
    #                 (dow,hour) baseline = as_of(28) = 265 -> 1765.
    #   after stamping visit 193 with 9999: median([2000,2010,2020,9999,
    #                 2040,2050,2060]) = 2040 -> 1775.
    assert base == pytest.approx(1765.0)  # non-zero -> real delay signal
    assert alt == pytest.approx(1775.0)
    assert alt != base


def test_output_sorted_and_unique(tmp_path):
    df, out_path = _build(tmp_path)
    diffs = df["hour_key"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all()  # strictly increasing
    assert df["hour_key"].duplicated().sum() == 0
    assert len(df) == KEPT
    # index-free schema: hour_key + features + label + label_hour, nothing else
    assert df.columns[0] == "hour_key"
    assert df.columns[-2:].tolist() == ["label", "label_hour"]
    assert [c for c in df.columns if c not in ("hour_key", "label", "label_hour")] == FEATURE_COLS
    # saved parquet exists and round-trips
    assert out_path.exists()
    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == len(df)
    assert reloaded.columns.tolist() == df.columns.tolist()


def test_no_null_labels(tmp_path):
    df, _ = _build(tmp_path)
    assert df["label"].notna().all()
    assert df["label"].dtype == "int8"
    assert set(df["label"].unique()) == {0, 1}
    assert df["label_hour"].notna().all()
    # tail rows dropped by shift(-6): 240 observed - 6 = 234 labeled
    assert len(df) == KEPT
    assert df["hour_key"].max() == START + pd.Timedelta(hours=N_HOURS - 1 - HORIZON)


def test_baseline_fitted_on_train_only():
    """FIT_BASELINE / threshold are train-period constants (hour_key < 2025-01-01).

    The literals themselves are re-fit against real data in
    src.eda_label.main(); this pins (a) the cutoff, (b) the exact literals
    that build() imports, and (c) that fit_baseline() ignores post-cutoff
    rows entirely. build()'s labels in test_features_precede_label are
    hand-computed from these literals, pinning the import wiring too.
    """
    assert FIT_CUTOFF == pd.Timestamp("2025-01-01")
    assert FIT_BASELINE == {
        "delay_median": 235.10833333333335,
        "delay_iqr": 43.98333333333335,
        "waiting_p75": 15.0,
        "queue_p75": 342.20258160400556,
    }
    assert CONGESTION_THRESHOLD == 0.016981388565773952

    hourly = pd.DataFrame(
        {
            "hour_key": pd.to_datetime(
                [
                    "2024-12-31 22:00",  # pre-cutoff
                    "2024-12-31 23:00",  # pre-cutoff
                    "2025-01-01 00:00",  # post-cutoff
                    "2025-01-01 01:00",  # post-cutoff
                ]
            ),
            "delay_p50": [100.0, 200.0, 1000.0, 2000.0],
            "waiting_count": [4, 8, 100, 200],
            "queue_mean_hours": [10.0, 30.0, 500.0, 700.0],
        }
    )
    fit = fit_baseline(hourly)
    assert set(fit) == set(FIT_BASELINE)
    # pre-2025 only: delay [100, 200] -> median 150, iqr 175-125 = 50
    assert fit["delay_median"] == 150.0
    assert fit["delay_iqr"] == 50.0
    assert fit["waiting_p75"] == pytest.approx(7.0)   # q75 of [4, 8]
    assert fit["queue_p75"] == pytest.approx(25.0)    # q75 of [10, 30]
    # the filter matters: an all-rows fit would give 600, not 150
    assert fit["delay_median"] != hourly["delay_p50"].median()

    # perturbing post-cutoff rows cannot move the fit
    perturbed = hourly.copy()
    perturbed.loc[2:, "delay_p50"] = [9999.0, 8888.0]
    perturbed.loc[2:, "waiting_count"] = [7, 9]
    perturbed.loc[2:, "queue_mean_hours"] = [900.0, 950.0]
    assert fit_baseline(perturbed) == fit
