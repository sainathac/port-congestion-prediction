"""Port-hour feature table with a t+horizon congestion label (Task 3).

Consumes Task 2's ``compute_hour_aggregates`` / ``congestion_index`` /
``FIT_BASELINE`` / ``CONGESTION_THRESHOLD`` (rulings R1/R4 — the threshold
and baselines are imported, never re-derived here).

Leakage rule (plan): no feature may contain same-hour-or-later label
components. Concretely:

- lags use ``shift(k)`` so lag-k at row t is the value at t-k;
- rollings use ``shift(1)`` first, so a w-hour rolling at row t covers
  [t-w, t-1] — strictly before t;
- ``delay_dev_samedow`` sources an as-of COMPLETED-visit delay statistic
  (median of visit-final ``delay_minutes`` stamps for visits that
  completed in the past 24h) against a past-only same-hour-of-week
  expanding median (``shift(1)`` within the (dow, hour) group). The
  completed-visit-only source matters because ``delay_minutes`` is the
  dataset's target variable: a whole-visit outcome stamped on every
  hourly row of the visit, not knowable while the visit is open;
- temporal columns are deterministic calendar features of the *predicted*
  hour t+horizon (known at prediction time; no observed data), per spec §2;
- weather is observed at ≤ t only: hourly mean at t + past-only rollings.

The label is ``congestion_index(t+horizon) >= CONGESTION_THRESHOLD``
(int8). Rows whose calendar t+horizon hour is not observed (the
``shift(-h)`` tail, plus rows whose horizon lands in one of the raw
file's data-outage gaps) are dropped — otherwise the row-shift would
silently source the label from t+7..t+30.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.eda_label import (
    CONGESTION_THRESHOLD,
    FIT_BASELINE,
    compute_hour_aggregates,
    congestion_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "archive" / "ais_2023_2025_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"

# Column projection for the 575 MB raw file (18 of 53 columns).
RAW_COLUMNS = [
    "hour_key",
    "mmsi",
    "visit_id",
    "is_in_waiting_area",
    "time_in_zone_hours",
    "delay_minutes",
    "ship_density",
    "port_throughput",
    "avg_port_speed",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "wave_height",
    "draft_imputed_flag",
    "length_imputed_flag",
    "width_imputed_flag",
    "sog_imputed_flag",
    "cog_imputed_flag",
]
IMPUTED_FLAGS = [c for c in RAW_COLUMNS if c.endswith("_imputed_flag")]

LAG_STEPS = range(1, 7)
ROLLING_WINDOWS = (3, 12, 24)
_WEATHER_COLS = ["wind_speed_10m", "wind_gusts_10m", "precipitation", "wave_height"]
_ROLLING_COLS = ["arrivals", *_WEATHER_COLS]

# Queue group: at t and lags t-1..t-6 (spec §2).
_QUEUE_AT_T = ["waiting_count", "queue_mean_hours", "ship_density_mean"]
# Port state: lags t-1..t-6 only (spec §2 wording).
_PORT_LAG_COLS = ["port_throughput", "avg_port_speed"]
# Temporal: calendar of the predicted hour t+horizon (spec §2).
_TEMPORAL_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_night_shift",
    "is_gate_hours",
]

FEATURE_COLS: list[str] = []
for _col in _QUEUE_AT_T:
    FEATURE_COLS.append(_col)
    FEATURE_COLS.extend(f"{_col}_lag{k}" for k in LAG_STEPS)
for _col in _PORT_LAG_COLS:
    FEATURE_COLS.extend(f"{_col}_lag{k}" for k in LAG_STEPS)
FEATURE_COLS.extend(_TEMPORAL_COLS)
for _col in _WEATHER_COLS:
    FEATURE_COLS.append(_col)
    FEATURE_COLS.extend(f"{_col}_roll{w}h" for w in ROLLING_WINDOWS)
FEATURE_COLS.extend(["wind_x_queue", "gusts_x_waiting"])
FEATURE_COLS.extend(["vessel_count", "arrivals"])
FEATURE_COLS.extend(f"arrivals_roll{w}h" for w in ROLLING_WINDOWS)
FEATURE_COLS.extend(["delay_dev_samedow", "imputed_fraction"])

_NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]


def build(raw_path: Path, out_path: Path, horizon_hours: int = 6) -> pd.DataFrame:
    """Aggregate raw AIS rows into a leakage-safe port-hour feature table.

    Reads ``raw_path`` with pyarrow column projection, aggregates to one
    row per observed hour, engineers features using only data at or before
    each row's ``hour_key``, labels congestion at ``hour_key +
    horizon_hours``, drops rows without an observed horizon hour, and
    saves the table to ``out_path`` (index-free parquet). Returns the
    same DataFrame.
    """
    df = pq.read_table(raw_path, columns=RAW_COLUMNS).to_pandas()
    df["imputed_any"] = (df[IMPUTED_FLAGS] == 1).any(axis=1)

    hourly = compute_hour_aggregates(df)
    extras = (
        df.groupby("hour_key", sort=True)
        .agg(
            port_throughput=("port_throughput", "mean"),
            avg_port_speed=("avg_port_speed", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            wind_gusts_10m=("wind_gusts_10m", "mean"),
            precipitation=("precipitation", "mean"),
            wave_height=("wave_height", "mean"),
            imputed_fraction=("imputed_any", "mean"),
        )
        .reset_index()
    )
    hourly = hourly.merge(extras, on="hour_key", how="left")
    hourly = hourly.sort_values("hour_key").reset_index(drop=True)

    # delay_dev_samedow (ruling R6): as-of COMPLETED-visit delay vs a
    # past-only same-hour-of-week baseline of that as-of series.
    #
    # WHY completed-visit-only: delay_minutes is the dataset's TARGET
    # variable — the Kaggle data dictionary defines it as "total dwell
    # time (waiting + transit) inside the port zone for an isolated port
    # visit", and each hourly row of a visit carries that visit-final
    # stamp (measured: 99.5% of non-null delay rows occur while the
    # visit is still open). The stamp at hour t is therefore NOT
    # knowable at t; sourcing the feature from the snapshot would embed
    # post-t visit results. Instead:
    #   per visit (mmsi, visit_id): v_end = max(hour_key),
    #     d = last non-null delay_minutes (the visit-final stamp);
    #   as_of(h) = median of d over visits with v_end in (h-24h, h],
    #     NaN when fewer than 3 completions in that window;
    #   delay_dev_samedow(h) = as_of(h) - past-only same-(dow, hour)
    #     expanding median of the as-of series, fillna(0.0) = neutral.
    visit_rows = df[["mmsi", "visit_id", "hour_key", "delay_minutes"]].sort_values(
        "hour_key", kind="stable"
    )
    visits = (
        visit_rows.groupby(["mmsi", "visit_id"], sort=False)
        .agg(v_end=("hour_key", "max"), d=("delay_minutes", "last"))
        .reset_index()
    )
    comp = visits.dropna(subset=["d"])
    # Visit c is knowable at grid hour h iff v_end <= h < v_end + 24h
    # (equivalently v_end in (h-24h, h]), so expand each completion onto
    # the grid hours it covers and take the pooled per-hour median.
    grid = hourly["hour_key"].to_numpy()
    v_end = comp["v_end"].to_numpy()
    v_end_stop = (comp["v_end"] + pd.Timedelta(hours=24)).to_numpy()
    lo = np.searchsorted(grid, v_end, side="left")
    hi = np.searchsorted(grid, v_end_stop, side="left")
    reps = hi - lo  # grid hours covered by each completion (0..25)
    tgt = np.repeat(lo, reps) + (
        np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
    )
    pairs = pd.DataFrame(
        {"tgt": tgt, "d": np.repeat(comp["d"].to_numpy(dtype=float), reps)}
    )
    stat = pairs.groupby("tgt", sort=True)["d"].agg(as_of="median", n="size")
    as_of = stat["as_of"].where(stat["n"] >= 3)  # <3 completions -> NaN
    hourly["as_of_delay"] = as_of.reindex(np.arange(len(grid))).to_numpy()

    hourly["_dow"] = hourly["hour_key"].dt.day_of_week + 1  # raw convention: 1=Mon..7=Sun
    hourly["_hr"] = hourly["hour_key"].dt.hour
    past_median = hourly.groupby(["_dow", "_hr"], sort=False)["as_of_delay"].transform(
        lambda s: s.shift(1).expanding().median()
    )
    hourly["delay_dev_samedow"] = (hourly["as_of_delay"] - past_median).fillna(0.0)
    hourly = hourly.drop(columns=["_dow", "_hr"])

    # Lags: lag-k at row t is the value at t-k.
    for _col in [*_QUEUE_AT_T, *_PORT_LAG_COLS]:
        for k in LAG_STEPS:
            hourly[f"{_col}_lag{k}"] = hourly[_col].shift(k)

    # Rollings: shift(1) first so the w-hour window at row t covers
    # [t-w, t-1] — never t or anything after it (plan mandate).
    for _col in _ROLLING_COLS:
        past = hourly[_col].shift(1)
        for w in ROLLING_WINDOWS:
            hourly[f"{_col}_roll{w}h"] = past.rolling(w, min_periods=1).mean()

    # Temporal: calendar features of the predicted hour t+horizon (spec §2).
    # Pure functions of (hour_key, horizon) — deterministic, known at t.
    pred = hourly["hour_key"] + pd.Timedelta(hours=horizon_hours)
    pred_hour = pred.dt.hour
    pred_dow = pred.dt.day_of_week + 1  # 1=Mon..7=Sun
    pred_month = pred.dt.month
    hourly["hour_sin"] = np.sin(2 * np.pi * pred_hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * pred_hour / 24)
    hourly["dow_sin"] = np.sin(2 * np.pi * pred_dow / 7)
    hourly["dow_cos"] = np.cos(2 * np.pi * pred_dow / 7)
    hourly["month_sin"] = np.sin(2 * np.pi * pred_month / 12)
    hourly["month_cos"] = np.cos(2 * np.pi * pred_month / 12)
    hourly["is_weekend"] = (pred_dow >= 6).astype("int8")
    hourly["is_night_shift"] = pred_hour.isin(_NIGHT_HOURS).astype("int8")
    hourly["is_gate_hours"] = pred_hour.between(7, 16).astype("int8")

    # Interactions: both inputs observed at t (weather x traffic).
    hourly["wind_x_queue"] = hourly["wind_speed_10m"] * hourly["queue_mean_hours"]
    hourly["gusts_x_waiting"] = hourly["wind_gusts_10m"] * hourly["waiting_count"]

    # Label at t+horizon via row shift on the sorted hourly series:
    # fut[t] = index[t+horizon], NaN on the tail (and never compared there).
    index = congestion_index(hourly, FIT_BASELINE)
    fut = index.shift(-horizon_hours)
    label_hour = hourly["hour_key"] + pd.Timedelta(hours=horizon_hours)
    src_hour = hourly["hour_key"].shift(-horizon_hours)
    has_src = src_hour.notna()
    keep = has_src & (src_hour == label_hour)
    n_tail = int((~has_src).sum())
    n_gap = int((has_src & (src_hour != label_hour)).sum())

    out = hourly.loc[keep, ["hour_key", *FEATURE_COLS]].copy()
    out["label"] = (fut.loc[keep] >= CONGESTION_THRESHOLD).astype("int8")
    out["label_hour"] = label_hour.loc[keep]
    out = out.reset_index(drop=True)

    rate = float(out["label"].mean())
    print(f"features: {len(FEATURE_COLS)} columns")
    print(f"observed port-hours: {len(hourly):,}")
    print(
        f"dropped: {n_tail} tail (no t+{horizon_hours} row) + "
        f"{n_gap} (t+{horizon_hours} hour missing) -> {len(out):,} labeled rows"
    )
    print(
        f"positive rate: {rate:.4f} "
        f"(target 0.20-0.35, in_band={0.20 <= rate <= 0.35})"
    )
    splits = [
        ("train <2025-01-01", out["hour_key"] < pd.Timestamp("2025-01-01")),
        (
            "val 2025-H1",
            (out["hour_key"] >= pd.Timestamp("2025-01-01"))
            & (out["hour_key"] < pd.Timestamp("2025-07-01")),
        ),
        ("test 2025-H2", out["hour_key"] >= pd.Timestamp("2025-07-01")),
    ]
    for name, mask in splits:
        n = int(mask.sum())
        split_rate = float(out.loc[mask, "label"].mean()) if n else float("nan")
        print(f"  {name}: n={n:,} positive_rate={split_rate:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"saved: {out_path}")
    return out


def main() -> None:
    build(DATA_PATH, OUT_PATH)


if __name__ == "__main__":
    main()
