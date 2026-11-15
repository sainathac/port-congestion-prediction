"""Congestion index, hourly aggregates, and the label-threshold decision.

Label definition (documented in ``docs/label_decision.md``):

    index = 0.4*norm(waiting_count) + 0.4*norm(delay_p50) + 0.2*norm(queue_mean_hours)
    norm(x) = clip((x - baseline)/max(baseline, 1), 0, 5) / 5

``CONGESTION_THRESHOLD`` and ``FIT_BASELINE`` are the single source of truth
consumed by ``src/build_dataset.py`` (Task 3, ruling R1).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "archive" / "ais_2023_2025_clean.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Baselines are fit on hourly-aggregate rows with hour_key < 2025-01-01
# (train period only — no future leakage into the label definition).
FIT_CUTOFF = pd.Timestamp("2025-01-01")

# Literal values fitted by main() on the real data (see docs/label_decision.md).
FIT_BASELINE: dict[str, float] = {
    "delay_median": 235.10833333333335,
    "delay_iqr": 43.98333333333335,
    "waiting_p75": 15.0,
    "queue_p75": 342.20258160400556,
}

# Chosen label threshold: 75th percentile of the congestion index over
# pre-2025 hours (25.0% of fit-period hours positive; 26.3% overall).
# The fixed candidates 0.55/0.60/0.65/0.70 all yield 0% — see
# docs/label_decision.md for the measured rationale.
CONGESTION_THRESHOLD: float = 0.016981388565773952


def compute_hour_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vessel rows to one row per hour.

    Expects columns: hour_key, mmsi, visit_id, is_in_waiting_area,
    time_in_zone_hours, delay_minutes, ship_density.

    A visit is a (mmsi, visit_id) port call; vessel_count/waiting_count count
    visits, arrivals counts visits first observed in that hour. Null waiting
    flags count as not waiting. Delay percentiles skip nulls.
    """
    d = df[
        [
            "hour_key",
            "mmsi",
            "visit_id",
            "is_in_waiting_area",
            "time_in_zone_hours",
            "delay_minutes",
            "ship_density",
        ]
    ].copy()
    d["is_waiting"] = d["is_in_waiting_area"].fillna(False).astype(bool)
    d["first_hour"] = d.groupby(["mmsi", "visit_id"], sort=False)["hour_key"].transform(
        "min"
    )
    d["is_arrival"] = d["hour_key"] == d["first_hour"]

    visit_hour = (
        d.groupby(["hour_key", "mmsi", "visit_id"], sort=True)
        .agg(waiting=("is_waiting", "max"), arrival=("is_arrival", "first"))
        .reset_index()
    )

    hourly = (
        d.groupby("hour_key", sort=True)
        .agg(
            delay_p50=("delay_minutes", "median"),
            delay_p90=("delay_minutes", lambda s: s.quantile(0.9)),
            queue_mean_hours=("time_in_zone_hours", "mean"),
            ship_density_mean=("ship_density", "mean"),
        )
        .reset_index()
    )
    counts = (
        visit_hour.groupby("hour_key", sort=True)
        .agg(
            vessel_count=("mmsi", "size"),
            waiting_count=("waiting", "sum"),
            arrivals=("arrival", "sum"),
        )
        .reset_index()
    )

    out = hourly.merge(counts, on="hour_key", how="left")
    out = out[
        [
            "hour_key",
            "vessel_count",
            "waiting_count",
            "queue_mean_hours",
            "delay_p50",
            "delay_p90",
            "ship_density_mean",
            "arrivals",
        ]
    ]
    return out.sort_values("hour_key").reset_index(drop=True)


def fit_baseline(hourly: pd.DataFrame) -> dict[str, float]:
    """Fit index baselines on hourly rows with hour_key < 2025-01-01."""
    pre = hourly[hourly["hour_key"] < FIT_CUTOFF]
    delay = pre["delay_p50"]
    return {
        "delay_median": float(delay.median()),
        "delay_iqr": float(delay.quantile(0.75) - delay.quantile(0.25)),
        "waiting_p75": float(pre["waiting_count"].quantile(0.75)),
        "queue_p75": float(pre["queue_mean_hours"].quantile(0.75)),
    }


def _norm(x: pd.Series, baseline: float) -> pd.Series:
    scaled = (x.astype(float) - baseline) / max(baseline, 1.0)
    # A NaN component (an hour with no delay readings) contributes 0 severity
    # so labels never come out null (Task 3's test_no_null_labels).
    return scaled.clip(0.0, 5.0).fillna(0.0) / 5.0


def congestion_index(hourly: pd.DataFrame, baseline: dict[str, float]) -> pd.Series:
    """Weighted congestion index in [0, 1] per hourly row.

    baseline needs keys waiting_p75, delay_median, queue_p75 (FIT_BASELINE).
    Monotone non-decreasing in waiting_count, delay_p50, queue_mean_hours.
    """
    waiting = _norm(hourly["waiting_count"], baseline["waiting_p75"])
    delay = _norm(hourly["delay_p50"], baseline["delay_median"])
    queue = _norm(hourly["queue_mean_hours"], baseline["queue_p75"])
    index = 0.4 * waiting + 0.4 * delay + 0.2 * queue
    return pd.Series(index.to_numpy(), index=hourly.index, name="congestion_index")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq

    # Light-mode chart chrome (dataviz reference palette, light surface).
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "text.color": "#0b0b0b",
            "xtick.color": "#898781",
            "ytick.color": "#898781",
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    data_blue = "#2a78d6"
    accent_orange = "#eb6834"
    muted_gray = "#898781"
    band_blue = "#cde2fb"

    columns = [
        "hour_key",
        "mmsi",
        "visit_id",
        "is_in_waiting_area",
        "time_in_zone_hours",
        "delay_minutes",
        "ship_density",
        "vessel_area",
        "sog",
    ]
    print(f"loading {DATA_PATH.name} ({len(columns)} columns) ...")
    df = pq.read_table(DATA_PATH, columns=columns).to_pandas()
    hourly = compute_hour_aggregates(df)
    print(f"rows: {len(df):,} -> {len(hourly):,} port-hours")

    recomputed = fit_baseline(hourly)
    for key, value in recomputed.items():
        literal = FIT_BASELINE[key]
        match = abs(value - literal) <= 1e-9 * max(1.0, abs(literal))
        print(f"FIT_BASELINE[{key}] = {value!r} (literal {literal!r}) match={match}")
        assert match, f"FIT_BASELINE[{key}] literal is stale"

    index = congestion_index(hourly, FIT_BASELINE)
    pre_mask = hourly["hour_key"] < FIT_CUTOFF
    threshold = float(index[pre_mask].quantile(0.75))
    print(
        f"index: min={index.min():.6f} median={index.median():.6f} "
        f"p90={index.quantile(0.9):.6f} max={index.max():.6f}"
    )
    match = abs(threshold - CONGESTION_THRESHOLD) <= 1e-6
    print(
        f"CONGESTION_THRESHOLD literal={CONGESTION_THRESHOLD!r} "
        f"recomputed pre-2025 p75={threshold!r} match={match}"
    )
    assert match, "CONGESTION_THRESHOLD literal is stale"

    candidates = [0.55, 0.60, 0.65, 0.70]
    candidate_rates = {t: float((index >= t).mean()) for t in candidates}
    for t, rate in candidate_rates.items():
        print(f"positive rate at fixed threshold {t:.2f}: {rate:.6f}")

    chosen_rate = float((index >= CONGESTION_THRESHOLD).mean())
    pre_rate = float((index[pre_mask] >= CONGESTION_THRESHOLD).mean())
    rate_2025 = float((index[~pre_mask] >= CONGESTION_THRESHOLD).mean())
    print(
        f"chosen threshold {CONGESTION_THRESHOLD:.6f}: "
        f"pre-2025 rate={pre_rate:.4f} overall={chosen_rate:.4f} 2025={rate_2025:.4f}"
    )

    fence = FIT_BASELINE["delay_median"] + FIT_BASELINE["delay_iqr"]
    share_above_fence = float((hourly["delay_p50"] >= fence).mean())
    print(
        f"cross-check: delay_median+delay_iqr = {fence:.1f} min; "
        f"{share_above_fence:.4f} of hours have delay_p50 >= fence"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # (a) delay distribution + percentiles
    delay = df["delay_minutes"]
    n_null = int(delay.isna().sum())
    n_zero = int((delay == 0).sum())
    positive = delay[delay > 0]
    pct = {p: float(quant) for p, quant in
           ((p, positive.quantile(p / 100)) for p in (50, 75, 90, 95))}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        positive,
        bins=np.logspace(0, np.log10(float(positive.max()) + 1), 60),
        color=data_blue,
        edgecolor="#fcfcfb",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    top = ax.get_ylim()[1] * 1.15  # headroom so labels clear the tallest bar
    ax.set_ylim(0, top)
    for i, (p, value) in enumerate(pct.items()):
        ax.axvline(value, color=accent_orange, linewidth=1.2)
        y = top * (0.92 if i % 2 == 0 else 0.78)
        ax.text(
            value,
            y,
            f"p{p} = {value:,.0f} min",
            color="#52514e",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1.5},
        )
    ax.set_xlabel("delay_minutes (log scale)")
    ax.set_ylabel("vessel rows")
    ax.set_title(
        f"Delay distribution, delay > 0 ({len(positive):,} rows; "
        f"{n_zero:,} zero, {n_null:,} null omitted)",
        fontsize=11,
    )
    fig.tight_layout()
    path_a = FIGURES_DIR / "eda_delay_distribution.png"
    fig.savefig(path_a, dpi=150)
    plt.close(fig)
    print(f"wrote {path_a}")

    # (b) congestion-index histogram with candidate + chosen thresholds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        index.to_numpy(),
        bins=100,
        color=data_blue,
        edgecolor="#fcfcfb",
        linewidth=0.3,
        label="port-hours",
    )
    for i, t in enumerate(candidates):
        ax.axvline(
            t,
            color=muted_gray,
            linestyle="--",
            linewidth=1.2,
            label="fixed candidates 0.55-0.70" if i == 0 else None,
        )
    ax.axvline(
        CONGESTION_THRESHOLD,
        color=accent_orange,
        linewidth=2.0,
        label=f"chosen {CONGESTION_THRESHOLD:.4f}",
    )
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("congestion index")
    ax.set_ylabel("port-hours")
    ax.set_title(
        f"Congestion index over {len(hourly):,} port-hours "
        f"(max {index.max():.3f} — candidates unreachable)",
        fontsize=11,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 0.95), fontsize=8,
              framealpha=0.9)
    fig.tight_layout()
    path_b = FIGURES_DIR / "eda_index_histogram.png"
    fig.savefig(path_b, dpi=150)
    plt.close(fig)
    print(f"wrote {path_b}")

    # (c) positive rate at candidate thresholds vs 20-35% target
    labels_c = [f"{t:.2f}" for t in candidates] + [f"chosen\n{CONGESTION_THRESHOLD:.4f}"]
    rates_c = [candidate_rates[t] for t in candidates] + [chosen_rate]
    bar_colors = [muted_gray] * len(candidates) + [accent_orange]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhspan(0.20, 0.35, color=band_blue, zorder=0)
    ax.text(
        0.02,
        0.275,
        "target 20-35%",
        transform=ax.get_yaxis_transform(),
        color="#52514e",
        fontsize=8,
        va="center",
    )
    ax.bar(labels_c, rates_c, color=bar_colors, width=0.6, zorder=1)
    for i, rate in enumerate(rates_c):
        y = rate + 0.012 if rate > 0 else 0.012
        ax.text(i, y, f"{rate:.1%}", ha="center", fontsize=9, color="#0b0b0b")
    ax.set_ylabel("positive rate (share of port-hours)")
    ax.set_title("Positive rate by label threshold", fontsize=11)
    ax.set_ylim(0, 0.45)
    fig.tight_layout()
    path_c = FIGURES_DIR / "eda_positive_rates.png"
    fig.savefig(path_c, dpi=150)
    plt.close(fig)
    print(f"wrote {path_c}")


if __name__ == "__main__":
    main()
