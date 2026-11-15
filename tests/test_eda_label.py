"""Tests for hourly aggregates and the congestion index (Task 2)."""

import pandas as pd
import pytest

from src.eda_label import (
    CONGESTION_THRESHOLD,
    FIT_BASELINE,
    compute_hour_aggregates,
    congestion_index,
)


def _vessel_rows() -> pd.DataFrame:
    # Two vessels in the 10:00 hour (one waiting), one vessel at 11:00.
    return pd.DataFrame(
        {
            "hour_key": pd.to_datetime(
                ["2024-06-01 10:00", "2024-06-01 10:00", "2024-06-01 11:00"]
            ),
            "mmsi": [111, 222, 333],
            "visit_id": [1, 1, 1],
            "is_in_waiting_area": [True, False, False],
            "time_in_zone_hours": [12.0, 3.0, 1.0],
            "delay_minutes": [100.0, 50.0, 20.0],
            "ship_density": [40, 42, 44],
        }
    )


def test_compute_hour_aggregates_groups_by_hour():
    out = compute_hour_aggregates(_vessel_rows())

    assert list(out.columns) == [
        "hour_key",
        "vessel_count",
        "waiting_count",
        "queue_mean_hours",
        "delay_p50",
        "delay_p90",
        "ship_density_mean",
        "arrivals",
    ]
    assert len(out) == 2

    first = out.iloc[0]
    assert first["hour_key"] == pd.Timestamp("2024-06-01 10:00")
    # two vessels same hour, one in waiting area
    assert first["vessel_count"] == 2
    assert first["waiting_count"] == 1
    assert first["queue_mean_hours"] == pytest.approx(7.5)
    assert first["delay_p50"] == pytest.approx(75.0)
    assert first["delay_p90"] == pytest.approx(95.0)
    assert first["ship_density_mean"] == pytest.approx(41.0)
    # both visits are first observed in this hour
    assert first["arrivals"] == 2

    second = out.iloc[1]
    assert second["hour_key"] == pd.Timestamp("2024-06-01 11:00")
    assert second["vessel_count"] == 1
    assert second["waiting_count"] == 0
    assert second["arrivals"] == 1


def test_congestion_index_monotone_in_waiting_and_delay():
    baseline = {"waiting_p75": 5.0, "delay_median": 100.0, "queue_p75": 10.0}
    base = pd.DataFrame(
        {
            "hour_key": pd.to_datetime(["2024-06-01 10:00"]),
            "waiting_count": [10],
            "delay_p50": [100.0],
            "queue_mean_hours": [5.0],
        }
    )
    base_index = float(congestion_index(base, baseline).iloc[0])
    assert base_index == pytest.approx(0.08)  # 0.4 * ((10-5)/5/5)

    def _index_with(**overrides) -> float:
        frame = base.assign(**overrides)
        return float(congestion_index(frame, baseline).iloc[0])

    # raising waiting_count never lowers the index (and rises when unsaturated)
    assert _index_with(waiting_count=[20]) > base_index
    assert _index_with(waiting_count=[50]) >= base_index  # clipped at norm=1

    # raising delay_p50 never lowers the index
    assert _index_with(delay_p50=[300.0]) > base_index
    assert _index_with(delay_p50=[5000.0]) >= base_index  # clipped at norm=1

    # raising queue_mean_hours never lowers the index
    assert _index_with(queue_mean_hours=[12.0]) > base_index


def test_label_constants_are_emitted():
    """Ruling R1: threshold sits alongside FIT_BASELINE for Task 3 to import."""
    assert set(FIT_BASELINE) == {"delay_median", "delay_iqr", "waiting_p75", "queue_p75"}
    assert all(isinstance(v, float) for v in FIT_BASELINE.values())
    assert FIT_BASELINE["delay_median"] > 0
    assert FIT_BASELINE["delay_iqr"] > 0
    assert FIT_BASELINE["waiting_p75"] > 0
    assert FIT_BASELINE["queue_p75"] > 0
    assert isinstance(CONGESTION_THRESHOLD, float)
    assert 0.0 < CONGESTION_THRESHOLD < 1.0
