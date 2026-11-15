# Congestion label decision (Task 2)

**Canonical citation for the report** (ruling R4). Every number below regenerates from
`uv run python -m src.eda_label` on `archive/ais_2023_2025_clean.parquet`
(21,678,617 vessel rows → 22,534 port-hours). Figures: `reports/figures/eda_{delay_distribution,index_histogram,positive_rates}.png`.

## 1. Congestion index formula

Per port-hour (one row from `compute_hour_aggregates`):

```
index = 0.4 * norm(waiting_count)
      + 0.4 * norm(delay_p50)
      + 0.2 * norm(queue_mean_hours)

norm(x) = clip((x - baseline) / max(baseline, 1), 0, 5) / 5
```

Result is in [0, 1]; monotone non-decreasing in each input (test-enforced).

**Rationale (spec leaves exact weights to the builder):**
- **0.4 / 0.4 / 0.2** — waiting-area occupancy and delay are the two signals the spec
  names first ("waiting-area occupancy vs. normal", "delay above the port's own
  historical baseline"); queue length is the third, corroborating signal, hence half
  weight. Equal top weights avoid letting any single noisy channel dominate.
- **baseline-relative scaling** — each component is measured against the port's own
  history, not an absolute unit, so components are commensurate before weighting.
- **clip at 5× baseline, then /5** — caps extreme hours (delay outliers reach 41,451 min
  in raw data) so one pathological hour cannot swamp the index; the /5 maps the clip
  range into [0, 1].
- **NaN component → 0 severity** — an hour with no delay readings contributes 0 to the
  index rather than null, so labels never come out null (Task 3's `test_no_null_labels`).

## 2. Baseline — `FIT_BASELINE` (fit on `hour_key < 2025-01-01`, train period only)

| key | value | what it baselines |
|---|---|---|
| `delay_median` | 235.10833333333335 | `delay_p50` (minutes) |
| `delay_iqr` | 43.98333333333335 | delay spread (minutes); also the cross-check fence, §5 |
| `waiting_p75` | 15.0 | `waiting_count` (visits) |
| `queue_p75` | 342.20258160400556 | `queue_mean_hours` (hours) |

Fit period only — no future leakage into the label definition. `src/eda_label.py`
recomputes these in `main()` and asserts the literals match (stale-literal guard).

**Deviation (flagged for reviewer):** the brief listed three keys
(`delay_median`, `delay_iqr`, `waiting_p75`). A **4th key, `queue_p75`, was added**:
the queue component's `norm()` needs a baseline in *its own unit* (hours), and neither
minutes-based key (`delay_median`/`delay_iqr`) nor the visit-count `waiting_p75` can
serve. `delay_iqr` is not used inside the index at all — it exists for the §5
cross-check — so without `queue_p75` the queue term would have had no defensible
baseline. The pre-2025 p75 of `queue_mean_hours` is the natural analog of
`waiting_p75`.

## 3. Fixed candidates failed measurement (plan's starting point)

The plan's Step 5 sweep (0.55 / 0.60 / 0.65 / 0.70) was run against the real index:

| threshold | positive rate |
|---|---|
| 0.55 | **0.000000** |
| 0.60 | 0.000000 |
| 0.65 | 0.000000 |
| 0.70 | 0.000000 |

**Index distribution over all 22,534 port-hours:** min 0.000000, median 0.007211,
p90 0.035692, **max 0.424054**.

The index scale is compressed far below the fixed candidates — the largest hour ever
observed (0.424054) does not reach even the lowest candidate (0.55). The fixed values
were a starting point, not a constraint; measurement rejected them outright.

## 4. Threshold decision (ruling R4)

**`CONGESTION_THRESHOLD = 0.016981388565773952`** — the 75th percentile of the
congestion index over pre-2025 hours (`hour_key < 2025-01-01`), i.e. the top quartile
of the train period's hours are positive by construction.

**Deviation (flagged for reviewer):** this replaces the plan's fixed candidates with an
empirical quantile. Rationale: the spec's **binding requirement is the 20–35% positive
band** ("Threshold chosen so the positive class is ~20–35% of hours"); the plan's
candidates were a starting point that measurement rejected (§3). A train-period p75
guarantees 25% on the fit window by definition, needs no hand-tuning, and uses only
train-period information (consistent with `FIT_BASELINE`'s cutoff, no leakage).

**Achieved positive rates** (`index >= threshold`):

| window | positive rate |
|---|---|
| pre-2025 (fit period) | **25.00%** |
| overall (2023-06 → 2025-12) | **26.32%** |
| 2025 only | **28.43%** |

All three sit inside the spec's 20–35% band; the t+6 shift in Task 3 moves the rate
within the band as well (re-checked there per plan Step 5).

**Important: aggregation rows vs. model rows.** The 22,534 hours and rates in this
section describe the initial hourly aggregation, before a six-hour-ahead label is
attached. `src.build_dataset` then removes 156 hours for which the t+6 target cannot
be observed: 150 because the corresponding future hour is missing inside a data gap,
and 6 at the end of the series. The resulting modeling table contains **22,378
labeled port-hours**. Because those unavailable target rows are excluded, the rates
on the final labeled table are slightly different (26.36% overall and 28.58% for
2025). The congestion-index formula and threshold are unchanged; this is a difference
in the rows available for model training and evaluation, not a change in the label
rule.

## 5. Cross-check: naive delay fence

Spec's example rule, `delay_median + delay_iqr` = 235.1 + 44.0 = **279.1 min**:

- only **14.61%** of port-hours have `delay_p50 ≥ 279.1 min` — *below* the 20–35%
  band, so a raw delay fence undershoots on its own.

This supports the index-based label over a single delay threshold: combining waiting
occupancy and queue with delay brings the positive share into the target band without
relying on one delay cutoff alone.

## 6. Delay field semantics

**Kaggle data dictionary:** `delay_minutes` = "Total dwell time (waiting + transit)
inside the port zone for an isolated port visit" — a whole-visit outcome, stamped as
the visit-final value on every hourly row of that visit (measured: 70.7% of
multi-hour visits carry a constant stamp from hour 0; 99.5% of non-null delay rows
occur while the visit is still open). The congestion index's delay term at the label
hour (t+6) is therefore a **retrospective severity measure of the vessels present at
t+6** — by design. Labels are observed after the fact by construction; that is
standard supervised practice (the label is what gets predicted and is only known
later), and Task 2's baseline/threshold remain fit on train-period hours only.

**Feature vs. label (fix round 1, ruling R6):** the `delay_dev_samedow` FEATURE was
switched from the snapshot hourly `delay_p50` source to a
**completed-visit-only** as-of statistic — median of visit-final `delay_minutes`
stamps over visits that completed in `(h−24h, h]` (NaN below 3 completions),
deviated against a past-only same-hour-of-week expanding median of that as-of
series — so no feature ever reads a visit outcome before the visit ends. The label
path (`congestion_index`, `FIT_BASELINE`, `CONGESTION_THRESHOLD`) is untouched.
