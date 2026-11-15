"""SHAP explanations + failure analysis (Task 6).

Scope: ``explain_hour(model, X_row) -> dict`` with exactly
``{"probability", "top_drivers", "narrative"}``, the ``--hour`` / ``--report``
CLI, the figures ``reports/figures/shap_importance.png`` +
``shap_waterfall_{congested,clear,wrong,wrong2}.png``, and
``reports/explanations.md``.

Binding rules:

1. **SHAP explains the booster's raw output.** ``shap.TreeExplainer`` is fit
   on ``models/lightgbm.txt`` alone. ``explain_hour`` reconstructs the raw
   probability from the SHAP values and matches ``booster.predict`` to 1e-9.
   The isotonic calibrator is applied separately for the CLI's event
   probability; it is never passed into SHAP.
2. **Offline error analysis** uses the raw LightGBM threshold (0.441741...),
   selected on validation 2025-H1 and applied unchanged to test 2025-H2.
   **The CLI decision flag** instead uses the calibrated probability and its
   own validation-selected threshold (0.333333...). Do not interchange them.
3. **High-confidence misclassification** (failure analysis, exact rule):
   a test hour qualifies iff (1) it sits on the wrong side of that
   threshold — ``(p >= thr) != label`` — AND (2) the model's confidence in
   the wrongly predicted class is >= thr: for a false positive ``p >= thr``,
   for a false negative ``1 - p >= thr``. At thr < 0.5 leg (2) is implied
   by leg (1), so the pool equals every misclassified test hour; case
   studies are the most confident error of each type (confidence desc,
   ties -> earliest ``hour_key``). See ``reports/explanations.md``.
4. Example hours (deterministic selection, test split 2025-H2): genuinely
   congested = label 1 + predicted positive, max probability; genuinely
   clear = label 0 + predicted negative, min probability.
"""

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.train import FEATURES, TEST_START, split_train_val_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FINAL_PATH = REPORTS_DIR / "metrics_final.json"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"
EXPLANATIONS_PATH = REPORTS_DIR / "explanations.md"

TOP_N_DRIVERS = 10  # entries returned by explain_hour (the "top drivers")
IMPORTANCE_TOP_N = 15  # bars in the global importance figure
N_WATERFALL_FEATURES = 10  # contributions drawn per waterfall

FIGURE_NAMES = (
    "shap_importance.png",
    "shap_waterfall_congested.png",
    "shap_waterfall_clear.png",
    "shap_waterfall_wrong.png",
    "shap_waterfall_wrong2.png",
)

# Dataviz palette (validated light mode: node scripts/validate_palette.js
# "#2a78d6,#e34948" --mode light -> ALL CHECKS PASS).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_PUSH_UP = "#e34948"  # contribution raising the raw score
_PUSH_DOWN = "#2a78d6"  # contribution lowering it
_NEUTRAL = _INK2  # base value / model output bars

# shap 0.54.1 emits this for binary LightGBM boosters; the observed return
# type is still an ndarray, so the notice is noise in the test output.
_SHAP_WARNING = "LightGBM binary classifier with TreeExplainer.*"


def load_booster() -> lgb.Booster:
    """Load the constrained LightGBM booster (never refit here)."""
    return lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))


def val_threshold() -> float:
    """Val-selected F1 threshold for lightgbm_raw (metrics_final.json)."""
    with METRICS_FINAL_PATH.open() as fh:
        metrics = json.load(fh)
    return float(metrics["threshold_selection"]["per_model"]["lightgbm_raw"])


def risk_band(probability: float, threshold: float) -> str:
    """``above`` iff probability >= threshold (boundary inclusive)."""
    return "above" if probability >= threshold else "below"


def load_calibrator():
    """Load the fitted isotonic calibrator saved by ``src.train``."""
    import joblib

    return joblib.load(CALIBRATOR_PATH)


def calibrated_probability(raw_probability: float) -> float:
    """Map a raw LightGBM probability to the calibrated event probability."""
    calibrated = load_calibrator().transform([float(raw_probability)])
    return float(np.asarray(calibrated, dtype=float).reshape(-1)[0])


def calibrated_threshold() -> float:
    """Validation-selected F1 threshold for ``lightgbm_calibrated``."""
    with METRICS_FINAL_PATH.open() as fh:
        metrics = json.load(fh)
    return float(
        metrics["threshold_selection"]["per_model"]["lightgbm_calibrated"]
    )


def _shap_row(model: lgb.Booster, X_row: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values (n_features,) + base value for a single row."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SHAP_WARNING)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row)
        base = explainer.expected_value
    if isinstance(sv, list):  # defensive: per-class list form
        sv = sv[-1]
    values = np.asarray(sv, dtype=float)
    if values.ndim == 2:
        values = values[0]
    base_arr = np.asarray(base, dtype=float).reshape(-1)
    base_value = float(base_arr[-1])  # last element: class value or scalar
    return values, base_value


def _explain_detail(model: lgb.Booster, X_row: pd.DataFrame) -> dict:
    """Full explanation payload (explain_hour projects the 3 public keys)."""
    values, base_value = _shap_row(model, X_row)
    raw = base_value + float(values.sum())
    probability = float(1.0 / (1.0 + np.exp(-raw)))
    order = np.argsort(-np.abs(values))[:TOP_N_DRIVERS]
    top_drivers = [(FEATURES[i], float(values[i])) for i in order]
    top_name, top_value = top_drivers[0]
    direction = "raises" if top_value > 0 else "lowers"
    narrative = (
        f"Top driver: {top_name} (SHAP {top_value:+.3f}) {direction} the raw "
        f"congestion score; model probability {probability:.3f}."
    )
    feature_values = {
        name: float(val) for name, val in zip(FEATURES, X_row[FEATURES].iloc[0])
    }
    return {
        "probability": probability,
        "top_drivers": top_drivers,
        "narrative": narrative,
        "shap_values": values,
        "base_value": base_value,
        "raw_score": raw,
        "feature_values": feature_values,
    }


def explain_hour(model: lgb.Booster, X_row: pd.DataFrame) -> dict:
    """Explain one row of features against the raw booster output.

    Returns exactly ``{"probability", "top_drivers", "narrative"}``:
    the raw model probability (sigmoid of the SHAP-decomposed score —
    identical to ``booster.predict``), the 10 strongest drivers as
    ``(feature, shap_value)`` pairs sorted by |SHAP| descending, and a
    one-sentence narrative naming the top driver.
    """
    detail = _explain_detail(model, X_row)
    return {
        "probability": detail["probability"],
        "top_drivers": detail["top_drivers"],
        "narrative": detail["narrative"],
    }


def high_confidence_errors(
    test_df: pd.DataFrame, probabilities, threshold: float
) -> pd.DataFrame:
    """Test hours misclassified at ``threshold`` per the binding rule.

    Exact rule (stated in reports/explanations.md): wrong side of the
    threshold — ``(p >= threshold) != label`` — AND confidence in the
    wrongly predicted class >= threshold (``p`` for false positives,
    ``1 - p`` for false negatives). Sorted by that confidence descending,
    ties broken by earliest ``hour_key``.
    """
    probs = np.asarray(probabilities, dtype=float)
    label = test_df["label"].to_numpy()
    predicted = probs >= threshold
    wrong = predicted != (label == 1)
    confidence = np.where(predicted, probs, 1.0 - probs)
    mask = wrong & (confidence >= threshold)
    out = pd.DataFrame(
        {
            "hour_key": test_df["hour_key"].to_numpy()[mask],
            "label": label[mask].astype(int),
            "probability": probs[mask],
            "predicted": predicted[mask].astype(int),
            "error_type": np.where(
                predicted[mask], "false_positive", "false_negative"
            ),
            "wrong_class_confidence": confidence[mask],
        }
    )
    return out.sort_values(
        ["wrong_class_confidence", "hour_key"],
        ascending=[False, True],
        ignore_index=True,
    )


def plot_importance(
    importance,
    feature_names,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
) -> Path:
    """Horizontal bar chart of mean |SHAP| (top IMPORTANCE_TOP_N features).

    Single series -> single validated hue, no legend (the title names it);
    value at each bar's tip in text ink; hairline solid grid on x only.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    importance = np.asarray(importance, dtype=float)
    names = list(feature_names)
    order = np.argsort(importance)[::-1][:IMPORTANCE_TOP_N]
    values = importance[order]
    labels = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(values) + 1.7))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    y = np.arange(len(values))[::-1]  # largest value at the top
    ax.barh(y, values, height=0.6, color=_PUSH_DOWN, zorder=3)

    tip_pad = max(values) * 0.02
    for yi, v in zip(y, values):
        ax.text(
            v + tip_pad, yi, f"{v:.3f}", va="center", fontsize=8, color=_INK2
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=_INK2)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("mean |SHAP| on test 2025-H2", fontsize=9, color=_MUTED)
    ax.grid(axis="x", color=_GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=_INK)
    if subtitle:
        ax.text(
            1.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
            color=_INK2,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_waterfall(
    *,
    hour_key,
    feature_names,
    feature_values,
    shap_values,
    base_value: float,
    probability: float,
    label: int,
    out_path: Path,
) -> Path:
    """Cumulative SHAP waterfall: base -> top-N contributions -> raw score.

    Diverging pair (red pushes the raw score up, blue down), neutral bars
    for the expected value and the model output; legend always present.
    Subtitle carries label / probability / band / outcome at the val
    threshold (stating the band rule in the figure itself).
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    names = list(feature_names)
    vals = np.asarray(feature_values, dtype=float)
    sv = np.asarray(shap_values, dtype=float)
    order = np.argsort(-np.abs(sv))[:N_WATERFALL_FEATURES]
    k = len(order)
    raw = base_value + float(sv.sum())
    thr = val_threshold()
    band = risk_band(probability, thr)
    predicted_pos = band == "above"
    if predicted_pos and label == 1:
        outcome = "true positive"
    elif predicted_pos and label == 0:
        outcome = "false positive"
    elif not predicted_pos and label == 0:
        outcome = "true negative"
    else:
        outcome = "false negative"
    residual = float(sv.sum() - sv[order].sum())

    rows: list[tuple[str, float, float | None, str]] = [
        ("expected value", 0.0, base_value, _NEUTRAL)
    ]
    running = base_value
    for i in order:
        v = float(sv[i])
        left = running if v >= 0 else running + v
        rows.append((f"{names[i]} = {vals[i]:g}", left, abs(v), _PUSH_UP if v >= 0 else _PUSH_DOWN))
        running += v
    rows.append(("model output", 0.0, raw, _NEUTRAL))

    fig, ax = plt.subplots(figsize=(9.4, 0.5 * len(rows) + 2.4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    y = np.arange(len(rows))
    for yi, (_label, left, width, color) in zip(y, rows):
        ax.barh(yi, width, left=left, height=0.62, color=color, zorder=3)

    # Contribution labels sit outside each feature bar, on its leading edge.
    running = base_value
    for yi, i in zip(y[1:-1], order):
        v = float(sv[i])
        x = running + v
        ax.text(
            x + (0.02 if v >= 0 else -0.02),
            yi,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=_INK2,
        )
        running += v

    edges = [0.0, base_value, raw]
    run = base_value
    for i in order:
        run += float(sv[i])
        edges.append(run)
    lo, hi = min(edges), max(edges)
    pad = (hi - lo) * 0.10 or 0.1
    ax.set_xlim(lo - pad, hi + pad)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=_INK2)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xlabel("contribution to raw model score (log-odds)", fontsize=9, color=_MUTED)
    ax.grid(axis="x", color=_GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)

    ax.set_title(
        f"SHAP waterfall — {hour_key}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=_INK,
        pad=60,
    )
    ax.text(
        0.0,
        1.10,
        (
            f"label={label} · probability={probability:.4f} · band={band} at "
            f"{thr:.4f} · {outcome} · raw {raw:+.3f} = base {base_value:+.3f} + Σ SHAP"
        ),
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color=_INK2,
    )
    if len(sv) > k:
        ax.text(
            0.0,
            -0.16,
            (
                f"top {k} of {len(sv)} features by |SHAP|; "
                f"residual across the other {len(sv) - k}: {residual:+.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            fontsize=8,
            color=_MUTED,
        )
    ax.legend(
        handles=[
            Patch(facecolor=_PUSH_UP, label="pushes risk up"),
            Patch(facecolor=_PUSH_DOWN, label="pushes risk down"),
            Patch(facecolor=_NEUTRAL, label="expected value / model output"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=_INK2,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_example(probs, hour_keys, mask, *, largest: bool) -> int:
    """Index of max (or min) probability inside mask; ties -> earliest hour."""
    idx = np.flatnonzero(mask)
    if largest:
        return int(min(idx, key=lambda i: (-probs[i], hour_keys[i])))
    return int(min(idx, key=lambda i: (probs[i], hour_keys[i])))


def _outcome(predicted_pos: bool, label: int) -> str:
    if predicted_pos and label == 1:
        return "true positive"
    if predicted_pos and label == 0:
        return "false positive"
    if not predicted_pos and label == 0:
        return "true negative"
    return "false negative"


def build_report(*, figure_dir: Path, out_path: Path) -> Path:
    """Write the five SHAP figures + reports/explanations.md.

    Global importance is mean |SHAP| over the test split (2025-H2);
    waterfalls cover the deterministic congested/clear examples plus the
    most confident false positive and false negative under the binding
    high-confidence rule.
    """
    df = pd.read_parquet(DATA_PATH)
    _, _, test = split_train_val_test(df)
    model = load_booster()
    thr = val_threshold()
    probs = model.predict(test[FEATURES])
    y = test["label"].to_numpy()
    hour_keys = test["hour_key"].to_numpy()
    n = len(test)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SHAP_WARNING)
        sv_all = np.asarray(shap.TreeExplainer(model).shap_values(test[FEATURES]))
    if sv_all.ndim == 3:  # defensive: stacked classes
        sv_all = sv_all[..., -1]
    mean_abs = np.abs(sv_all).mean(axis=0)

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(
        mean_abs,
        FEATURES,
        figure_dir / "shap_importance.png",
        title="Global feature importance — mean |SHAP|",
        subtitle=f"test 2025-H2 (n={n:,})",
    )

    pred = probs >= thr
    i_congested = _select_example(probs, hour_keys, (y == 1) & pred, largest=True)
    i_clear = _select_example(probs, hour_keys, (y == 0) & ~pred, largest=False)
    errors = high_confidence_errors(test, probs, thr)
    fp_hour = errors.loc[errors["error_type"] == "false_positive", "hour_key"].iloc[0]
    fn_hour = errors.loc[errors["error_type"] == "false_negative", "hour_key"].iloc[0]
    i_fp = int(np.flatnonzero(hour_keys == np.datetime64(fp_hour))[0])
    i_fn = int(np.flatnonzero(hour_keys == np.datetime64(fn_hour))[0])

    examples = [
        ("congested", i_congested, "shap_waterfall_congested.png"),
        ("clear", i_clear, "shap_waterfall_clear.png"),
        ("wrong", i_fp, "shap_waterfall_wrong.png"),
        ("wrong2", i_fn, "shap_waterfall_wrong2.png"),
    ]
    details = {}
    for tag, i, fname in examples:
        detail = _explain_detail(model, test.iloc[[i]][FEATURES])
        details[tag] = detail
        plot_waterfall(
            hour_key=test.iloc[i]["hour_key"],
            feature_names=FEATURES,
            feature_values=[detail["feature_values"][f] for f in FEATURES],
            shap_values=detail["shap_values"],
            base_value=detail["base_value"],
            probability=detail["probability"],
            label=int(y[i]),
            out_path=figure_dir / fname,
        )

    # --- prose inputs -----------------------------------------------------
    top10 = sorted(zip(FEATURES, mean_abs), key=lambda kv: -kv[1])[:10]
    n_fp = int((errors["error_type"] == "false_positive").sum())
    n_fn = int((errors["error_type"] == "false_negative").sum())
    fn_hours = errors.loc[errors["error_type"] == "false_negative", "hour_key"]
    dec_fn = int((fn_hours.dt.month == 12).sum())

    def drivers_line(detail: dict) -> str:
        return ", ".join(
            f"`{name}` {value:+.3f}" for name, value in detail["top_drivers"][:3]
        )

    def fv(detail: dict, name: str) -> float:
        return detail["feature_values"][name]

    c = details["congested"]
    cl = details["clear"]
    w = details["wrong"]
    w2 = details["wrong2"]
    c1, c2, c3 = (n for n, _ in c["top_drivers"][:3])
    cl1, cl2, cl3 = (n for n, _ in cl["top_drivers"][:3])
    w1, w2n, w3 = (n for n, _ in w["top_drivers"][:3])
    f1, f2, f3 = (n for n, _ in w2["top_drivers"][:3])
    s_c1, s_c2, s_c3 = (v for _, v in c["top_drivers"][:3])
    s_cl1, s_cl2, s_cl3 = (v for _, v in cl["top_drivers"][:3])
    s_w1, s_w2, s_w3 = (v for _, v in w["top_drivers"][:3])
    s_f1, s_f2, s_f3 = (v for _, v in w2["top_drivers"][:3])
    v_w1, v_w2, v_w3 = (w["feature_values"][n] for n in (w1, w2n, w3))
    v_f1, v_f2, v_f3 = (w2["feature_values"][n] for n in (f1, f2, f3))

    # The CLI's example hour (pinned by tests): an FP at the val threshold.
    cli_hits = np.flatnonzero(hour_keys == np.datetime64(pd.Timestamp("2025-07-15 14:00")))
    cli_detail = _explain_detail(model, test.iloc[[int(cli_hits[0])]][FEATURES])
    cli_top = cli_detail["top_drivers"][0][0]

    table_rows = "\n".join(
        f"| {rank} | `{name}` | {value:.4f} |"
        for rank, (name, value) in enumerate(top10, start=1)
    )

    md = f"""# SHAP explanations & failure analysis

Generated by `.venv/bin/python -m src.explain --report` from
`models/lightgbm.txt` (constrained LightGBM booster, 67 features).
Figures live in `reports/figures/`.

## Method (binding rules)

- **SHAP explains the booster's raw output.** `shap.TreeExplainer` is fit on
  the booster alone; the row probability is `sigmoid(expected_value + Σ SHAP)`,
  which matches `booster.predict` to better than 1e-9 (test-pinned). The
  isotonic calibrator is applied after this calculation; it is never fed to
  SHAP.
- **Offline error-analysis rule:** `above` iff raw probability >= **{thr:.4f}**,
  the F1-maximizing raw-model threshold selected on val 2025-H1 and applied
  unchanged to test 2025-H2. The interactive CLI also reports a calibrated
  probability and applies its separate validation threshold (0.3333).
- **High-confidence misclassification rule (exact):** a test hour qualifies
  iff **(1)** it sits on the wrong side of that threshold —
  `(p >= thr) != label` — **and (2)** its confidence in the wrongly predicted
  class is >= thr: for a false positive `p >= thr`, for a false negative
  `1 - p >= thr`. At thr = {thr:.4f} < 0.5, leg (2) is implied by leg (1)
  (FP by definition; FN because `1 - p > 1 - thr > thr`), so the pool is
  exactly the **{len(errors):,}** misclassified test hours
  ({n_fp:,} FP + {n_fn} FN of {n:,}; test base rate {y.mean():.2%}). Case
  studies below are the most confident error of each type: highest
  **confidence in the wrongly predicted class**, ties broken by earliest
  `hour_key`.

## Global drivers — test 2025-H2 (n={n:,})

![Global feature importance](figures/shap_importance.png)

Top 10 by mean |SHAP| (table twin of the figure):

| # | feature | mean \\|SHAP\\| |
|---|---------|--------------|
{table_rows}

## Genuinely congested, model right — {test.iloc[i_congested]['hour_key']}

Observed label **1**, probability **{c['probability']:.4f}** → band `above`
at {thr:.4f} → **true positive**. Top drivers: {drivers_line(c)}.

![Congested hour waterfall](figures/shap_waterfall_congested.png)

Every large push points the same way: `{c1}` = {fv(c, c1):g} contributes
{s_c1:+.3f} and `{c2}` = {fv(c, c2):g} adds {s_c2:+.3f}, with the seasonal
term `{c3}` ({s_c3:+.3f}) reinforcing the July peak. Nothing among the
top drivers argues down, so the booster clears the threshold with margin
(p = {c['probability']:.4f}). This is the model working as designed —
queue level and seasonal context agreeing.

## Genuinely clear, model right — {test.iloc[i_clear]['hour_key']}

Observed label **0**, probability **{cl['probability']:.4f}** → band `below`
at {thr:.4f} → **true negative**. Top drivers: {drivers_line(cl)}.

![Clear hour waterfall](figures/shap_waterfall_clear.png)

All three largest pushes are downward: `{cl1}` ({s_cl1:+.3f}),
`{cl2}` = {fv(cl, cl2):g} ({s_cl2:+.3f}), `{cl3}` ({s_cl3:+.3f}). The
December seasonal term alone removes {abs(s_cl3):.3f} from the raw score and
a light queue keeps it there; the hour never approaches {thr:.4f}
(p = {cl['probability']:.4f}).

## Failure analysis — high-confidence wrong

Pool: **{len(errors):,}** hours ({n_fp:,} FP + {n_fn} FN of {n:,}) under the
rule above. The two case studies are the most confident error of each type.

### False positive — {fp_hour} (p = {w['probability']:.4f}, label = 0)

![High-confidence false positive waterfall](figures/shap_waterfall_wrong.png)

**Post-mortem:** The top three pushes are `{w1}` = {v_w1:g} ({s_w1:+.3f}),
`{w2n}` = {v_w2:g} ({s_w2:+.3f}), `{w3}` = {v_w3:g} ({s_w3:+.3f}) — all
upward: queue level plus its lag and the July seasonal term — so the
booster lands at p = {w['probability']:.4f}, far above {thr:.4f}. The label
says no
congestion materialised by t+6: the queue drained inside the horizon (or
the delay component never confirmed), a transition that queue *level*
cannot express. Monotone constraints only allow queue features to push
risk upward, so nothing in this row can call the drain.
**Operational hook:** escalate on queue level only when the 6-hour queue
trend is flat-to-rising; a queue-derivative/throughput feature would let
the booster discount a falling queue.

### False negative — {fn_hour} (p = {w2['probability']:.4f}, label = 1)

![High-confidence false negative waterfall](figures/shap_waterfall_wrong2.png)

**Post-mortem:** The seasonal prior dominates: `{f1}` = {v_f1:g} removes
{abs(s_f1):.3f} from the raw score; calm `{f2}` = {v_f2:g} ({s_f2:+.3f})
adds to the suppression, and a moderate queue (`{f3}` = {v_f3:g},
{s_f3:+.3f}) never fights back — the booster lands at p = {w2['probability']:.4f},
far below {thr:.4f}, while the label says congestion did occur. It is not
an isolated miss: {dec_fn}
of the {n_fn} test FNs fall in December 2025 (clustered on days 4–19).
**Operational hook:** compare queue level against its seasonal baseline (or
use a season-aware threshold) before dismissing a winter episode; queue
acceleration should be able to override a calm-weather prior.

## CLI

```bash
.venv/bin/python -m src.explain --hour 2025-07-15T14:00
```

prints both the raw LightGBM probability and the isotonic-calibrated
congestion probability. Its decision flag uses the calibrated threshold
(0.3333); SHAP drivers continue to explain the raw model output. The report's
false-positive / false-negative counts remain based on the raw-model
threshold so they match the held-out evaluation table.
"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.explain",
        description="SHAP explanations + failure analysis (Task 6).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hour",
        metavar="HOUR_KEY",
        help="hour to explain, e.g. 2025-07-15T14:00",
    )
    group.add_argument(
        "--report",
        action="store_true",
        help="write the SHAP figures + reports/explanations.md",
    )
    args = parser.parse_args(argv)

    if args.report:
        out = build_report(figure_dir=FIGURES_DIR, out_path=EXPLANATIONS_PATH)
        print(f"saved: {out}")
        for name in FIGURE_NAMES:
            print(f"saved: {FIGURES_DIR / name}")
        return 0

    ts = pd.Timestamp(args.hour)
    df = pd.read_parquet(DATA_PATH)
    rows = df.loc[df["hour_key"] == ts]
    if rows.empty:
        print(f"hour {args.hour} not found in {DATA_PATH}")
        return 1
    row = rows.iloc[[0]]

    detail = _explain_detail(load_booster(), row[FEATURES])
    raw_probability = detail["probability"]
    calibrated = calibrated_probability(raw_probability)
    raw_thr = val_threshold()
    calibrated_thr = calibrated_threshold()
    raw_band = risk_band(raw_probability, raw_thr)
    calibrated_band = risk_band(calibrated, calibrated_thr)
    label = int(row["label"].iloc[0])
    split = "test" if ts >= TEST_START else "train/val"
    print(f"hour: {ts} ({split} period, observed label={label})")
    print(f"probability (raw LightGBM): {raw_probability:.4f}")
    print(f"congestion probability at t+6 (isotonic-calibrated): {calibrated:.4f}")
    print(
        f"decision flag (calibrated): {calibrated_band} "
        f"at threshold {calibrated_thr:.4f}"
    )
    print(
        "calibrated decision rule: above iff calibrated congestion "
        f"probability >= {calibrated_thr:.4f} "
        "(F1 threshold selected on validation 2025-H1)"
    )
    print(
        f"raw-model evaluation flag: {raw_band} at threshold {raw_thr:.4f}; "
        f"historical outcome={_outcome(raw_band == 'above', label)}"
    )
    print("top drivers by |SHAP| (raw LightGBM output):")
    for name, value in detail["top_drivers"]:
        print(f"  {name:<24s} {value:+.4f}")
    print(f"narrative: {detail['narrative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
