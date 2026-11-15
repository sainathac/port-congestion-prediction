"""Baselines, LightGBM, expanding-window CV, Optuna, calibration (Task 4).

Scope: ``FEATURES`` / ``MONOTONIC_CONSTRAINTS``, calendar splits with
the label horizon purged at every boundary, expanding-window CV, the
persistence + logistic baselines, LightGBM with monotonic constraints,
Optuna tuning confined to the train period, isotonic calibration on the
2025-H1 validation window, and the reliability diagram.

Split hygiene (controller binding resolution): membership is defined by
``hour_key`` (train through 2024-12-31, val 2025-H1, test 2025-H2) and
PURGED so no training-window row has ``label_hour`` landing on or after
the next window's start:

    max(train.label_hour) < min(val.hour_key)
    max(val.label_hour)  < min(test.hour_key)

The same purge applies inside every expanding-window CV fold. This is
strictly stronger than a plain hour_key-overlap check: a row at
2024-12-31 23:00 has hour_key inside train but its label reads data at
2025-01-01 05:00 — inside the val window.

Data windows, in the order the pipeline touches them:

1. Optuna + fold metrics: expanding-window CV on the PURGED TRAIN
   period only — neither 2025-H1 nor 2025-H2 is ever passed in.
2. Final model selection: top-5 trials by CV PR-AUC are refit on train
   and ranked by PR-AUC on val 2025-H1 (the only val usage besides
   calibration).
3. Calibration: isotonic regression on the winner's out-of-sample
   predictions for val 2025-H1 (model fit on train only, so those
   predictions are out-of-fold with respect to the calibration data).
4. Test 2025-H2: scored exactly once, at the end, raw and calibrated.

Persistence semantics: the forecast for hour h is the congestion
observed AT h — the ``label`` of the row whose ``label_hour == h``
(the row at h-6) — known at h, never anything after h. It is NOT the
row's own label (congestion at h+6).

Optuna search space (as described in the assignment; the bounded search
and early stopping keep the trial run manageable):

    learning_rate     loguniform(0.01, 0.2)
    num_leaves        int loguniform(7, 127)
    min_child_samples int uniform(5, 100)
    feature_fraction  uniform(0.5, 1.0)
    bagging_fraction  uniform(0.5, 1.0)   (bagging_freq=1 fixed)
    n_estimators      int loguniform(100, 1000) — upper bound only;
                      early stopping (50 rounds) on the fold's valid
                      window picks the actual number of trees
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_dataset import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "port_hours.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def _repo_path(path: Path) -> str:
    """Return a stable repository-relative path for saved metadata."""
    return path.relative_to(PROJECT_ROOT).as_posix()


# Calendar split boundaries (hour_key membership; purge below).
VAL_START = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-07-01")

# Explicit ordered feature list (Task 3's FEATURE_COLS — single source of
# truth for column order; LightGBM's monotonic constraints are positional,
# so this order is the contract).
FEATURES: list[str] = list(FEATURE_COLS)

# +1 for queue/waiting/density/arrivals (spec §3: more queue/waiting can
# never lower predicted risk); 0 elsewhere — including
# delay_dev_samedow (as-of completed-visit statistic, treated like any
# other numeric feature), vessel_count, and the weather x traffic
# interactions (constraining a product is ill-defined).
_CONSTRAINED_PREFIXES = (
    "waiting_count",
    "queue_mean_hours",
    "ship_density_mean",
    "arrivals",
)
MONOTONIC_CONSTRAINTS: list[int] = [
    1 if f.startswith(_CONSTRAINED_PREFIXES) else 0 for f in FEATURES
]

# Hand-picked logistic subset (brief Step 3): the at-t queue/waiting/
# density/traffic state plus one weather pair and the two quality
# columns — small, explainable, no lags of the driver itself.
LOGISTIC_FEATURES: list[str] = [
    "waiting_count",
    "queue_mean_hours",
    "ship_density_mean",
    "vessel_count",
    "arrivals",
    "arrivals_roll3h",
    "wind_speed_10m",
    "wave_height",
    "delay_dev_samedow",
    "imputed_fraction",
]


def split_train_val_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calendar splits by hour_key, label horizon purged at each boundary.

    train: ``hour_key < 2025-01-01``, purged to ``label_hour < 2025-01-01``;
    val: 2025-H1, purged to ``label_hour < 2025-07-01``;
    test: 2025-H2 (nothing follows it — no purge needed).

    Purging against the calendar boundary (rather than the next window's
    first observed row) is at least as strict as
    ``max(label_hour) < min(next.hour_key)`` because the next window
    starts at or after that boundary by construction.
    """
    train = df.loc[df["hour_key"] < VAL_START].copy()
    val = df.loc[
        (df["hour_key"] >= VAL_START) & (df["hour_key"] < TEST_START)
    ].copy()
    test = df.loc[df["hour_key"] >= TEST_START].copy()
    train = train.loc[train["label_hour"] < VAL_START]
    val = val.loc[val["label_hour"] < TEST_START]
    return train, val, test


def expanding_window_splits(
    df: pd.DataFrame, n_splits: int = 3, min_train_days: int = 365
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window CV folds over ``df`` with the label horizon purged.

    The first validation window starts at ``min(hour_key) +
    min_train_days``; the remaining span is cut into ``n_splits`` equal
    calendar windows. For fold k the train side is every row strictly
    before that fold's validation start, purged of rows whose
    ``label_hour`` reaches into the validation window — so
    ``max(train.label_hour) < min(valid.hour_key)`` holds by
    construction, and train windows strictly grow fold over fold.
    """
    df = df.sort_values("hour_key", kind="stable")
    t0 = df["hour_key"].min()
    t_end = df["hour_key"].max()
    first_valid_start = t0 + pd.Timedelta(days=min_train_days)
    span = (t_end + pd.Timedelta(hours=1)) - first_valid_start
    fold_len = span / n_splits
    starts = [first_valid_start + k * fold_len for k in range(n_splits)]

    for k, valid_start in enumerate(starts):
        valid_end = (
            starts[k + 1] if k + 1 < n_splits else t_end + pd.Timedelta(hours=1)
        )
        valid = df.loc[
            (df["hour_key"] >= valid_start) & (df["hour_key"] < valid_end)
        ]
        train = df.loc[df["hour_key"] < valid_start]
        train = train.loc[train["label_hour"] < valid_start]
        yield train, valid


def persistence_scores(df: pd.DataFrame) -> pd.Series:
    """Forecast for hour h = congestion observed AT h (known at h).

    congestion(h) is the ``label`` of the row whose ``label_hour == h``
    (the row at h-6). Rows whose hour has no observed label (series
    start, outage boundaries) map to NaN — scored rows are the ones
    where ``congestion(h)`` exists.
    """
    by_label_hour = df.set_index("label_hour")["label"]
    return df["hour_key"].map(by_label_hour).astype(float)


def make_logistic_pipeline() -> Pipeline:
    """Balanced logistic regression on ``LOGISTIC_FEATURES``.

    Median imputation + standardization are fit inside the pipeline on
    whatever training fold it is given (no statistics from val/test).
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> float:
    """ECE over ``n_bins`` equal-width probability bins.

    ECE = sum_b (n_b / n) * |mean_prob_b - frac_positive_b|, empty bins
    skipped (contribute nothing).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p, edges[1:-1], right=False)
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        total += n_b * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total / len(y))


# --- LightGBM + Optuna (Step 4) ---------------------------------------------

EARLY_STOPPING_ROUNDS = 50
OPTUNA_SEED = 42
TOP_K_FOR_VAL_SELECTION = 5

SEARCH_SPACE_DOC = {
    "learning_rate": "loguniform(0.01, 0.2)",
    "num_leaves": "int loguniform(7, 127)",
    "min_child_samples": "int uniform(5, 100)",
    "feature_fraction": "uniform(0.5, 1.0)",
    "bagging_fraction": "uniform(0.5, 1.0), bagging_freq=1 fixed",
    "n_estimators": (
        "int loguniform(100, 1000) upper bound; early stopping "
        f"({EARLY_STOPPING_ROUNDS} rounds) on the fold's valid window "
        "selects the actual tree count"
    ),
}


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
) -> lgb.LGBMClassifier:
    """Brief-specified LightGBM binary classifier with per-fit imbalance weight.

    ``scale_pos_weight`` is recomputed from whatever ``y_train`` is passed
    (fold train or the full train window). ``monotone_constraints`` is the
    positional ``MONOTONIC_CONSTRAINTS`` list aligned with ``FEATURES``
    (X must be column-ordered as FEATURES). With ``X_eval`` the fit stops
    early on that window's logloss (used inside CV; the eval window is
    never trained on).
    """
    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        # LightGBM's real config name is monotone_constraints (no "ic"):
        # the brief's "monotonic_constraints" spelling is silently ignored
        # by LightGBM 4.7.0 (dump stays [], trees unconstrained).
        monotone_constraints=MONOTONIC_CONSTRAINTS,
        random_state=42,
        n_jobs=-1,
        bagging_freq=1,
        verbose=-1,
        **params,
    )
    if X_eval is not None:
        model.fit(
            X_train,
            y_train,
            eval_X=X_eval,
            eval_y=y_eval,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def _tune_objective(
    trial: optuna.Trial, folds: list[tuple[pd.DataFrame, pd.DataFrame]]
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
    }
    pr_aucs: list[float] = []
    best_iters: list[int] = []
    for tr, va in folds:
        model = fit_lgbm(
            tr[FEATURES], tr["label"], params, va[FEATURES], va["label"]
        )
        proba = model.predict_proba(va[FEATURES])[:, 1]
        pr_aucs.append(average_precision_score(va["label"], proba))
        best_iters.append(model.best_iteration_ or params["n_estimators"])
    # Mean early-stopped tree count, so the final refit (no eval window)
    # matches what CV measured instead of running to the n_estimators cap.
    trial.set_user_attr("mean_best_iter", float(np.mean(best_iters)))
    return float(np.mean(pr_aucs))


def tune_lgbm(
    train_df: pd.DataFrame, n_trials: int = 50, n_splits: int = 3
) -> dict:
    """Optuna over expanding-window CV on ``train_df`` ONLY.

    Callers must pass the purged train window — val 2025-H1 feeds final
    model selection/calibration afterwards, test 2025-H2 is never touched
    here (controller resolution: Optuna must never see the test window).
    """
    folds = list(expanding_window_splits(train_df, n_splits=n_splits))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    study.optimize(
        lambda trial: _tune_objective(trial, folds),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    complete = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    complete.sort(key=lambda t: t.value, reverse=True)
    top = [
        {"params": dict(t.params), "mean_best_iter": t.user_attrs["mean_best_iter"]}
        for t in complete[:TOP_K_FOR_VAL_SELECTION]
    ]
    return {
        "n_trials": len(complete),
        "objective": "mean PR-AUC across expanding-window CV folds (train period)",
        "best_params": dict(study.best_params),
        "best_cv_pr_auc": float(study.best_value),
        "top_candidates": top,
        "search_space": SEARCH_SPACE_DOC,
    }


def _score_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """roc_auc / pr_auc / brier / ece on rows where scores exist.

    Persistence maps to NaN when congestion(t) is unobserved (series
    start / outage boundaries); those rows are dropped for that model
    only, and ``n_scored`` records the coverage.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "n_scored": int(mask.sum()),
    }


def _mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _cv_summary(per_fold: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for metric in ("roc_auc", "pr_auc", "brier", "ece"):
        out[metric] = _mean_std([f[metric] for f in per_fold])
    out["n_scored_total"] = int(sum(f["n_scored"] for f in per_fold))
    return out


def plot_reliability(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram: raw vs calibrated probabilities (test set)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    y = np.asarray(y_true, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _bin_means(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.digitize(p, edges[1:-1], right=False)
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                xs.append(float(p[mask].mean()))
                ys.append(float(y[mask].mean()))
        return np.array(xs), np.array(ys)

    raw_x, raw_y = _bin_means(np.asarray(raw_probs, dtype=float))
    cal_x, cal_y = _bin_means(np.asarray(cal_probs, dtype=float))
    ece_raw = expected_calibration_error(y, raw_probs)
    ece_cal = expected_calibration_error(y, cal_probs)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(raw_x, raw_y, "o-", color="#1f77b4", lw=1.5, ms=5,
            label=f"raw LightGBM (ECE={ece_raw:.3f})")
    ax.plot(cal_x, cal_y, "s-", color="#d62728", lw=1.5, ms=5,
            label=f"isotonic-calibrated (ECE={ece_cal:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(
        f"Reliability — test 2025-H2 (n={len(y)}, base rate={y.mean():.3f})"
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def train_all(df: pd.DataFrame, n_trials: int = 50, n_splits: int = 3) -> dict:
    """Tune, select, calibrate, and evaluate — full Task 4 pipeline.

    Returns the metrics dict that is also written to
    ``reports/metrics.json``; saves ``models/lightgbm.txt``,
    ``models/calibrator.pkl``, ``models/logistic.pkl``, and
    ``reports/figures/reliability_diagram.png``.
    """
    t_start = time.time()
    # Persistence score travels with each row (score at h only reads the
    # label stamped at hour h — see module docstring).
    df = df.copy()
    df["_persistence"] = persistence_scores(df)
    train, val, test = split_train_val_test(df)

    split_block = {}
    for name, part, n_raw in (
        ("train", train, int((df["hour_key"] < VAL_START).sum())),
        ("val", val, int(((df["hour_key"] >= VAL_START)
                          & (df["hour_key"] < TEST_START)).sum())),
        ("test", test, int((df["hour_key"] >= TEST_START).sum())),
    ):
        split_block[name] = {
            "n": len(part),
            "n_raw_calendar": n_raw,
            "n_purged": n_raw - len(part),
            "start": str(part["hour_key"].min()),
            "end": str(part["hour_key"].max()),
            "label_hour_max": str(part["label_hour"].max()),
            "positive_rate": float(part["label"].mean()),
        }
    print(
        "splits (hour_key membership, label horizon purged):\n"
        + "\n".join(
            f"  {k}: n={v['n']:,} purged={v['n_purged']} "
            f"positive_rate={v['positive_rate']:.4f} "
            f"[{v['start']} .. {v['end']}]"
            for k, v in split_block.items()
        )
    )

    # 1) Optuna on train-period CV only.
    t0 = time.time()
    tune = tune_lgbm(train, n_trials=n_trials, n_splits=n_splits)
    tune_seconds = time.time() - t0
    print(
        f"optuna: {tune['n_trials']} trials in {tune_seconds:.1f}s | "
        f"best mean fold PR-AUC={tune['best_cv_pr_auc']:.4f} | "
        f"best params={tune['best_params']}"
    )

    folds = list(expanding_window_splits(train, n_splits=n_splits))
    fold_rows = []
    for i, (tr, va) in enumerate(folds):
        fold_rows.append(
            {
                "fold": i,
                "train_n": len(tr),
                "valid_n": len(va),
                "train_start": str(tr["hour_key"].min()),
                "train_end": str(tr["hour_key"].max()),
                "valid_start": str(va["hour_key"].min()),
                "valid_end": str(va["hour_key"].max()),
                # Rows the fold's horizon purge dropped: the calendar
                # prefix before the valid window, minus what survived.
                "purged_rows": int(
                    (train["hour_key"] < va["hour_key"].min()).sum() - len(tr)
                ),
                "train_positive_rate": float(tr["label"].mean()),
                "valid_positive_rate": float(va["label"].mean()),
            }
        )

    # 2) Final model selection: top-5 trials refit on train, ranked by
    #    PR-AUC on val 2025-H1 (test still untouched).
    candidates = []
    for rank, cand in enumerate(tune["top_candidates"]):
        params = dict(cand["params"])
        # Refit with the CV-chosen tree count (no eval window: val must
        # not leak into fitting, only into selection/calibration).
        refit_params = {**params, "n_estimators": int(cand["mean_best_iter"])}
        model = fit_lgbm(train[FEATURES], train["label"], refit_params)
        val_proba = model.predict_proba(val[FEATURES])[:, 1]
        val_pr_auc = float(average_precision_score(val["label"], val_proba))
        candidates.append(
            {
                "rank_by_cv": rank,
                "params": params,
                "mean_best_iter": cand["mean_best_iter"],
                "val_pr_auc": val_pr_auc,
                "_model": model,
                "_val_proba": val_proba,
            }
        )
    winner = max(candidates, key=lambda c: c["val_pr_auc"])
    final_params = {**winner["params"], "n_estimators": int(winner["mean_best_iter"])}
    print(
        "final selection on val 2025-H1 PR-AUC: "
        + ", ".join(f"#{c['rank_by_cv']}={c['val_pr_auc']:.4f}" for c in candidates)
        + f" -> winner #{winner['rank_by_cv']}"
    )

    # 3) Logistic baseline (fit on train only, same windows).
    logistic = make_logistic_pipeline()
    logistic.fit(train[LOGISTIC_FEATURES], train["label"])
    val_log_proba = logistic.predict_proba(val[LOGISTIC_FEATURES])[:, 1]

    # 4) Isotonic calibrator on the winner's out-of-sample val predictions.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(winner["_val_proba"], val["label"].to_numpy())
    val_cal_proba = calibrator.transform(winner["_val_proba"])

    # 5) Fold metrics for all three models. LGBM folds re-run the
    #    winner's ORIGINAL trial params (its n_estimators cap + early
    #    stopping on each fold's valid window) — the same procedure the
    #    trial was scored with; the shipped model alone uses the
    #    mean-early-stopped refit (no eval window).
    fold_metrics: dict[str, list[dict]] = {
        "persistence": [],
        "logistic": [],
        "lgbm": [],
    }
    for tr, va in folds:
        y_va = va["label"].to_numpy()
        fold_metrics["persistence"].append(
            _score_block(y_va, va["_persistence"].to_numpy())
        )
        pipe = make_logistic_pipeline()
        pipe.fit(tr[LOGISTIC_FEATURES], tr["label"])
        fold_metrics["logistic"].append(
            _score_block(y_va, pipe.predict_proba(va[LOGISTIC_FEATURES])[:, 1])
        )
        model = fit_lgbm(
            tr[FEATURES], tr["label"], winner["params"],
            va[FEATURES], va["label"],
        )
        fold_metrics["lgbm"].append(
            _score_block(y_va, model.predict_proba(va[FEATURES])[:, 1])
        )

    # 6) Test: the single touch of 2025-H2.
    test_y = test["label"].to_numpy()
    test_persistence = test["_persistence"].to_numpy()
    test_log_proba = logistic.predict_proba(test[LOGISTIC_FEATURES])[:, 1]
    test_raw = winner["_model"].predict_proba(test[FEATURES])[:, 1]
    test_cal = calibrator.transform(test_raw)

    val_y = val["label"].to_numpy()
    metrics: dict = {
        "data": {
            "path": _repo_path(DATA_PATH),
            "n_rows": len(df),
            "positive_rate": float(df["label"].mean()),
            "n_features": len(FEATURES),
        },
        "splits": split_block,
        "cv": {
            "n_splits": n_splits,
            "min_train_days": 365,
            "purge_rule": (
                "max(fold_train.label_hour) < min(fold_valid.hour_key)"
            ),
            "folds": fold_rows,
        },
        "optuna": {
            "n_trials": tune["n_trials"],
            "objective": tune["objective"],
            "best_cv_pr_auc": tune["best_cv_pr_auc"],
            "best_params": tune["best_params"],
            "search_space": tune["search_space"],
            "seconds": round(tune_seconds, 1),
            "final_selection": {
                "rule": (
                    "top-5 trials by CV PR-AUC refit on train, ranked by "
                    "PR-AUC on val 2025-H1"
                ),
                "candidates": [
                    {k: v for k, v in c.items() if not k.startswith("_")}
                    for c in candidates
                ],
                "winner_rank": winner["rank_by_cv"],
                "winner_refit_n_estimators": final_params["n_estimators"],
            },
        },
        "calibration": {
            "method": "isotonic",
            "fit_on": (
                "val 2025-H1 predictions from the train-fit winning model "
                "(out-of-fold with respect to the calibration data)"
            ),
        },
        "val_2025_h1": {
            "base_rate": float(val_y.mean()),
            "n": len(val),
            "persistence": _score_block(val_y, val["_persistence"].to_numpy()),
            "logistic": _score_block(val_y, val_log_proba),
            "lightgbm_raw": _score_block(val_y, winner["_val_proba"]),
            "lightgbm_calibrated": _score_block(val_y, val_cal_proba),
        },
        "test_2025_h2": {
            "base_rate": float(test_y.mean()),
            "n": len(test),
            "persistence": _score_block(test_y, test_persistence),
            "logistic": _score_block(test_y, test_log_proba),
            "lightgbm_raw": _score_block(test_y, test_raw),
            "lightgbm_calibrated": _score_block(test_y, test_cal),
        },
    }
    for model_name, per_fold in fold_metrics.items():
        metrics[f"{model_name}_cv"] = _cv_summary(per_fold)
        metrics[f"{model_name}_cv_folds"] = per_fold

    # Flat mean±std keys (brief interface: persistence_auc / logistic_auc /
    # lgbm_auc ... pr_auc, brier, ece per fold as mean ± std).
    for model_name in ("persistence", "logistic", "lgbm"):
        for metric in ("roc_auc", "pr_auc", "brier", "ece"):
            block = metrics[f"{model_name}_cv"][metric]
            metrics[f"{model_name}_{metric}_mean"] = block["mean"]
            metrics[f"{model_name}_{metric}_std"] = block["std"]

    # Artifacts.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lightgbm.txt"
    winner["_model"].booster_.save_model(str(model_path))
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    joblib.dump(calibrator, calibrator_path)
    logistic_path = MODELS_DIR / "logistic.pkl"
    joblib.dump(logistic, logistic_path)
    figure_path = plot_reliability(
        test_y, test_raw, test_cal, FIGURES_DIR / "reliability_diagram.png"
    )
    metrics["artifacts"] = {
        "lightgbm": _repo_path(model_path),
        "calibrator": _repo_path(calibrator_path),
        "logistic": _repo_path(logistic_path),
        "reliability_diagram": _repo_path(figure_path),
        "metrics_json": _repo_path(METRICS_PATH),
    }
    metrics["runtime_seconds"] = round(time.time() - t_start, 1)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=False)

    print("cv mean±std (folds over train period):")
    for model_name in ("persistence", "logistic", "lgbm"):
        cv = metrics[f"{model_name}_cv"]
        print(
            f"  {model_name:>11}: roc_auc={cv['roc_auc']['mean']:.4f}"
            f"±{cv['roc_auc']['std']:.4f} "
            f"pr_auc={cv['pr_auc']['mean']:.4f}"
            f"±{cv['pr_auc']['std']:.4f} "
            f"brier={cv['brier']['mean']:.4f}"
            f"±{cv['brier']['std']:.4f} "
            f"ece={cv['ece']['mean']:.4f}±{cv['ece']['std']:.4f}"
        )
    print("test 2025-H2 (base rate "
          f"{metrics['test_2025_h2']['base_rate']:.4f}):")
    for key in ("persistence", "logistic", "lightgbm_raw", "lightgbm_calibrated"):
        b = metrics["test_2025_h2"][key]
        print(
            f"  {key:>20}: roc_auc={b['roc_auc']:.4f} pr_auc={b['pr_auc']:.4f} "
            f"brier={b['brier']:.4f} ece={b['ece']:.4f} n={b['n_scored']}"
        )
    print(f"saved: {model_path}")
    print(f"saved: {calibrator_path}")
    print(f"saved: {logistic_path}")
    print(f"saved: {figure_path}")
    print(f"saved: {METRICS_PATH}")
    print(f"total runtime: {metrics['runtime_seconds']}s")
    return metrics


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    train_all(df)


if __name__ == "__main__":
    main()
