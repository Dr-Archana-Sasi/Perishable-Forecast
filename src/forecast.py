"""Step 4: probabilistic forecasts of daily demand for the evaluation week.

Every model is trained twice: on recorded sales (standard practice) and on the
recovered demand from Step 3 (proposed). Forecasts are made at one origin, the
last training day (2024-06-25), for horizons 1-7 (2024-06-26 to 2024-07-02).

Models
    Seasonal naive  value on the same weekday one week earlier (point forecast)
    Empirical-28    quantiles of the last 28 days (common retail baseline)
    ETS             exponential smoothing, additive error and weekly season
                    (ETS(A,N,A)), fitted per series with statsforecast
    LightGBM        one global quantile-regression model per quantile, trained
                    on rolling weekly origins with lagged demand, calendar,
                    promotion, weather and product/store features

Quantiles forecast: 0.1, 0.3, 0.5, 0.7, 0.9.

Evaluation (true demand is only observable without a stockout)
    A  all evaluation days, truth = recovered demand
    B  stockout-free evaluation days, truth = recorded sales (exact)

Assumption: covariates on the target day (discount, holiday, promotion,
weather) are treated as known in advance. Promotions and holidays are planned;
for weather the observed values stand in for a weather forecast.

Usage
    python src/forecast.py
    python src/forecast.py --sample-series 5000   # quick check

Outputs
    data/processed/forecasts_eval.parquet
    results/tables/T8_forecast_accuracy.csv
    results/tables/T9_forecast_significance.csv
    results/tables/T10_lightgbm_feature_importance.csv
    results/figures/F8_accuracy_by_horizon.png
"""
import argparse
import gc
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_PROCESSED, FIGURES, TABLES, SEED, INK_2, set_plot_style, set_seed
from data_utils import KEYS

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]
HORIZON = 7
N_TRAIN_DAYS = 90
EPS = 0.05                      # added to the 28-day mean when scaling series

TARGETS = {"sales": "sale_amount", "recovered": "demand"}
TARGET_LABELS = {"sales": "Recorded sales", "recovered": "Recovered demand"}
MODEL_ORDER = ["Seasonal naive", "Empirical-28", "ETS", "LightGBM"]
MODEL_COLORS = {"Seasonal naive": "#8a8984", "Empirical-28": "#eda100",
                "ETS": "#1baf7a", "LightGBM": "#2a78d6"}

STATIC = ["city_id", "store_id", "management_group_id", "first_category_id",
          "second_category_id", "third_category_id", "product_id"]
DAY_COVARIATES = ["discount", "holiday_flag", "activity_flag", "precpt",
                  "avg_temperature", "avg_humidity", "avg_wind_level"]

LGB_PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
                  seed=SEED, deterministic=True, force_row_wise=True)
MAX_ROUNDS = 1000
EARLY_STOP = 50


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def load(sample_series):
    df = pd.read_parquet(DATA_PROCESSED / "panel_recovered.parquet")
    df = df.sort_values(KEYS + ["dt"]).reset_index(drop=True)
    if sample_series:
        series = df[KEYS].drop_duplicates().reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        keep = series.iloc[np.sort(rng.choice(len(series), sample_series, replace=False))]
        mask = pd.MultiIndex.from_frame(df[KEYS]).isin(pd.MultiIndex.from_frame(keep))
        df = df[mask].reset_index(drop=True)
    T = int(df.groupby(KEYS).size().iloc[0])
    n_s = len(df) // T
    if T != N_TRAIN_DAYS + HORIZON:
        raise ValueError(f"Expected {N_TRAIN_DAYS + HORIZON} days per series, found {T}")

    P = lambda c: df[c].to_numpy(dtype=np.float64).reshape(n_s, T)
    panel = {
        "Y": {v: P(c) for v, c in TARGETS.items()},
        "C": P("censored"),
        "X": {c: P(c) for c in DAY_COVARIATES},
        "dates": pd.DatetimeIndex(df["dt"].to_numpy()[:T]),
        "static": df[STATIC].iloc[::T].reset_index(drop=True),
        "keys": df[KEYS].iloc[::T].reset_index(drop=True),
        "n_s": n_s, "T": T,
    }
    return panel


# --------------------------------------------------------------------------- #
# Features for the global model
# --------------------------------------------------------------------------- #
def make_rows(panel, Y, origin: int, with_label: bool) -> pd.DataFrame:
    """One row per series and horizon for forecast origin `origin` (last observed day)."""
    n_s, C, X, dates = panel["n_s"], panel["C"], panel["X"], panel["dates"]
    win = Y[:, origin - 27:origin + 1]
    sc = win.mean(axis=1) + EPS
    base = {
        "log_scale": np.log(sc),
        "mean7": Y[:, origin - 6:origin + 1].mean(axis=1) / sc,
        "mean14": Y[:, origin - 13:origin + 1].mean(axis=1) / sc,
        "std28": win.std(axis=1) / sc,
        "zero_share28": (win == 0).mean(axis=1),
        "stockout_share28": C[:, origin - 27:origin + 1].mean(axis=1),
        "discount_mean28": X["discount"][:, origin - 27:origin + 1].mean(axis=1),
    }
    for k in range(7):
        base[f"lag{k}"] = Y[:, origin - k] / sc

    frames = []
    for h in range(1, HORIZON + 1):
        t = origin + h
        f = dict(base)
        f["horizon"] = np.full(n_s, h)
        f["dow"] = np.full(n_s, dates[t].dayofweek)
        for j in range(1, 5):
            f[f"same_dow_lag{j}"] = Y[:, t - 7 * j] / sc
        for c in DAY_COVARIATES:
            f[c] = X[c][:, t]
        f["series"] = np.arange(n_s)
        f["t"] = np.full(n_s, t)
        f["scale"] = sc
        if with_label:
            f["label"] = Y[:, t] / sc
        fr = pd.DataFrame(f)
        num = [c for c in fr.columns if c not in ("series", "t", "horizon", "dow")]
        fr[num] = fr[num].astype(np.float32)
        frames.append(fr)
    rows = pd.concat(frames, ignore_index=True)
    st = panel["static_cat"].iloc[rows["series"].to_numpy()].reset_index(drop=True)
    return pd.concat([rows, st], axis=1)


def feature_columns(rows):
    return [c for c in rows.columns if c not in ("series", "t", "scale", "label")]


def fit_lightgbm(panel, version):
    Y = panel["Y"][version]
    last = N_TRAIN_DAYS - 1                                    # index of last training day
    origins = list(range(last - HORIZON, 26, -HORIZON))[::-1]  # weekly origins, oldest first
    val_origin = origins[-1]
    tr = pd.concat([make_rows(panel, Y, o, True) for o in origins[:-1]], ignore_index=True)
    va = make_rows(panel, Y, val_origin, True)
    feats = feature_columns(tr)

    # 1) choose the number of boosting rounds on the last training week (median model)
    dtr = lgb.Dataset(tr[feats], tr["label"])
    dva = lgb.Dataset(va[feats], va["label"], reference=dtr)
    m = lgb.train({**LGB_PARAMS, "alpha": 0.5}, dtr, MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    n_rounds = max(m.best_iteration, 50)
    del m, dtr, dva
    gc.collect()

    # 2) refit every quantile on all origins
    full = pd.concat([tr, va], ignore_index=True)
    n_full = len(full)
    del tr, va
    gc.collect()
    dfull = lgb.Dataset(full[feats], full["label"]).construct()
    del full
    gc.collect()
    test = make_rows(panel, Y, last, False)
    preds, importance = {}, None
    for q, col in zip(QUANTILES, QCOLS):
        mq = lgb.train({**LGB_PARAMS, "alpha": q}, dfull, n_rounds)
        preds[col] = np.maximum(mq.predict(test[feats]), 0) * test["scale"].to_numpy()
        if q == 0.5:
            importance = pd.DataFrame({"feature": feats,
                                       "gain": mq.feature_importance("gain")})
    del dfull
    gc.collect()
    info = {"origins": len(origins), "train_rows": n_full, "rounds": n_rounds}
    out = pd.DataFrame({"series": test["series"], "t": test["t"], **preds})
    return out, importance, info


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _long(panel, values_by_h, cols):
    """values_by_h: array (n_s, H, len(cols)) -> long DataFrame."""
    n_s = panel["n_s"]
    s = np.repeat(np.arange(n_s), HORIZON)
    t = np.tile(np.arange(N_TRAIN_DAYS, N_TRAIN_DAYS + HORIZON), n_s)
    return pd.DataFrame({"series": s, "t": t,
                         **{c: values_by_h[:, :, i].ravel() for i, c in enumerate(cols)}})


def seasonal_naive(panel, Y):
    pt = Y[:, N_TRAIN_DAYS - 7:N_TRAIN_DAYS]                    # (n_s, 7)
    return _long(panel, np.repeat(pt[:, :, None], len(QCOLS), axis=2), QCOLS)


def empirical28(panel, Y):
    q = np.quantile(Y[:, N_TRAIN_DAYS - 28:N_TRAIN_DAYS], QUANTILES, axis=1).T  # (n_s, 5)
    return _long(panel, np.repeat(q[:, None, :], HORIZON, axis=1), QCOLS)


def ets(panel, Y):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    n_s = panel["n_s"]
    dates = panel["dates"][:N_TRAIN_DAYS]
    df = pd.DataFrame({"unique_id": np.repeat(np.arange(n_s), N_TRAIN_DAYS),
                       "ds": np.tile(dates, n_s),
                       "y": Y[:, :N_TRAIN_DAYS].ravel()})
    sf = StatsForecast(models=[AutoETS(season_length=7, model="ANA")], freq="D", n_jobs=-1)
    f = sf.forecast(df=df, h=HORIZON, level=[40, 80])
    if "unique_id" not in f.columns:
        f = f.reset_index()
    f = f.sort_values(["unique_id", "ds"])
    cols = ["AutoETS-lo-80", "AutoETS-lo-40", "AutoETS", "AutoETS-hi-40", "AutoETS-hi-80"]
    vals = np.maximum(f[cols].to_numpy(), 0).reshape(n_s, HORIZON, len(cols))
    return _long(panel, vals, QCOLS)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def pinball(y, qhat):
    """Mean pinball loss over the quantile columns, per row."""
    losses = [np.maximum(q * (y - qhat[:, i]), (q - 1) * (y - qhat[:, i]))
              for i, q in enumerate(QUANTILES)]
    return np.mean(losses, axis=0)


def evaluate(fc):
    rows = []
    sets = {"A: all days vs recovered demand": np.ones(len(fc), bool),
            "B: stockout-free days vs recorded sales": ~fc["censored"].to_numpy()}
    for (model, version), g in fc.groupby(["model", "version"], sort=False):
        for set_name, mask_all in sets.items():
            m = mask_all[g.index]
            d = g[m]
            y, med = d["truth"].to_numpy(), d["q50"].to_numpy()
            pl = pinball(y, d[QCOLS].to_numpy())
            point_only = model == "Seasonal naive"
            rows.append({
                "eval_set": set_name, "model": model, "trained_on": TARGET_LABELS[version],
                "rows": len(d),
                "WAPE": np.abs(med - y).sum() / y.sum(),
                "bias": (med - y).sum() / y.sum(),
                "scaled_pinball": np.nan if point_only else pl.sum() / y.sum(),
                "coverage_80": np.nan if point_only else
                    float(((y >= d["q10"]) & (y <= d["q90"])).mean()),
            })
    return pd.DataFrame(rows)


def significance(fc):
    """Paired Wilcoxon signed-rank tests over series (mean pinball loss per series)."""
    pairs = [(("LightGBM", "recovered"), ("LightGBM", "sales")),
             (("ETS", "recovered"), ("ETS", "sales")),
             (("Empirical-28", "recovered"), ("Empirical-28", "sales")),
             (("LightGBM", "recovered"), ("ETS", "recovered")),
             (("LightGBM", "recovered"), ("Empirical-28", "recovered"))]
    fc = fc.assign(loss=pinball(fc["truth"].to_numpy(), fc[QCOLS].to_numpy()))
    out = []
    for set_name, mask in [("A: all days vs recovered demand", slice(None)),
                           ("B: stockout-free days vs recorded sales", ~fc["censored"])]:
        per = fc[mask].groupby(["model", "version", "series"])["loss"].mean()
        for a, b in pairs:
            la, lb = per.loc[a], per.loc[b]
            la, lb = la.align(lb, join="inner")
            diff = (la - lb).to_numpy()
            stat = wilcoxon(diff, zero_method="wilcox")
            out.append({"eval_set": set_name,
                        "model_A": f"{a[0]} ({TARGET_LABELS[a[1]]})",
                        "model_B": f"{b[0]} ({TARGET_LABELS[b[1]]})",
                        "series": len(diff),
                        "mean_loss_A": la.mean(), "mean_loss_B": lb.mean(),
                        "share_series_A_better": float((diff < 0).mean()),
                        "wilcoxon_p": stat.pvalue})
    return pd.DataFrame(out)


def fig_by_horizon(fc):
    d = fc.copy()
    d["h"] = d["t"] - N_TRAIN_DAYS + 1
    d["abs_err"] = (d["q50"] - d["truth"]).abs()
    g = d.groupby(["model", "version", "h"]).agg(ae=("abs_err", "sum"), y=("truth", "sum"))
    g["WAPE"] = g["ae"] / g["y"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model in MODEL_ORDER:
        for version, ls, mk in [("recovered", "-", "o"), ("sales", "--", "s")]:
            if (model, version) not in g.index.droplevel(2):
                continue
            s = g.loc[(model, version), "WAPE"]
            ax.plot(s.index, s.values, ls=ls, marker=mk, ms=4, color=MODEL_COLORS[model],
                    label=f"{model} - {TARGET_LABELS[version].lower()}")
    ax.set(xlabel="Forecast horizon (days ahead)", ylabel="WAPE (truth: recovered demand)",
           title="Forecast error by horizon, evaluation week")
    ax.set_xticks(range(1, HORIZON + 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "F8_accuracy_by_horizon.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-series", type=int, default=None)
    args = ap.parse_args()
    set_seed()
    set_plot_style()

    panel = load(args.sample_series)
    panel["static_cat"] = panel["static"].astype("category")
    n_s = panel["n_s"]
    print(f"Panel: {n_s:,} series x {panel['T']} days "
          f"(train {N_TRAIN_DAYS}, evaluation {HORIZON})")

    forecasts, importances, info_rows = [], [], []
    for version in TARGETS:
        Y = panel["Y"][version]
        for name, fn in [("Seasonal naive", seasonal_naive), ("Empirical-28", empirical28),
                         ("ETS", ets)]:
            t0 = time.time()
            f = fn(panel, Y)
            f["model"], f["version"] = name, version
            forecasts.append(f)
            gc.collect()
            print(f"  {name:15s} on {version:9s} done in {time.time() - t0:6.1f} s")
        t0 = time.time()
        f, imp, info = fit_lightgbm(panel, version)
        f["model"], f["version"] = "LightGBM", version
        forecasts.append(f)
        imp["trained_on"] = TARGET_LABELS[version]
        importances.append(imp)
        info_rows.append({"trained_on": TARGET_LABELS[version], **info})
        print(f"  {'LightGBM':15s} on {version:9s} done in {time.time() - t0:6.1f} s "
              f"({info['rounds']} rounds, {info['train_rows']:,} training rows)")

    fc = pd.concat(forecasts, ignore_index=True)
    fc[QCOLS] = np.sort(fc[QCOLS].to_numpy(), axis=1)        # remove quantile crossing
    s, t = fc["series"].to_numpy(), fc["t"].to_numpy()
    fc["truth"] = panel["Y"]["recovered"][s, t]
    fc["recorded_sales"] = panel["Y"]["sales"][s, t]
    fc["censored"] = panel["C"][s, t].astype(bool)
    fc["dt"] = panel["dates"][t]
    fc = pd.concat([panel["keys"].iloc[s].reset_index(drop=True), fc], axis=1)
    fc.to_parquet(DATA_PROCESSED / "forecasts_eval.parquet", index=False)

    acc = evaluate(fc)
    acc.round(4).to_csv(TABLES / "T8_forecast_accuracy.csv", index=False)
    sig = significance(fc)
    sig.to_csv(TABLES / "T9_forecast_significance.csv", index=False)
    imp = pd.concat(importances, ignore_index=True)
    imp["gain_share"] = imp["gain"] / imp.groupby("trained_on")["gain"].transform("sum")
    imp.sort_values(["trained_on", "gain_share"], ascending=[True, False]).round(4).to_csv(
        TABLES / "T10_lightgbm_feature_importance.csv", index=False)
    pd.DataFrame(info_rows).to_csv(TABLES / "T10b_lightgbm_training_info.csv", index=False)
    fig_by_horizon(fc)

    pd.set_option("display.width", 200)
    print("\nForecast accuracy (evaluation week):")
    print(acc.round(4).to_string(index=False))
    print("\nPaired Wilcoxon tests over series (mean pinball loss):")
    print(sig[["eval_set", "model_A", "model_B", "share_series_A_better", "wilcoxon_p"]]
          .to_string(index=False))
    print(f"\nTables -> {TABLES}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
