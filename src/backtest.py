"""Rolling-origin backtest: forecasts and replenishment over four test weeks.

The main analysis (src/forecast.py, src/inventory_sim.py) uses one test week,
the evaluation week after the last training day (origin = day 90). This script
adds three earlier test weeks inside the training period, with forecast
origins at days 69, 76 and 83 (test weeks: days 70-76, 77-83, 84-90), and
combines them with the evaluation week into a four-week rolling-origin
evaluation.

For every backtest origin, nothing after the origin is used for training:
    * recovered demand used as a training target is recomputed from data up to
      the origin only (intraday profiles and fallback medians), and
    * all models (seasonal naive, Empirical-28, ETS, LightGBM) are fitted on
      data up to the origin, exactly as in src/forecast.py.
The truth for evaluation set A is the recovered demand of the main analysis
(the best available estimate); set B uses stockout-free days, where recorded
sales equal demand. Results of the evaluation week are taken from
data/processed/forecasts_eval.parquet, so src/forecast.py must have been run.

The script is resumable: forecasts of each origin are saved to
data/processed/backtest/ and skipped if they already exist.

Usage
    python src/backtest.py
    python src/backtest.py --sample-series 2000   # quick check

Outputs
    results/tables/T17_backtest_accuracy.csv
    results/tables/T18_backtest_summary.csv
    results/tables/T19_backtest_inventory.csv
    results/figures/F12_backtest.png
"""
import argparse
import gc
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_PROCESSED, FIGURES, TABLES, INK_2, set_plot_style, set_seed
from data_utils import KEYS, load_panel
from forecast import (QUANTILES, QCOLS, HORIZON, N_TRAIN_DAYS, STATIC, DAY_COVARIATES,
                      LGB_PARAMS, MAX_ROUNDS, EARLY_STOP, MODEL_ORDER, MODEL_COLORS,
                      TARGET_LABELS, make_rows, feature_columns, pinball)
from inventory_sim import simulate, quantile_at
from recover_demand import OPEN, build_profiles, rolling_median, recover

BACKTEST_ORIGINS = [68, 75, 82]          # 0-based index of the last observed day
EVAL_ORIGIN = N_TRAIN_DAYS - 1           # 89: the main evaluation week
OUT_DIR = DATA_PROCESSED / "backtest"
UNIT_COST = 0.5
SHELF_LIVES = [1, 2, 3]
ORDER_LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_panel(sample_series):
    """Panel arrays plus origin-restricted recovered demand for each backtest origin."""
    df, hs, hst, T = load_panel(sample_series)
    df = df.drop(columns=["split"])          # text column not needed here; saves memory
    gc.collect()
    n_s = len(df) // T
    sidx = np.repeat(np.arange(n_s), T)
    s2c = pd.factorize(df["third_category_id"].to_numpy()[::T])[0].astype(np.int64)
    n_c = int(s2c.max()) + 1
    day = np.tile(np.arange(T), n_s)

    H = hs.sum(axis=1).astype(float)
    sale = df["sale_amount"].to_numpy(dtype=float)
    M = (hst == 1) & OPEN
    del hst
    gc.collect()
    cens = M.any(axis=1)
    X = (hs * ~M).sum(axis=1).astype(float)
    ratio = np.where(H > 0, sale / np.maximum(H, 1e-12), 1.0)

    def masked_share(P, chunk=400_000):
        """sum over stockout hours of the profile share, computed in chunks to save memory."""
        out = np.empty(len(M))
        for i in range(0, len(M), chunk):
            out[i:i + chunk] = (P[sidx[i:i + chunk]] * M[i:i + chunk]).sum(axis=1)
        return out

    rec_by_origin = {}
    for o in BACKTEST_ORIGINS:
        upto = day <= o
        P = build_profiles(hs, sidx, s2c, upto & ~cens & (H > 0), n_s, n_c)["shrunk"]
        share_in = 1 - masked_share(P)
        fb = rolling_median(np.where(upto & ~cens, sale, np.nan).reshape(n_s, T)).ravel()
        est, _ = recover(X * ratio, sale, share_in, fb)
        rec = np.where(cens, est, sale)
        rec[~upto] = np.nan                       # never used beyond the origin
        rec_by_origin[o] = rec.reshape(n_s, T)
        print(f"  recovered demand up to day {o + 1}: done", flush=True)
    del hs, M, X, P, share_in, fb
    gc.collect()

    main = pd.read_parquet(DATA_PROCESSED / "panel_recovered.parquet")
    main = main.sort_values(KEYS + ["dt"]).reset_index(drop=True)
    if sample_series:
        mask = pd.MultiIndex.from_frame(main[KEYS]).isin(pd.MultiIndex.from_frame(df[KEYS]))
        main = main[mask].reset_index(drop=True)
    if not (main[KEYS].to_numpy() == df[KEYS].to_numpy()).all():
        raise ValueError("Row order of panel_recovered.parquet does not match the raw panel.")

    Pn = lambda a: np.asarray(a, dtype=np.float64).reshape(n_s, T)
    panel = {
        "Y_sales": Pn(sale), "Y_truth": Pn(main["demand"]), "C": Pn(cens),
        "X": {c: Pn(df[c]) for c in DAY_COVARIATES},
        "dates": pd.DatetimeIndex(df["dt"].to_numpy()[:T]),
        "static_cat": df[STATIC].iloc[::T].reset_index(drop=True).astype("category"),
        "keys": df[KEYS].iloc[::T].reset_index(drop=True),
        "n_s": n_s, "T": T, "rec": rec_by_origin,
    }
    return panel


# --------------------------------------------------------------------------- #
# Models at an arbitrary origin
# --------------------------------------------------------------------------- #
def _long(n_s, origin, vals):
    s = np.repeat(np.arange(n_s), HORIZON)
    t = np.tile(np.arange(origin + 1, origin + 1 + HORIZON), n_s)
    return pd.DataFrame({"series": s, "t": t,
                         **{c: vals[:, :, i].ravel() for i, c in enumerate(QCOLS)}})


def seasonal_naive(panel, Y, o):
    pt = Y[:, o - 6:o + 1]
    return _long(panel["n_s"], o, np.repeat(pt[:, :, None], len(QCOLS), axis=2))


def empirical28(panel, Y, o):
    q = np.quantile(Y[:, o - 27:o + 1], QUANTILES, axis=1).T
    return _long(panel["n_s"], o, np.repeat(q[:, None, :], HORIZON, axis=1))


def ets(panel, Y, o):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    n_s = panel["n_s"]
    df = pd.DataFrame({"unique_id": np.repeat(np.arange(n_s), o + 1),
                       "ds": np.tile(panel["dates"][:o + 1], n_s),
                       "y": Y[:, :o + 1].ravel()})
    sf = StatsForecast(models=[AutoETS(season_length=7, model="ANA")], freq="D", n_jobs=-1)
    f = sf.forecast(df=df, h=HORIZON, level=[40, 80])
    if "unique_id" not in f.columns:
        f = f.reset_index()
    f = f.sort_values(["unique_id", "ds"])
    cols = ["AutoETS-lo-80", "AutoETS-lo-40", "AutoETS", "AutoETS-hi-40", "AutoETS-hi-80"]
    vals = np.maximum(f[cols].to_numpy(), 0).reshape(n_s, HORIZON, len(cols))
    return _long(n_s, o, vals)


def lightgbm(panel, Y, o):
    """Same procedure as forecast.fit_lightgbm, with forecast origin o."""
    p = {"n_s": panel["n_s"], "C": panel["C"], "X": panel["X"], "dates": panel["dates"],
         "static_cat": panel["static_cat"]}
    origins = list(range(o - HORIZON, 26, -HORIZON))[::-1]
    tr = pd.concat([make_rows(p, Y, x, True) for x in origins[:-1]], ignore_index=True)
    va = make_rows(p, Y, origins[-1], True)
    feats = feature_columns(tr)
    dtr = lgb.Dataset(tr[feats], tr["label"])
    dva = lgb.Dataset(va[feats], va["label"], reference=dtr)
    m = lgb.train({**LGB_PARAMS, "alpha": 0.5}, dtr, MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    n_rounds = max(m.best_iteration, 50)
    del m, dtr, dva
    full = pd.concat([tr, va], ignore_index=True)
    del tr, va
    gc.collect()
    dfull = lgb.Dataset(full[feats], full["label"]).construct()
    del full
    gc.collect()
    test = make_rows(p, Y, o, False)
    preds = {}
    for q, col in zip(QUANTILES, QCOLS):
        mq = lgb.train({**LGB_PARAMS, "alpha": q}, dfull, n_rounds)
        preds[col] = np.maximum(mq.predict(test[feats]), 0) * test["scale"].to_numpy()
    del dfull
    gc.collect()
    return pd.DataFrame({"series": test["series"], "t": test["t"], **preds}), n_rounds


def run_origin(panel, o):
    out = OUT_DIR / f"forecasts_origin{o + 1}.parquet"
    if out.exists():
        print(f"Origin day {o + 1}: found {out.name}, skipping")
        return
    print(f"Origin day {o + 1} (test days {o + 2}-{o + 8}):")
    parts = []
    for version in ["sales", "recovered"]:
        Y = panel["Y_sales"] if version == "sales" else panel["rec"][o]
        for name, fn in [("Seasonal naive", seasonal_naive), ("Empirical-28", empirical28),
                         ("ETS", ets), ("LightGBM", lightgbm)]:
            t0 = time.time()
            res = fn(panel, Y, o)
            f, extra = (res if isinstance(res, tuple) else (res, None))
            f["model"], f["version"] = name, version
            parts.append(f)
            gc.collect()
            msg = f" ({extra} rounds)" if extra else ""
            print(f"  {name:15s} on {version:9s} done in {time.time() - t0:7.1f} s{msg}", flush=True)
    fc = pd.concat(parts, ignore_index=True)
    fc[QCOLS] = np.sort(fc[QCOLS].to_numpy(), axis=1)
    fc["origin_day"] = o + 1
    fc.to_parquet(out, index=False)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def load_all_forecasts(panel):
    frames = []
    for o in BACKTEST_ORIGINS:
        frames.append(pd.read_parquet(OUT_DIR / f"forecasts_origin{o + 1}.parquet"))
    ev = pd.read_parquet(DATA_PROCESSED / "forecasts_eval.parquet")
    keys = ev.drop_duplicates("series").sort_values("series")[KEYS].reset_index(drop=True)
    if len(keys) == panel["n_s"] and (keys.to_numpy() == panel["keys"].to_numpy()).all():
        ev = ev[["series", "t", *QCOLS, "model", "version"]].copy()
        ev["origin_day"] = EVAL_ORIGIN + 1
        frames.append(ev)
    else:
        print("Note: forecasts_eval.parquet covers different series (sample run); "
              "the evaluation week is left out.")
    fc = pd.concat(frames, ignore_index=True)
    s, t = fc["series"].to_numpy(), fc["t"].to_numpy()
    fc["truth"] = panel["Y_truth"][s, t]
    fc["recorded_sales"] = panel["Y_sales"][s, t]
    fc["censored"] = panel["C"][s, t].astype(bool)
    return fc


def week_label(panel, origin_day):
    d = panel["dates"]
    return f"{d[origin_day]:%d %b}–{d[origin_day + HORIZON - 1]:%d %b}"


def accuracy(fc, panel):
    rows, tests = [], []
    for (od, model, version), g in fc.groupby(["origin_day", "model", "version"], sort=True):
        for set_name, mask in [("A", np.ones(len(g), bool)), ("B", ~g["censored"].to_numpy())]:
            d = g[mask]
            y, med = d["truth"].to_numpy(), d["q50"].to_numpy()
            pl = pinball(y, d[QCOLS].to_numpy())
            point = model == "Seasonal naive"
            rows.append({"test_week": week_label(panel, od), "origin_day": od, "eval_set": set_name,
                         "model": model, "trained_on": TARGET_LABELS[version], "rows": len(d),
                         "WAPE": np.abs(med - y).sum() / y.sum(), "bias": (med - y).sum() / y.sum(),
                         "scaled_pinball": np.nan if point else pl.sum() / y.sum(),
                         "coverage_80": np.nan if point else float(((y >= d["q10"]) & (y <= d["q90"])).mean())})
    acc = pd.DataFrame(rows)

    fc = fc.assign(loss=pinball(fc["truth"].to_numpy(), fc[QCOLS].to_numpy()))
    for od, g in fc[fc["model"] == "LightGBM"].groupby("origin_day"):
        for set_name, mask in [("A", slice(None)), ("B", ~g["censored"])]:
            per = g[mask].groupby(["version", "series"])["loss"].mean()
            a, b = per.loc["recovered"].align(per.loc["sales"], join="inner")
            diff = (a - b).to_numpy()
            tests.append({"origin_day": od, "eval_set": set_name,
                          "share_series_recovered_better": float((diff < 0).mean()),
                          "wilcoxon_p": wilcoxon(diff).pvalue})
    return acc, pd.DataFrame(tests)


def summary(acc, tests):
    a = acc[acc["model"] != "Seasonal naive"]
    piv = a.pivot_table(index=["eval_set", "model", "origin_day"], columns="trained_on",
                        values="scaled_pinball")
    piv["sPL_change_pct"] = 100 * (piv["Recovered demand"] / piv["Recorded sales"] - 1)
    rows = []
    for (s, m), g in piv.groupby(level=[0, 1]):
        r = {"eval_set": s, "model": m,
             "mean_sPL_recorded": g["Recorded sales"].mean(),
             "mean_sPL_recovered": g["Recovered demand"].mean(),
             "mean_change_pct": g["sPL_change_pct"].mean(),
             "weeks_recovered_better": int((g["sPL_change_pct"] < 0).sum()),
             "weeks": len(g)}
        if m == "LightGBM":
            t = tests[tests["eval_set"] == s]
            r["mean_share_series_recovered_better"] = t["share_series_recovered_better"].mean()
        rows.append(r)
    best = a.groupby(["eval_set", "origin_day"]).apply(
        lambda g: g.loc[g["scaled_pinball"].idxmin(), ["model", "trained_on"]].str.cat(sep=" / "),
        include_groups=False)
    return pd.DataFrame(rows), best.rename("best_forecast").reset_index()


def inventory(fc, panel):
    rows = []
    n_s = panel["n_s"]
    for od, g in fc.groupby("origin_day"):
        g = g.sort_values(["model", "version", "series", "t"])
        first = g[(g["model"] == "LightGBM") & (g["version"] == "sales")]
        D = {"Recovered demand": first["truth"].to_numpy().reshape(n_s, HORIZON),
             "Recorded sales (lower bound)": first["recorded_sales"].to_numpy().reshape(n_s, HORIZON)}
        for (model, version), h in g.groupby(["model", "version"], sort=False):
            Q = h[QCOLS].to_numpy().reshape(n_s, HORIZON, len(QCOLS))
            for truth, Dm in D.items():
                d_sum = Dm.sum()
                for L in SHELF_LIVES:
                    costs = []
                    for lev in [1 - UNIT_COST, *ORDER_LEVELS]:
                        tot = simulate(Dm, quantile_at(Q, lev), L)
                        costs.append(((1 - UNIT_COST) * tot["lost"].sum() +
                                      UNIT_COST * tot["waste"].sum()) / d_sum)
                    k = int(np.argmin(costs[1:]))
                    rows.append({"test_week": week_label(panel, od), "origin_day": od, "truth": truth,
                                 "shelf_life_days": L, "model": model,
                                 "trained_on": TARGET_LABELS[version],
                                 "cost_at_critical_ratio": costs[0],
                                 "best_order_level": ORDER_LEVELS[k],
                                 "cost_at_best_level": costs[1 + k]})
    return pd.DataFrame(rows)


def fig_backtest(acc, inv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    weeks = sorted(acc["origin_day"].unique())
    labels = [acc.loc[acc["origin_day"] == w, "test_week"].iloc[0] for w in weeks]
    x = np.arange(len(weeks))
    for model in ["Empirical-28", "ETS", "LightGBM"]:
        for version, ls, mk in [("Recovered demand", "-", "o"), ("Recorded sales", "--", "s")]:
            d = acc[(acc["eval_set"] == "B") & (acc["model"] == model) & (acc["trained_on"] == version)]
            d = d.set_index("origin_day").reindex(weeks)
            axes[0].plot(x, d["scaled_pinball"], ls=ls, marker=mk, ms=4, color=MODEL_COLORS[model],
                         label=f"{model} – {version.lower()}")
            c = inv[(inv["truth"] == "Recovered demand") & (inv["shelf_life_days"] == 1) &
                    (inv["model"] == model) & (inv["trained_on"] == version)]
            c = c.set_index("origin_day").reindex(weeks)
            axes[1].plot(x, c["cost_at_critical_ratio"], ls=ls, marker=mk, ms=4, color=MODEL_COLORS[model])
    axes[0].set(ylabel="Scaled pinball loss", title="(a) Forecast loss, stockout-free days")
    axes[1].set(ylabel="Cost per unit of demand", title="(b) Replenishment cost, shelf life 1 day")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Test week")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "F12_backtest.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-series", type=int, default=None)
    args = ap.parse_args()
    set_seed()
    set_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing data and origin-restricted recovered demand ...")
    panel = build_panel(args.sample_series)
    for o in BACKTEST_ORIGINS:
        run_origin(panel, o)

    fc = load_all_forecasts(panel)
    acc, tests = accuracy(fc, panel)
    acc.round(4).to_csv(TABLES / "T17_backtest_accuracy.csv", index=False)
    summ, best = summary(acc, tests)
    summ.round(4).to_csv(TABLES / "T18_backtest_summary.csv", index=False)
    tests.to_csv(TABLES / "T18b_backtest_lightgbm_tests.csv", index=False)
    best.to_csv(TABLES / "T18c_backtest_best_forecast.csv", index=False)
    inv = inventory(fc, panel)
    inv.round(4).to_csv(TABLES / "T19_backtest_inventory.csv", index=False)
    fig_backtest(acc, inv)

    pd.set_option("display.width", 220)
    print("\nMean over the four test weeks (scaled pinball loss):")
    print(summ.round(4).to_string(index=False))
    print("\nBest forecast per week:")
    print(best.to_string(index=False))
    print("\nLightGBM, recovered vs recorded sales (per week):")
    print(tests.round(4).to_string(index=False))
    lg = inv[(inv["model"] == "LightGBM")].pivot_table(
        index=["truth", "shelf_life_days", "origin_day"], columns="trained_on",
        values=["cost_at_critical_ratio", "cost_at_best_level"])
    for col in ["cost_at_critical_ratio", "cost_at_best_level"]:
        lg[(col, "change_pct")] = 100 * (lg[(col, "Recovered demand")] / lg[(col, "Recorded sales")] - 1)
    print("\nLightGBM cost change, recovered vs recorded sales (%):")
    print(lg.xs("change_pct", axis=1, level=1).round(2).to_string())
    print(f"\nTables -> {TABLES}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
