"""Robustness of demand recovery: alternative methods and parameter sensitivity.

Uses exactly the same held-out validation days and imposed stockout patterns as
src/recover_demand.py (same seed and the same sequence of random draws), so the
numbers are directly comparable with Tables T5 and T6.

Alternative recovery methods compared with the proposed shrunk profile
    Rolling median     median of the series' stockout-free days within +/-14 days
    Weekday mean       mean of the series' stockout-free days on the same weekday
    Uniform-hour       projection assuming demand is spread evenly over the
                       opening hours, i.e. using only the time of the stockout
    Global / category / series profiles (as in Table T5)

Sensitivity of the proposed method
    k (shrinkage pseudo-days), tau (minimum in-stock profile share for
    projection) and the half-width of the rolling-median fallback window.

Usage
    python src/recovery_robustness.py
    python src/recovery_robustness.py --sample-series 5000   # quick check

Outputs
    results/tables/T15_recovery_method_comparison.csv
    results/tables/T16_recovery_sensitivity.csv
    results/figures/F11_recovery_methods.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, TABLES, SEED, INK_2, set_plot_style, set_seed
from data_utils import load_panel
from recover_demand import (OPEN, VAL_FRAC, K_SHRINK, TAU, HALF_WINDOW, SEVERITY_BINS,
                            SEVERITY_LABELS, build_profiles, rolling_median, recover,
                            wape, bias)

LABELS = {
    "observed": "Observed sales (no recovery)",
    "rolling_median": "Rolling median of stockout-free days",
    "weekday_mean": "Weekday mean of stockout-free days",
    "uniform": "Uniform-hour projection (stockout time only)",
    "global": "Global profile",
    "category": "Category profile",
    "series": "Series profile",
    "shrunk": "Shrunk series profile (proposed)",
}
COLORS = {"observed": "#8a8984", "rolling_median": "#e87ba4", "weekday_mean": "#4a3aa7",
          "uniform": "#eda100", "shrunk": "#2a78d6"}


def setup(sample_series):
    """Reproduce the validation split of recover_demand.py exactly."""
    df, hs, hst, T = load_panel(sample_series)
    n_s = len(df) // T
    sidx = np.repeat(np.arange(n_s), T)
    s2c = pd.factorize(df["third_category_id"].to_numpy()[::T])[0].astype(np.int64)
    n_c = int(s2c.max()) + 1
    rng = np.random.default_rng(SEED)            # same seed and draw order as recover_demand

    train = (df["split"] == "train").to_numpy()
    H = hs.sum(axis=1).astype(float)
    cens = (hst[:, OPEN] == 1).any(axis=1)
    clean = train & ~cens
    val = clean & (H > 0) & (rng.random(len(df)) < VAL_FRAC)
    est_days = clean & ~val
    vi = np.flatnonzero(val)
    donors = np.flatnonzero(train & cens)
    M = np.zeros((len(vi), 24), dtype=bool)
    M[:, OPEN] = hst[rng.choice(donors, size=len(vi))][:, OPEN] == 1

    dow = df["dt"].dt.dayofweek.to_numpy()
    return dict(df=df, hs=hs, T=T, n_s=n_s, n_c=n_c, sidx=sidx, s2c=s2c, H=H,
                est_days=est_days, vi=vi, M=M, dow=dow,
                true=H[vi], X=(hs[vi] * ~M).sum(axis=1).astype(float))


def weekday_mean(S):
    """Mean of stockout-free estimation days per series and weekday."""
    key = S["sidx"] * 7 + S["dow"]
    use = S["est_days"]
    tot = np.bincount(key[use], weights=S["H"][use], minlength=S["n_s"] * 7)
    cnt = np.bincount(key[use], minlength=S["n_s"] * 7)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = tot / cnt
    return m[key[S["vi"]]]                       # NaN where no clean day on that weekday


def estimates(S, k=K_SHRINK, tau=TAU, half=HALF_WINDOW, profiles=None):
    vi, M, X = S["vi"], S["M"], S["X"]
    fb = rolling_median(np.where(S["est_days"], S["H"], np.nan).reshape(S["n_s"], S["T"]),
                        half).ravel()[vi]
    if profiles is None:
        profiles = build_profiles(S["hs"], S["sidx"], S["s2c"],
                                  S["est_days"] & (S["H"] > 0), S["n_s"], S["n_c"], k)
    est = {}
    for name, P in profiles.items():
        share_in = 1 - (P[S["sidx"][vi]] * M).sum(axis=1)
        est[name], can = recover(X, X, share_in, fb, tau)
        if name == "shrunk":
            est["_proj_share"] = float(can.mean())
    return est, fb


def method_comparison(S):
    est, fb = estimates(S)
    proj = est.pop("_proj_share")
    X, M = S["X"], S["M"]
    out = {"observed": X,
           "rolling_median": np.maximum(X, np.where(np.isnan(fb), X, fb)),
           "weekday_mean": np.maximum(X, np.nan_to_num(weekday_mean(S), nan=0.0))}
    uniform_share = 1 - M[:, OPEN].sum(axis=1) / OPEN.sum()
    out["uniform"], _ = recover(X, X, uniform_share, fb)
    out.update(est)

    true = S["true"]
    sev = pd.cut(M.sum(axis=1), SEVERITY_BINS, labels=SEVERITY_LABELS, right=False)
    rows = []
    for key in LABELS:
        e = out[key]
        rows.append({"method": LABELS[key], "method_key": key, "stockout_hours": "all",
                     "days": len(true), "WAPE": wape(e, true), "bias": bias(e, true)})
        for lab in SEVERITY_LABELS:
            s = np.asarray(sev == lab)
            rows.append({"method": LABELS[key], "method_key": key, "stockout_hours": lab,
                         "days": int(s.sum()), "WAPE": wape(e[s], true[s]),
                         "bias": bias(e[s], true[s])})
    return pd.DataFrame(rows), proj


def sensitivity(S):
    grid = ([("k", v, dict(k=v)) for v in [0, 1, 5, 10, 20, 50, 100]] +
            [("tau", v, dict(tau=v)) for v in [0.1, 0.2, 0.3, 0.4, 0.5]] +
            [("fallback_half_window_days", v, dict(half=v)) for v in [7, 14, 28]])
    rows = []
    for param, value, kw in grid:
        est, _ = estimates(S, **kw)
        e = est["shrunk"]
        rows.append({"parameter": param, "value": value,
                     "is_default": (param == "k" and value == K_SHRINK) or
                                   (param == "tau" and value == TAU) or
                                   (param == "fallback_half_window_days" and value == HALF_WINDOW),
                     "WAPE": wape(e, S["true"]), "bias": bias(e, S["true"]),
                     "share_recovered_by_projection": est["_proj_share"]})
        print(f"  {param}={value}: WAPE {rows[-1]['WAPE']:.4f}, bias {rows[-1]['bias']:+.4f}")
    return pd.DataFrame(rows)


def fig_methods(tab):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(SEVERITY_LABELS))
    for key in ["observed", "rolling_median", "weekday_mean", "uniform", "shrunk"]:
        d = tab[(tab["method_key"] == key) & (tab["stockout_hours"] != "all")]
        d = d.set_index("stockout_hours").reindex(SEVERITY_LABELS)
        ax.plot(x, d["WAPE"], marker="s" if key == "observed" else "o", ms=5,
                ls="--" if key == "observed" else "-", color=COLORS[key], label=LABELS[key])
    ax.set_xticks(x, SEVERITY_LABELS)
    ax.set(xlabel="Stockout hours applied (06:00-22:00)", ylabel="WAPE",
           title="Recovery error of alternative methods")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "F11_recovery_methods.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-series", type=int, default=None)
    args = ap.parse_args()
    set_seed()
    set_plot_style()

    S = setup(args.sample_series)
    print(f"Validation days: {len(S['vi']):,}")
    comp, proj = method_comparison(S)
    comp.drop(columns="method_key").round(4).to_csv(TABLES / "T15_recovery_method_comparison.csv",
                                                     index=False)
    fig_methods(comp)
    pd.set_option("display.width", 200)
    print("\nRecovery methods (all validation days):")
    print(comp[comp["stockout_hours"] == "all"][["method", "WAPE", "bias"]].round(4)
          .to_string(index=False))

    print("\nSensitivity of the proposed method:")
    sens = sensitivity(S)
    sens.round(4).to_csv(TABLES / "T16_recovery_sensitivity.csv", index=False)
    print(f"\nTables -> {TABLES}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
