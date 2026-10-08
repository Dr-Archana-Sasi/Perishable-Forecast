"""Step 3: recover latent (true) daily demand on days with stockouts.

Method
    On a stockout day, sales are recorded only while the item is in stock. Each
    series has an intraday demand profile p_h (share of daily demand in hour h),
    estimated from days with no stockout during opening hours (06:00-22:00).
    If S is the set of stockout hours and X the sales in the remaining hours,
    daily demand is estimated by projection:

        D = max(observed sales, X / (1 - sum_{h in S} p_h))

    Series profiles are shrunk towards the third-level-category profile,
        p = (n * p_series + k * p_category) / (n + k),
    where n is the number of stockout-free days of the series. When less than
    TAU of the daily profile is in stock, projection is unstable and a rolling
    median of stockout-free days within +/- 14 days is used instead (for training
    days, computed from training days only).

Validation (synthetic censoring)
    A random 20% of stockout-free training days are held out. Real stockout
    patterns, copied from other days, are applied to them, and each method's
    estimate is compared with the known true total. Profiles and fallbacks are
    estimated without the held-out days.

Usage
    python src/recover_demand.py                       # full dataset
    python src/recover_demand.py --sample-series 5000  # quick check

Outputs
    data/processed/panel_recovered.parquet
    results/tables/T5_recovery_validation.csv
    results/tables/T6_recovery_validation_by_severity.csv
    results/tables/T7_recovery_summary.csv
    results/figures/F6_recovery_validation.png
    results/figures/F7_recovery_example.png
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config import (DATA_PROCESSED, FIGURES, TABLES, OPEN_HOURS, SEED, INK_2,
                    C_INSTOCK, C_STOCKOUT, set_plot_style, set_seed)
from data_utils import KEYS, load_panel

TAU = 0.30        # minimum in-stock share of the daily profile needed to project
K_SHRINK = 10.0   # pseudo-days used to shrink series profiles towards the category
HALF_WINDOW = 14  # rolling-median fallback uses +/- 14 days
VAL_FRAC = 0.20   # share of stockout-free training days held out for validation

OPEN = np.zeros(24, dtype=bool)
OPEN[OPEN_HOURS] = True

METHOD_LABELS = {
    "observed": "Observed sales (no recovery)",
    "global": "Global profile",
    "category": "Category profile",
    "series": "Series profile",
    "shrunk": "Shrunk series profile (proposed)",
}
METHOD_COLORS = {"observed": "#8a8984", "global": "#eda100", "category": "#1baf7a",
                 "series": "#eb6834", "shrunk": "#2a78d6"}
# Validation days are grouped by the number of stockout hours applied to them.
# (Grouping by the realised hidden share would condition on the outcome and bias
# the per-group comparison.)
SEVERITY_BINS = [1, 5, 9, 13, 17]
SEVERITY_LABELS = ["1-4", "5-8", "9-12", "13-16"]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _group_sum(idx: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Sum the 24 columns of w within groups given by idx (fast bincount)."""
    return np.stack([np.bincount(idx, weights=w[:, h], minlength=n) for h in range(24)],
                    axis=1)


def _normalise(S: np.ndarray) -> np.ndarray:
    tot = S.sum(axis=1, keepdims=True)
    return np.divide(S, tot, out=np.full_like(S, np.nan, dtype=float), where=tot > 0)


def build_profiles(hs, sidx, s2c, use, n_s, n_c, k=K_SHRINK) -> dict:
    """Hourly demand profiles (n_series x 24) for each estimation method."""
    S = _group_sum(sidx[use], hs[use].astype(float), n_s)
    n_days = np.bincount(sidx[use], minlength=n_s).astype(float)

    G = S.sum(axis=0)
    p_glob = G / G.sum()
    p_cat = _normalise(_group_sum(s2c, S, n_c))
    p_cat[np.isnan(p_cat).any(axis=1)] = p_glob          # empty category -> global
    p_cat_s = p_cat[s2c]

    p_ser = _normalise(S)
    no_data = np.isnan(p_ser).any(axis=1)
    p_ser_filled = np.where(no_data[:, None], p_cat_s, p_ser)  # no clean days -> category

    w = (n_days / (n_days + k))[:, None]
    p_shrunk = w * np.nan_to_num(p_ser) + (1 - w) * p_cat_s

    return {"global": np.tile(p_glob, (n_s, 1)), "category": p_cat_s,
            "series": p_ser_filled, "shrunk": p_shrunk}


def rolling_median(panel: np.ndarray, half: int = HALF_WINDOW, chunk: int = 2000) -> np.ndarray:
    """Centred rolling median along days, ignoring NaN (n_series x T)."""
    n, T = panel.shape
    pad = np.full((n, T + 2 * half), np.nan, dtype=np.float32)
    pad[:, half:half + T] = panel
    out = np.empty((n, T), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN
        for i in range(0, n, chunk):
            win = sliding_window_view(pad[i:i + chunk], 2 * half + 1, axis=1)
            out[i:i + chunk] = np.nanmedian(win, axis=2)
    return out


def recover(X, observed, share_in, fallback, tau=TAU):
    """Projection where enough of the profile is in stock, otherwise fallback."""
    proj = np.divide(X, share_in, out=np.zeros_like(X, dtype=float), where=share_in > 0)
    fb = np.where(np.isnan(fallback), observed, fallback)
    can_project = share_in >= tau
    est = np.where(can_project, proj, fb)
    return np.maximum(observed, est), can_project


def wape(est, true):
    return float(np.abs(est - true).sum() / true.sum())


def bias(est, true):
    return float((est - true).sum() / true.sum())


# --------------------------------------------------------------------------- #
# Validation with synthetic censoring
# --------------------------------------------------------------------------- #
def validate(df, hs, hst, sidx, s2c, n_s, n_c, T, rng):
    train = (df["split"] == "train").to_numpy()
    H = hs.sum(axis=1).astype(float)
    cens = (hst[:, OPEN] == 1).any(axis=1)

    clean = train & ~cens
    val = clean & (H > 0) & (rng.random(len(df)) < VAL_FRAC)
    est_days = clean & ~val
    profiles = build_profiles(hs, sidx, s2c, est_days & (H > 0), n_s, n_c)
    fb = rolling_median(np.where(est_days, H, np.nan).reshape(n_s, T)).ravel()

    vi = np.flatnonzero(val)
    donors = np.flatnonzero(train & cens)
    M = np.zeros((len(vi), 24), dtype=bool)
    M[:, OPEN] = hst[rng.choice(donors, size=len(vi))][:, OPEN] == 1

    true = H[vi]
    X = (hs[vi] * ~M).sum(axis=1).astype(float)

    est = {"observed": X}
    for name, P in profiles.items():
        share_in = 1 - (P[sidx[vi]] * M).sum(axis=1)
        est[name], can_proj = recover(X, X, share_in, fb[vi])
    proj_share = float(can_proj.mean())

    overall = pd.DataFrame([{"method": METHOD_LABELS[m], "validation_days": len(vi),
                             "WAPE": wape(e, true), "bias": bias(e, true)}
                            for m, e in est.items()])
    sev = pd.cut(M.sum(axis=1), SEVERITY_BINS, labels=SEVERITY_LABELS, right=False)
    rows = []
    for m, e in est.items():
        for lab in SEVERITY_LABELS:
            s = np.asarray(sev == lab)
            if s.any():
                rows.append({"method": METHOD_LABELS[m], "method_key": m,
                             "stockout_hours": lab, "days": int(s.sum()),
                             "WAPE": wape(e[s], true[s]), "bias": bias(e[s], true[s])})
    return overall, pd.DataFrame(rows), proj_share


# --------------------------------------------------------------------------- #
# Recovery on the real data
# --------------------------------------------------------------------------- #
def recover_panel(df, hs, hst, sidx, s2c, n_s, n_c, T):
    train = (df["split"] == "train").to_numpy()
    H = hs.sum(axis=1).astype(float)
    sale = df["sale_amount"].to_numpy(dtype=float)
    M = (hst == 1) & OPEN
    cens = M.any(axis=1)

    rel_gap = np.abs(H - sale) / np.maximum(sale, 1e-9)
    print(f"Check: median relative gap between hourly sum and daily sales = "
          f"{np.median(rel_gap[sale > 0]):.2e}")

    # Profiles from all stockout-free training days (no evaluation-week data)
    P = build_profiles(hs, sidx, s2c, train & ~cens & (H > 0), n_s, n_c)["shrunk"]
    share_in = 1 - (P[sidx] * M).sum(axis=1)
    X = (hs * ~M).sum(axis=1).astype(float)
    ratio = np.where(H > 0, sale / np.maximum(H, 1e-12), 1.0)   # hourly -> daily scale
    # Fallback: rolling median of stockout-free days. Training days use training
    # data only, so no evaluation-week information enters the training targets.
    fb_train = rolling_median(np.where(~cens & train, sale, np.nan).reshape(n_s, T)).ravel()
    fb_all = rolling_median(np.where(~cens, sale, np.nan).reshape(n_s, T)).ravel()
    fb = np.where(train, fb_train, fb_all)

    est, can_proj = recover(X * ratio, sale, share_in, fb)
    out = df.copy()
    out["stockout_open_hours"] = M.sum(axis=1).astype(np.int8)
    out["censored"] = cens
    out["instock_profile_share"] = np.where(cens, share_in, 1.0).astype(np.float32)
    out["demand"] = np.where(cens, est, sale)
    out["recovery_method"] = np.where(~cens, "none", np.where(can_proj, "projection", "fallback"))
    return out


def summarise(out) -> pd.DataFrame:
    rows = []
    for split in ["train", "eval", "all"]:
        d = out if split == "all" else out[out["split"] == split]
        c = d[d["censored"]]
        rows.append({
            "split": split,
            "store_product_days": len(d),
            "censored_days_pct": 100 * d["censored"].mean(),
            "recovered_by_projection_pct": 100 * (d["recovery_method"] == "projection").mean(),
            "recovered_by_fallback_pct": 100 * (d["recovery_method"] == "fallback").mean(),
            "demand_uplift_all_days_pct": 100 * (d["demand"].sum() / d["sale_amount"].sum() - 1),
            "demand_uplift_censored_days_pct": 100 * (c["demand"].sum() / c["sale_amount"].sum() - 1),
        })
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_validation(sev_tab):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(SEVERITY_LABELS))
    for key in METHOD_LABELS:
        d = sev_tab[sev_tab["method_key"] == key].set_index("stockout_hours").reindex(SEVERITY_LABELS)
        style = dict(color=METHOD_COLORS[key], marker="o", ms=5, label=METHOD_LABELS[key])
        if key == "observed":
            style.update(ls="--", marker="s")
        axes[0].plot(x, d["WAPE"], **style)
        axes[1].plot(x, d["bias"], **style)
    axes[0].set(ylabel="WAPE", title="(a) Error of recovered daily demand")
    axes[1].set(ylabel="Bias (share of true demand)", title="(b) Bias of recovered daily demand")
    axes[1].axhline(0, color=INK_2, lw=1)
    for ax in axes:
        ax.set_xticks(x, SEVERITY_LABELS)
        ax.set_xlabel("Stockout hours applied (06:00-22:00)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURES / "F6_recovery_validation.png")
    plt.close(fig)


def fig_example(out):
    tr = out[out["split"] == "train"]
    s = tr.groupby(KEYS).agg(mean_sales=("sale_amount", "mean"),
                             cens_days=("censored", "mean"))
    cand = s[s["cens_days"].between(0.2, 0.5) & (s["mean_sales"] > 0)]
    key = (cand if len(cand) else s).sort_values("mean_sales", ascending=False).index[0]
    d = out[(out["store_id"] == key[0]) & (out["product_id"] == key[1])].sort_values("dt")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(d["dt"], d["demand"], color=C_INSTOCK, lw=1.8, label="Recovered demand")
    ax.plot(d["dt"], d["sale_amount"], color=INK_2, lw=1.2, ls="--", label="Recorded sales")
    c = d["censored"].to_numpy()
    ax.scatter(d.loc[c, "dt"], d.loc[c, "sale_amount"], color=C_STOCKOUT, s=26, zorder=3,
               edgecolor="white", linewidth=1, label="Day with stockout")
    ev_start = d.loc[d["split"] == "eval", "dt"].min()
    if pd.notna(ev_start):
        ax.axvline(ev_start, color=INK_2, lw=1, ls=":")
        ax.text(ev_start, ax.get_ylim()[1], " evaluation week", va="top", fontsize=8, color=INK_2)
    ax.set(ylabel="Daily quantity (normalised)",
           title=f"Recorded sales and recovered demand (store {key[0]}, product {key[1]})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "F7_recovery_example.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-series", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    set_plot_style()
    rng = np.random.default_rng(SEED)

    df, hs, hst, T = load_panel(args.sample_series)
    n_s = len(df) // T
    sidx = np.repeat(np.arange(n_s), T)
    s2c_codes, _ = pd.factorize(df["third_category_id"].to_numpy()[::T])
    s2c = s2c_codes.astype(np.int64)
    n_c = int(s2c.max()) + 1
    print(f"Loaded {len(df):,} rows: {n_s:,} series x {T} days")

    overall, sev_tab, proj_share = validate(df, hs, hst, sidx, s2c, n_s, n_c, T, rng)
    overall.round(4).to_csv(TABLES / "T5_recovery_validation.csv", index=False)
    sev_tab.drop(columns="method_key").round(4).to_csv(
        TABLES / "T6_recovery_validation_by_severity.csv", index=False)
    print("\nValidation on held-out stockout-free days (synthetic censoring):")
    print(overall.round(4).to_string(index=False))
    print(f"Share of validation days recovered by projection: {proj_share:.3f}")

    out = recover_panel(df, hs, hst, sidx, s2c, n_s, n_c, T)
    out.to_parquet(DATA_PROCESSED / "panel_recovered.parquet", index=False)
    summary = summarise(out)
    summary.to_csv(TABLES / "T7_recovery_summary.csv", index=False)
    print("\nRecovery on the real data:")
    print(summary.to_string(index=False))

    fig_validation(sev_tab)
    fig_example(out)
    print(f"\nPanel   -> {DATA_PROCESSED / 'panel_recovered.parquet'}")
    print(f"Tables  -> {TABLES}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
