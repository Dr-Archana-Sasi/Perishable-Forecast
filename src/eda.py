"""Step 2: exploratory data analysis of FreshRetailNet-50K.

Usage:
    python src/eda.py                      # full training split (used in the paper)
    python src/eda.py --sample-series 5000 # random subset of series, much faster

Outputs
    results/tables/T1_dataset_summary.csv
    results/tables/T2_stockout_by_category.csv
    results/tables/T3_sales_vs_stockout_hours.csv
    results/tables/T4_covariate_effects.csv
    results/figures/F1_distributions.png
    results/figures/F2_hourly_profile.png
    results/figures/F3_sales_vs_stockout_hours.png
    results/figures/F4_covariate_effects.png
    results/figures/F5_example_series.png

Note: T3/F3 are descriptive, not causal. Short stockouts often occur on
high-demand days (the item sells out), while long stockouts suppress recorded
sales. Either way, recorded sales on stockout days understate true demand.
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (FIGURES, TABLES, OPEN_HOURS, C_INSTOCK, C_STOCKOUT, INK_2,
                    set_plot_style, set_seed)
from data_utils import KEYS, load_split, sample_keys


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(split: str = "train", sample_series: int | None = None):
    """Training split, optionally restricted to a reproducible random subset of series."""
    if not sample_series:
        return load_split(split)
    df, _, _ = load_split(split)
    keys = sample_keys(df, sample_series)
    del df
    return load_split(split, keys)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def dataset_summary(df, hst) -> pd.DataFrame:
    open_hst = hst[:, OPEN_HOURS]
    rows = {
        "Rows (store-product-days)": len(df),
        "Store-product series": df.groupby(KEYS).ngroups,
        "Stores": df["store_id"].nunique(),
        "Products (SKUs)": df["product_id"].nunique(),
        "Cities": df["city_id"].nunique(),
        "First-level categories": df["first_category_id"].nunique(),
        "Second-level categories": df["second_category_id"].nunique(),
        "Third-level categories": df["third_category_id"].nunique(),
        "Start date": df["dt"].min().date(),
        "End date": df["dt"].max().date(),
        "Days": df["dt"].nunique(),
        "Zero-sales days (%)": 100 * (df["sale_amount"] == 0).mean(),
        "Open hours out of stock (%)": 100 * open_hst.mean(),
        "Days with any stockout (%)": 100 * (df["stock_hour6_22_cnt"] > 0).mean(),
        "Days fully out of stock 06-22 (%)": 100 * (open_hst.min(axis=1) == 1).mean(),
        "Days with discount < 1.0 (%)": 100 * (df["discount"] < 1.0).mean(),
        "Holiday days (%)": 100 * df["holiday_flag"].mean(),
        "Activity/promotion days (%)": 100 * df["activity_flag"].mean(),
    }
    out = pd.DataFrame({"Statistic": rows.keys(), "Value": rows.values()})
    out["Value"] = out["Value"].apply(lambda v: round(v, 2) if isinstance(v, float) else v)
    return out


def add_relative_sales(df) -> pd.DataFrame:
    """Sales divided by the series mean, so items of different scale are comparable."""
    m = df.groupby(KEYS)["sale_amount"].transform("mean")
    df["rel_sales"] = np.where(m > 0, df["sale_amount"] / m, np.nan)
    return df


def stockout_by_category(df) -> pd.DataFrame:
    g = df.groupby("first_category_id")
    out = pd.DataFrame({
        "series": g[KEYS].apply(lambda x: x.drop_duplicates().shape[0]),
        "mean_daily_sales": g["sale_amount"].mean(),
        "stockout_hours_per_day": g["stock_hour6_22_cnt"].mean(),
        "days_with_stockout_pct": 100 * g["stock_hour6_22_cnt"].apply(lambda s: (s > 0).mean()),
    }).sort_values("series", ascending=False)
    return out.round(3).reset_index()


def sales_vs_stockout_hours(df) -> pd.DataFrame:
    bins = [-0.5, 0.5, 4.5, 8.5, 12.5, 16.5]
    labels = ["0", "1-4", "5-8", "9-12", "13-16"]
    b = pd.cut(df["stock_hour6_22_cnt"], bins=bins, labels=labels)
    g = df.groupby(b, observed=False)["rel_sales"]
    out = pd.DataFrame({"days": g.size(), "mean_rel_sales": g.mean(),
                        "median_rel_sales": g.median()})
    out.index.name = "stockout_hours"
    return out.round(4).reset_index()


def covariate_effects(df) -> pd.DataFrame:
    """Mean relative sales on in-stock days only, to avoid censoring bias."""
    d = df[df["stock_hour6_22_cnt"] == 0].copy()
    d["discount_bin"] = pd.cut(d["discount"], [0, 0.7, 0.8, 0.9, 0.999, 1.01],
                               labels=["<0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "none"])
    d["precip_bin"] = pd.qcut(d["precpt"].rank(method="first"), 4,
                              labels=["Q1 (driest)", "Q2", "Q3", "Q4 (wettest)"])
    frames = []
    for var in ["discount_bin", "holiday_flag", "activity_flag", "precip_bin"]:
        g = d.groupby(var, observed=False)["rel_sales"]
        f = pd.DataFrame({"days": g.size(), "mean_rel_sales": g.mean()}).reset_index()
        f.insert(0, "variable", var)
        f = f.rename(columns={var: "level"})
        f["level"] = f["level"].astype(str)
        if var in ("holiday_flag", "activity_flag"):
            f["level"] = f["level"].map({"0": "No", "1": "Yes"})
        frames.append(f)
    return pd.concat(frames, ignore_index=True).round(4)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_distributions(df):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(np.log1p(df["sale_amount"]), bins=60, color=C_INSTOCK, edgecolor="white", linewidth=0.4)
    ax[0].set(xlabel="log(1 + daily sales, normalised)", ylabel="Store-product-days",
              title="(a) Daily sales")
    counts = df["stock_hour6_22_cnt"].value_counts().sort_index()
    ax[1].bar(counts.index, counts.values, color=C_STOCKOUT, width=0.8)
    ax[1].set(xlabel="Stockout hours per day (06:00-22:00)", ylabel="Store-product-days",
              title="(b) Stockout duration")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(0, 17, 2))
    fig.tight_layout()
    fig.savefig(FIGURES / "F1_distributions.png")
    plt.close(fig)


def fig_hourly_profile(hs, hst):
    hours = np.arange(24)
    instock = np.where(hst == 0, hs, np.nan)
    stockout = np.where(hst == 1, hs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # hours with no rows of a status
        m_in, m_out = np.nanmean(instock, 0), np.nanmean(stockout, 0)
    share_out = 100 * hst.mean(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(hours, m_in, color=C_INSTOCK, marker="o", ms=4, label="In stock")
    ax[0].plot(hours, m_out, color=C_STOCKOUT, marker="s", ms=4, label="Out of stock")
    ax[0].set(xlabel="Hour of day", ylabel="Mean hourly sales (normalised)",
              title="(a) Hourly sales by stock status")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    open_mask = np.isin(hours, OPEN_HOURS)
    ax[1].bar(hours[open_mask], share_out[open_mask], color=C_STOCKOUT, width=0.8,
              label="Opening hours (06:00-22:00)")
    ax[1].bar(hours[~open_mask], share_out[~open_mask], color="#c9c8c3", width=0.8,
              label="Outside opening hours (not used)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax[1].set(xlabel="Hour of day", ylabel="Store-product-days out of stock (%)",
              title="(b) Stockout share by hour")
    for a in ax:
        a.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    fig.savefig(FIGURES / "F2_hourly_profile.png")
    plt.close(fig)


def fig_sales_vs_stockout(t3):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(t3["stockout_hours"].astype(str), t3["mean_rel_sales"], color=C_INSTOCK, width=0.6)
    ax.axhline(1.0, color=INK_2, lw=1, ls="--")
    for i, v in enumerate(t3["mean_rel_sales"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set(xlabel="Stockout hours in the day", ylabel="Mean sales / series mean",
           title="Recorded sales by stockout duration")
    fig.tight_layout()
    fig.savefig(FIGURES / "F3_sales_vs_stockout_hours.png")
    plt.close(fig)


def fig_covariates(t4):
    titles = {"discount_bin": "(a) Discount rate", "holiday_flag": "(b) Holiday",
              "activity_flag": "(c) Promotion activity", "precip_bin": "(d) Precipitation quartile"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (var, title) in zip(axes, titles.items()):
        d = t4[t4["variable"] == var]
        ax.bar(d["level"], d["mean_rel_sales"], color=C_INSTOCK, width=0.6)
        ax.axhline(1.0, color=INK_2, lw=1, ls="--")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Mean sales / series mean\n(in-stock days only)")
    fig.tight_layout()
    fig.savefig(FIGURES / "F4_covariate_effects.png")
    plt.close(fig)


def fig_example_series(df):
    """One series with frequent but partial stockouts, chosen deterministically."""
    s = df.groupby(KEYS).agg(mean_sales=("sale_amount", "mean"),
                             so_days=("stock_hour6_22_cnt", lambda x: (x > 0).mean()))
    cand = s[(s["so_days"].between(0.2, 0.5)) & (s["mean_sales"] > 0)]
    key = (cand if len(cand) else s).sort_values("mean_sales", ascending=False).index[0]
    d = df[(df["store_id"] == key[0]) & (df["product_id"] == key[1])].sort_values("dt")
    so = d["stock_hour6_22_cnt"] > 0

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(d["dt"], d["sale_amount"], color=C_INSTOCK, lw=1.5, label="Recorded daily sales")
    ax.scatter(d.loc[so, "dt"], d.loc[so, "sale_amount"], color=C_STOCKOUT, s=28, zorder=3,
               edgecolor="white", linewidth=1, label="Day with stockout")
    ax.set(xlabel="Date", ylabel="Daily sales (normalised)",
           title=f"Example series (store {key[0]}, product {key[1]})")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "F5_example_series.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample-series", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    set_plot_style()
    df, hs, hst = load(args.split, args.sample_series)
    print(f"Loaded {len(df):,} rows")

    t1 = dataset_summary(df, hst)
    t1.to_csv(TABLES / "T1_dataset_summary.csv", index=False)
    print("\n" + t1.to_string(index=False))

    df = add_relative_sales(df)
    stockout_by_category(df).to_csv(TABLES / "T2_stockout_by_category.csv", index=False)
    t3 = sales_vs_stockout_hours(df)
    t3.to_csv(TABLES / "T3_sales_vs_stockout_hours.csv", index=False)
    print("\n" + t3.to_string(index=False))
    t4 = covariate_effects(df)
    t4.to_csv(TABLES / "T4_covariate_effects.csv", index=False)

    fig_distributions(df)
    fig_hourly_profile(hs, hst)
    fig_sales_vs_stockout(t3)
    fig_covariates(t4)
    fig_example_series(df)
    print(f"\nTables -> {TABLES}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
