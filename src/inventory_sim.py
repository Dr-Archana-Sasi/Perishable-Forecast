"""Step 5: inventory simulation for perishable items driven by the forecasts.

Each series is replenished every morning during the evaluation week with zero
lead time. The order brings stock up to the forecast quantile at the critical
ratio tau of the day's demand:

    order_t = max(0, Q_tau(day t) - usable stock on hand)

Stock is sold first-in-first-out and spoils after L days (L = 1 means unsold
stock is wasted at the end of the day). Unmet demand is lost. Stock still on
hand at the end of the evaluation week is counted as waste.

Costs (selling price normalised to 1, unit purchase cost c)
    underage (lost margin per unit of lost sales)    cu = 1 - c
    overage  (purchase cost of a wasted unit)         co = c
    critical ratio                                    tau = cu / (cu + co) = 1 - c
    Q_tau is linearly interpolated between the forecast quantiles 0.1-0.9.

Demand used to run the simulation ("truth")
    recovered  recovered demand from Step 3 (main analysis)
    recorded   recorded sales, a lower bound on true demand (robustness check)

Fair comparison with a tuned ordering level
    With L > 1, leftover stock can still be sold the next day, so the cost-
    minimising ordering level is higher than the single-period critical ratio.
    Forecasts that are simply higher could then look better. To separate
    forecast quality from the ordering level, every forecast set is also run
    with ordering quantiles 0.1, 0.2, ..., 0.9 and its lowest cost is reported.
    The level is chosen on the evaluation week itself (ex post), identically for
    all models, so absolute costs are optimistic but comparisons are fair.

Usage
    python src/inventory_sim.py

Outputs
    results/tables/T11_inventory_base_case.csv
    results/tables/T12_inventory_sensitivity.csv
    results/tables/T13_inventory_significance.csv
    results/tables/T14_inventory_tuned_level.csv
    results/figures/F9_inventory_tradeoff.png
    results/figures/F10_cost_sensitivity.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_PROCESSED, FIGURES, TABLES, INK_2, set_plot_style, set_seed

QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
QCOLS = ["q10", "q30", "q50", "q70", "q90"]
UNIT_COSTS = [0.3, 0.4, 0.5, 0.6, 0.7]     # tau = 0.7 ... 0.3
SHELF_LIVES = [1, 2, 3]
BASE = {"unit_cost": 0.5, "shelf_life": 1}
TRUTHS = {"recovered": "truth", "recorded": "recorded_sales"}
TRUTH_LABELS = {"recovered": "Recovered demand", "recorded": "Recorded sales (lower bound)"}
VERSION_LABELS = {"sales": "Recorded sales", "recovered": "Recovered demand"}
MODEL_ORDER = ["Seasonal naive", "Empirical-28", "ETS", "LightGBM"]
MODEL_COLORS = {"Seasonal naive": "#8a8984", "Empirical-28": "#eda100",
                "ETS": "#1baf7a", "LightGBM": "#2a78d6"}
SHELF_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
ORDER_LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)


def load():
    fc = pd.read_parquet(DATA_PROCESSED / "forecasts_eval.parquet")
    fc = fc.sort_values(["model", "version", "series", "t"]).reset_index(drop=True)
    n_s = fc["series"].nunique()
    H = fc["t"].nunique()
    arrays = {}
    for (model, version), g in fc.groupby(["model", "version"], sort=False):
        arrays[(model, version)] = g[QCOLS].to_numpy().reshape(n_s, H, len(QCOLS))
    first = fc[(fc["model"] == fc["model"].iloc[0]) & (fc["version"] == fc["version"].iloc[0])]
    demand = {k: first[c].to_numpy().reshape(n_s, H) for k, c in TRUTHS.items()}
    return arrays, demand, n_s, H


def quantile_at(Qarr: np.ndarray, tau: float) -> np.ndarray:
    """Linear interpolation of the forecast quantiles at level tau -> (n_s, H)."""
    j = int(np.clip(np.searchsorted(QUANTILES, tau) - 1, 0, len(QUANTILES) - 2))
    w = (tau - QUANTILES[j]) / (QUANTILES[j + 1] - QUANTILES[j])
    return (1 - w) * Qarr[:, :, j] + w * Qarr[:, :, j + 1]


def simulate(D: np.ndarray, S: np.ndarray, L: int) -> dict:
    """FIFO perishable inventory. D, S: (n_s, H). Returns per-series totals."""
    n, H = D.shape
    B = np.zeros((n, L))            # B[:, j]: units with j + 1 days of life left
    tot = {k: np.zeros(n) for k in ("demand", "order", "sales", "lost", "waste", "stockout_days")}
    for t in range(H):
        q = np.maximum(S[:, t] - B.sum(axis=1), 0.0)
        B[:, L - 1] += q
        rem = D[:, t].copy()
        sold = np.zeros(n)
        for j in range(L):                          # oldest stock first
            s = np.minimum(B[:, j], rem)
            B[:, j] -= s
            rem -= s
            sold += s
        tot["demand"] += D[:, t]
        tot["order"] += q
        tot["sales"] += sold
        tot["lost"] += rem
        tot["stockout_days"] += rem > 1e-9
        tot["waste"] += B[:, 0]                     # expires tonight
        B[:, :-1] = B[:, 1:]
        B[:, -1] = 0.0
    # Stock left at the end of the horizon is counted as waste (conservative:
    # otherwise over-ordering in the last days would go unpenalised when L > 1).
    tot["waste"] += B.sum(axis=1)
    return tot


def summarise(tot, cu, co, H):
    d = tot["demand"].sum()
    cost = cu * tot["lost"] + co * tot["waste"]
    return {
        "fill_rate": tot["sales"].sum() / d,
        "waste_pct_of_orders": tot["waste"].sum() / max(tot["order"].sum(), 1e-12),
        "lost_sales_pct_of_demand": tot["lost"].sum() / d,
        "stockout_day_rate": tot["stockout_days"].sum() / (len(tot["demand"]) * H),
        "cost_per_unit_demand": cost.sum() / d,
    }, cost


def main() -> None:
    set_seed()
    set_plot_style()
    arrays, demand, n_s, H = load()
    print(f"{len(arrays)} forecast sets, {n_s:,} series x {H} days")

    rows, per_series_cost = [], {}
    for truth, D in demand.items():
        for c in UNIT_COSTS:
            cu, co = 1 - c, c
            tau = cu / (cu + co)
            for L in SHELF_LIVES:
                for (model, version), Qarr in arrays.items():
                    tot = simulate(D, quantile_at(Qarr, tau), L)
                    m, cost = summarise(tot, cu, co, H)
                    rows.append({"truth": TRUTH_LABELS[truth], "unit_cost": c,
                                 "critical_ratio": round(tau, 2), "shelf_life_days": L,
                                 "model": model, "trained_on": VERSION_LABELS[version], **m})
                    per_series_cost[(truth, c, L, model, version)] = cost
    res = pd.DataFrame(rows)

    res["_m"] = res["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    res = res.sort_values(["truth", "unit_cost", "shelf_life_days", "_m", "trained_on"],
                          ascending=[False, True, True, True, False]).drop(columns="_m")
    base = res[(res["unit_cost"] == BASE["unit_cost"]) &
               (res["shelf_life_days"] == BASE["shelf_life"])].copy()
    res.round(4).to_csv(TABLES / "T12_inventory_sensitivity.csv", index=False)
    base.round(4).to_csv(TABLES / "T11_inventory_base_case.csv", index=False)

    # Paired Wilcoxon tests on per-series cost, base case
    pairs = [(("LightGBM", "recovered"), ("LightGBM", "sales")),
             (("ETS", "recovered"), ("ETS", "sales")),
             (("Empirical-28", "recovered"), ("Empirical-28", "sales")),
             (("LightGBM", "recovered"), ("ETS", "recovered")),
             (("LightGBM", "recovered"), ("Empirical-28", "recovered"))]
    sig = []
    for truth in demand:
        for a, b in pairs:
            ca = per_series_cost[(truth, BASE["unit_cost"], BASE["shelf_life"], *a)]
            cb = per_series_cost[(truth, BASE["unit_cost"], BASE["shelf_life"], *b)]
            diff = ca - cb
            sig.append({"truth": TRUTH_LABELS[truth],
                        "model_A": f"{a[0]} ({VERSION_LABELS[a[1]]})",
                        "model_B": f"{b[0]} ({VERSION_LABELS[b[1]]})",
                        "total_cost_change_A_vs_B_pct": 100 * (ca.sum() / cb.sum() - 1),
                        "share_series_A_cheaper": float((diff < 0).mean()),
                        "share_series_equal": float((diff == 0).mean()),
                        "wilcoxon_p": wilcoxon(diff, zero_method="wilcox").pvalue})
    sig = pd.DataFrame(sig)
    sig.to_csv(TABLES / "T13_inventory_significance.csv", index=False)

    tuned = tuned_level(arrays, demand, H)
    tuned.round(4).to_csv(TABLES / "T14_inventory_tuned_level.csv", index=False)

    fig_tradeoff(base)
    fig_sensitivity(res)

    pd.set_option("display.width", 220)
    cols = ["truth", "model", "trained_on", "fill_rate", "waste_pct_of_orders",
            "stockout_day_rate", "cost_per_unit_demand"]
    print(f"\nBase case (unit cost {BASE['unit_cost']}, critical ratio "
          f"{1 - BASE['unit_cost']:.1f}, shelf life {BASE['shelf_life']} day):")
    print(base[cols].round(4).to_string(index=False))
    print("\nPaired Wilcoxon tests on per-series cost (base case):")
    print(sig.round(4).to_string(index=False))
    print("\nLowest cost with each forecast set's best ordering level "
          f"(unit cost {BASE['unit_cost']}):")
    t = tuned[tuned["unit_cost"] == BASE["unit_cost"]]
    print(t[["truth", "shelf_life_days", "model", "trained_on", "best_order_level",
             "min_cost_per_unit_demand"]].round(4).to_string(index=False))
    print(f"\nTables -> {TABLES}\nFigures -> {FIGURES}")


def tuned_level(arrays, demand, H):
    """Lowest cost over ordering quantile levels, per forecast set and scenario.

    Lost sales and waste do not depend on the unit cost, so each level is
    simulated once and costs are computed afterwards for every unit cost.
    """
    rows = []
    for truth, D in demand.items():
        d_sum = D.sum()
        for L in SHELF_LIVES:
            for (model, version), Qarr in arrays.items():
                lost, waste = [], []
                for lev in ORDER_LEVELS:
                    tot = simulate(D, quantile_at(Qarr, lev), L)
                    lost.append(tot["lost"].sum())
                    waste.append(tot["waste"].sum())
                lost, waste = np.array(lost), np.array(waste)
                for c in UNIT_COSTS:
                    cost = ((1 - c) * lost + c * waste) / d_sum
                    k = int(np.argmin(cost))
                    rows.append({"truth": TRUTH_LABELS[truth], "unit_cost": c,
                                 "shelf_life_days": L, "model": model,
                                 "trained_on": VERSION_LABELS[version],
                                 "best_order_level": ORDER_LEVELS[k],
                                 "min_cost_per_unit_demand": cost[k]})
    out = pd.DataFrame(rows)
    out["_m"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["truth", "unit_cost", "shelf_life_days", "_m", "trained_on"],
                           ascending=[False, True, True, True, False]).drop(columns="_m")


def fig_tradeoff(base):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, truth in zip(axes, TRUTH_LABELS.values()):
        d = base[base["truth"] == truth]
        for model in MODEL_ORDER:
            pts = {}
            for version, mk in [("Recorded sales", "s"), ("Recovered demand", "o")]:
                r = d[(d["model"] == model) & (d["trained_on"] == version)]
                if r.empty:
                    continue
                x, y = 100 * r["waste_pct_of_orders"].iloc[0], 100 * r["fill_rate"].iloc[0]
                pts[version] = (x, y)
                filled = mk == "o"
                ax.scatter(x, y, s=60, marker=mk, zorder=3, linewidth=1.5,
                           facecolor=MODEL_COLORS[model] if filled else "white",
                           edgecolor="white" if filled else MODEL_COLORS[model])
            if len(pts) == 2:
                (x0, y0), (x1, y1) = pts["Recorded sales"], pts["Recovered demand"]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[model], lw=1.2))
        ax.set(xlabel="Waste (% of units ordered)", ylabel="Fill rate (% of demand met)",
               title=f"Simulated with: {truth.lower()}")
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], marker="o", ls="", label=m)
               for m in MODEL_ORDER]
    handles += [plt.Line2D([], [], color=INK_2, marker="s", markerfacecolor="white", ls="",
                           label="trained on recorded sales"),
                plt.Line2D([], [], color=INK_2, marker="o", ls="",
                           label="trained on recovered demand")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06),
               fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "F9_inventory_tradeoff.png")
    plt.close(fig)


def fig_sensitivity(res):
    d = res[res["model"] == "LightGBM"].pivot_table(
        index=["truth", "shelf_life_days", "critical_ratio"], columns="trained_on",
        values="cost_per_unit_demand")
    d["change_pct"] = 100 * (d["Recovered demand"] / d["Recorded sales"] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, truth in zip(axes, TRUTH_LABELS.values()):
        for L in SHELF_LIVES:
            s = d.loc[(truth, L), "change_pct"]
            ax.plot(s.index, s.values, marker="o", ms=5, color=SHELF_COLORS[L],
                    label=f"shelf life {L} day{'s' if L > 1 else ''}")
        ax.axhline(0, color=INK_2, lw=1)
        ax.set(xlabel="Critical ratio (1 - unit cost)", title=f"Simulated with: {truth.lower()}")
    axes[0].set_ylabel("Cost change, LightGBM on recovered\nvs on recorded sales (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "F10_cost_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
