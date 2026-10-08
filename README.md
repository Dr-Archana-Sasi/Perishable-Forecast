# Stockout-Aware Probabilistic Forecasting and Replenishment for Perishable Goods

Code accompanying the paper *"A Stockout-Aware Probabilistic Forecasting and
Replenishment Framework for Perishable Goods Supply Chains"* (Sasi, 2026).

When a perishable item sells out, recorded sales understate true demand.
Forecasts trained on recorded sales therefore under-forecast, which leads to more
stockouts. The framework has four stages:

1. **Demand recovery** – estimate true daily demand on stockout days from the
   intraday sales profile, validated by synthetic censoring of stockout-free days.
2. **Probabilistic forecasting** – quantile forecasts (0.1–0.9) of daily demand
   with seasonal naive, empirical-quantile, ETS and LightGBM models, each trained
   on recorded sales and on recovered demand.
3. **Shelf-life-aware ordering** – daily order-up-to policy at the newsvendor
   critical ratio, with first-in-first-out sale and spoilage after 1–3 days.
4. **Simulation** – fill rate, waste and cost for every forecast set, unit cost
   and shelf life, with paired Wilcoxon tests across series.

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
(Wang et al., 2025; CC BY 4.0): 50,000 store–product series of hourly sales of
perishable items with hourly stockout annotations, 90 training days
(2024-03-28 to 2024-06-25) and a 7-day evaluation week (2024-06-26 to 2024-07-02).
The data is not stored in this repository; `src/download_data.py` downloads it.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or open `notebooks/run_pipeline.ipynb` in Google Colab and run the cells in order.

## Reproducing the results

Run from the repository root, in this order. Run times are for a free Google
Colab CPU runtime (2 cores, 12 GB RAM).

| Step | Command | Run time | Outputs |
|---|---|---|---|
| 1. Download data | `python src/download_data.py` | ~2 min | `data/raw/train.parquet`, `data/raw/eval.parquet` |
| 2. Exploratory analysis | `python src/eda.py` | ~5 min | Tables T1–T4, Figures F1–F5 |
| 3. Demand recovery | `python src/recover_demand.py` | ~3 min | `data/processed/panel_recovered.parquet`, Tables T5–T7, Figures F6–F7 |
| 4. Forecasting | `python src/forecast.py` | ~65 min | `data/processed/forecasts_eval.parquet`, Tables T8–T10b, Figure F8 |
| 5. Inventory simulation | `python src/inventory_sim.py` | ~1 min | Tables T11–T14, Figures F9–F10 |
| 6. Record software versions | `python src/record_environment.py` | seconds | `results/environment.txt` |

`src/eda.py`, `src/recover_demand.py` and `src/forecast.py` accept
`--sample-series N` for a quick check on a random subset of N series. The
results in the paper use the full data (no `--sample-series`).

All randomness is seeded (seed 42, `src/config.py`) and LightGBM runs in
deterministic mode. The versions used for the paper are in `results/environment.txt`.

## Outputs

| File | Content |
|---|---|
| T1 | Dataset summary |
| T2 | Stockouts by product category |
| T3 | Recorded sales by stockout duration |
| T4 | Sales by discount, holiday, promotion and precipitation (stockout-free days) |
| T5, T6 | Validation of demand recovery (overall; by stockout hours) |
| T7 | Share of censored days and demand uplift after recovery |
| T8 | Forecast accuracy (WAPE, bias, scaled pinball loss, 80% interval coverage) |
| T9 | Paired Wilcoxon tests of forecast loss across series |
| T10, T10b | LightGBM feature importance and training details |
| T11 | Inventory results, base case (unit cost 0.5, shelf life 1 day) |
| T12 | Inventory results for all unit costs, shelf lives and demand assumptions |
| T13 | Paired Wilcoxon tests of inventory cost across series |
| T14 | Lowest cost per forecast set with its best ordering level |
| F1–F10 | Figures as listed in the scripts' headers |

## Repository structure

```
src/
  config.py             paths, seed, plotting style
  data_utils.py         loading the hourly data as arrays
  download_data.py      step 1
  eda.py                step 2
  recover_demand.py     step 3
  forecast.py           step 4
  inventory_sim.py      step 5
  record_environment.py step 6
notebooks/
  run_pipeline.ipynb    runs all steps in Google Colab
data/                   created when the data is downloaded (not tracked)
results/                tables, figures and environment.txt
```

## Citation

See `CITATION.cff`. Please also cite the dataset:

> Wang, Y., Gu, J., Long, L., Li, X., Shen, L., Fu, Z., Zhou, X., & Jiang, X. (2025).
> FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand
> Recovery and Forecasting in Fresh Retail. arXiv:2505.16319.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, by its authors.
