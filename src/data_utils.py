"""Shared data-loading helpers for FreshRetailNet-50K.

The raw Parquet files store the 24 hourly sales and stock-status values of each
store-product-day as list columns. These helpers return

    df  : one row per store-product-day (scalar columns only)
    hs  : (n_rows, 24) float32 array of hourly sales
    hst : (n_rows, 24) int8 array of hourly stock status (1 = out of stock)

with rows sorted by store, product and date so that every series occupies a
contiguous block of rows.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import DATA_RAW, SEED

KEYS = ["store_id", "product_id"]


def _list_col_to_matrix(table, name: str, dtype) -> np.ndarray:
    """Convert a list column of fixed length 24 into an (n, 24) array."""
    col = table.column(name).combine_chunks()
    lengths = col.value_lengths().to_numpy(zero_copy_only=False)
    if not (lengths == 24).all():
        raise ValueError(f"{name}: expected 24 hourly values per row, "
                         f"found lengths {np.unique(lengths)}")
    return col.flatten().to_numpy(zero_copy_only=False).astype(dtype).reshape(-1, 24)


def load_split(split: str, keys: pd.DataFrame | None = None):
    """Load one split ('train' or 'eval'), optionally keeping only the given series."""
    table = pq.read_table(DATA_RAW / f"{split}.parquet")
    scalar_cols = [c for c in table.column_names
                   if c not in ("hours_sale", "hours_stock_status")]
    df = table.select(scalar_cols).to_pandas()
    hs = _list_col_to_matrix(table, "hours_sale", np.float32)
    hst = _list_col_to_matrix(table, "hours_stock_status", np.int8)
    del table

    if keys is not None:
        mask = pd.MultiIndex.from_frame(df[KEYS]).isin(pd.MultiIndex.from_frame(keys[KEYS]))
        df, hs, hst = df[mask], hs[mask], hst[mask]

    df = df.reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["dt"])
    return df, hs, hst


def sample_keys(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Random, reproducible subset of n store-product series."""
    series = df[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(series), size=min(n, len(series)), replace=False))
    return series.iloc[idx]


def load_panel(sample_series: int | None = None):
    """Load train and eval splits as one sorted panel.

    Returns df (with a 'split' column), hs, hst and T, the number of days per
    series. Every series is checked to have exactly T days.
    """
    df_tr, hs_tr, hst_tr = load_split("train")
    keys = None
    if sample_series:
        keys = sample_keys(df_tr, sample_series)
        mask = pd.MultiIndex.from_frame(df_tr[KEYS]).isin(pd.MultiIndex.from_frame(keys))
        df_tr, hs_tr, hst_tr = df_tr[mask].reset_index(drop=True), hs_tr[mask], hst_tr[mask]
    df_ev, hs_ev, hst_ev = load_split("eval", keys)

    df_tr["split"], df_ev["split"] = "train", "eval"
    df = pd.concat([df_tr, df_ev], ignore_index=True)
    hs = np.vstack([hs_tr, hs_ev])
    hst = np.vstack([hst_tr, hst_ev])
    del df_tr, df_ev, hs_tr, hs_ev, hst_tr, hst_ev

    order = np.lexsort((df["dt"].values, df["product_id"].values, df["store_id"].values))
    df, hs, hst = df.iloc[order].reset_index(drop=True), hs[order], hst[order]

    sizes = df.groupby(KEYS, sort=False).size()
    T = int(sizes.iloc[0])
    if not (sizes == T).all():
        raise ValueError("Series have different numbers of days; panel is not balanced.")
    return df, hs, hst, T
