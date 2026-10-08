"""Step 1: download FreshRetailNet-50K from Hugging Face and save it as Parquet.

Usage:
    python src/download_data.py

Creates data/raw/train.parquet and data/raw/eval.parquet and prints a short
integrity report (rows, series, date range).
"""
from datasets import load_dataset
import pandas as pd

from config import DATA_RAW, HF_DATASET

SERIES_KEYS = ["store_id", "product_id"]


def main() -> None:
    print(f"Downloading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET)

    for split in ds.keys():
        out = DATA_RAW / f"{split}.parquet"
        ds[split].to_parquet(str(out))
        df = pd.read_parquet(out, columns=SERIES_KEYS + ["dt"])
        n_series = df.groupby(SERIES_KEYS).ngroups
        print(f"\n[{split}] saved -> {out}")
        print(f"  rows          : {len(df):,}")
        print(f"  series        : {n_series:,}")
        print(f"  date range    : {df['dt'].min()} to {df['dt'].max()}")
        print(f"  days / series : {len(df) / n_series:.1f}")


if __name__ == "__main__":
    main()
