"""Record the Python version and package versions used for a run.

Usage:
    python src/record_environment.py

Writes results/environment.txt so that the exact software versions behind the
reported results are archived with them.
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from config import ROOT

PACKAGES = ["numpy", "pandas", "pyarrow", "matplotlib", "datasets", "scikit-learn",
            "lightgbm", "statsforecast", "scipy", "statsmodels"]


def main() -> None:
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for p in PACKAGES:
        try:
            lines.append(f"{p}=={version(p)}")
        except PackageNotFoundError:
            lines.append(f"{p} (not installed)")
    out = ROOT / "results" / "environment.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
