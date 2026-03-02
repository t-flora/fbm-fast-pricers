"""
Block 1: Calibrate Hurst exponent (H) and vol-of-vol (nu) from Oxford-Man data.

Downloads (or reads from data/raw/) the Oxford-Man Realized Library CSV,
computes log-volatility increments, and estimates H via fractional linear
regression (log-log regression of variogram increments).

Usage:
    python data/calibrate.py [--ticker SPX2.rv] [--lag-max 20]

Output:
    Prints H and nu to stdout — copy into src/common/params.hpp.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DEFAULT_FILE = os.path.join(RAW_DIR, "oxfordmanrealizedvolatilityindices.csv")

# Oxford-Man data download URL (manual download required; direct fetch may 403)
DATA_URL = "https://realized.oxford-man.ox.ac.uk/data/download"


def load_oxford_man(filepath: str, rv_col: str = "rv5") -> pd.Series:
    """Load Oxford-Man CSV and return a Series of realized variance for one index."""
    df = pd.read_csv(filepath, header=2, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    if rv_col not in df.columns:
        available = [c for c in df.columns if "rv" in c.lower()]
        raise ValueError(f"Column '{rv_col}' not found. Available RV columns: {available}")
    return df[rv_col].dropna()


def estimate_hurst(log_vol: np.ndarray, lag_max: int = 20) -> float:
    """
    Estimate Hurst exponent H via variogram regression.
    E[|X(t+h) - X(t)|^2] ~ h^{2H}  =>  slope of log-log plot = 2H
    """
    lags = np.arange(1, lag_max + 1)
    variogram = np.array([
        np.mean((log_vol[lag:] - log_vol[:-lag]) ** 2)
        for lag in lags
    ])
    log_lags = np.log(lags)
    log_var = np.log(variogram)
    result = stats.linregress(log_lags, log_var)
    slope, r = result.slope, result.rvalue
    H = slope / 2.0
    print(f"  Variogram regression: slope={slope:.4f}, R²={r**2:.4f}")
    return H


def estimate_nu(log_vol: np.ndarray) -> float:
    """Estimate vol-of-vol (nu) as std of log-vol increments."""
    increments = np.diff(log_vol)
    return float(np.std(increments))


def main():
    parser = argparse.ArgumentParser(description="Calibrate RFSV model from Oxford-Man data")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Path to Oxford-Man CSV (default: data/raw/oxfordmanrealizedvolatilityindices.csv)")
    parser.add_argument("--rv-col", default="rv5",
                        help="Realized variance column name (default: rv5)")
    parser.add_argument("--ticker", default=None,
                        help="Filter to specific .SPX2, .FTSE, etc. (optional)")
    parser.add_argument("--lag-max", type=int, default=20,
                        help="Maximum lag for variogram (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: Data file not found at {args.file}")
        print(f"Please download the Oxford-Man Realized Library from:")
        print(f"  {DATA_URL}")
        print(f"and place the CSV in {RAW_DIR}/")
        sys.exit(1)

    print(f"Loading data from {args.file} ...")
    rv = load_oxford_man(args.file, rv_col=args.rv_col)
    print(f"  Loaded {len(rv)} daily observations ({rv.index[0].date()} to {rv.index[-1].date()})")

    # log-volatility = 0.5 * log(realized_variance)
    log_vol = 0.5 * np.log(rv.values)

    print(f"\nEstimating Hurst exponent (lag_max={args.lag_max}) ...")
    H = estimate_hurst(log_vol, lag_max=args.lag_max)

    print(f"\nEstimating vol-of-vol ...")
    nu = estimate_nu(log_vol)

    print(f"\n{'='*50}")
    print(f"Calibrated parameters:")
    print(f"  H   = {H:.4f}   (Hurst exponent)")
    print(f"  nu  = {nu:.4f}  (vol-of-vol)")
    print(f"{'='*50}")
    print(f"\nPaste into src/common/params.hpp:")
    print(f"  constexpr double H   = {H:.4f};")
    print(f"  constexpr double nu  = {nu:.4f};")


if __name__ == "__main__":
    main()
