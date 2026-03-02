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


def load_yfinance_rv(ticker: str = "^GSPC", start: str = "2000-01-01",
                     end: str = "2024-01-01", window: int = 5) -> pd.Series:
    """
    Compute proxy realized variance from non-overlapping window sums of squared returns.

    LIMITATION: Intraday (5-min) data is needed for reliable H estimation.
    Oxford-Man's 5-min RV gives H ≈ 0.10 for SPX with R² > 0.97.
    Daily squared returns are chi-squared(1) noisy; non-overlapping window RV
    (window=5 ~ 1 week) reduces noise while preserving temporal structure, but
    yields a rough estimate only.  Treat yfinance-calibrated H as approximate.

    Reference: Gatheral, Jaisson & Rosenbaum (2014) establish H ≈ 0.10 using
    Oxford-Man 5-min realized variance.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: uv add yfinance")

    raw = yf.download(ticker, start=start, end=end, progress=False)
    # Handle multi-level columns from newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    rv_daily = (log_ret ** 2).values

    # Non-overlapping window sums to reduce chi-sq noise without inducing autocorrelation
    n_windows = len(rv_daily) // window
    rv_windows = rv_daily[:n_windows * window].reshape(n_windows, window).sum(axis=1)

    # Reconstruct as a Series with dates at end of each window
    dates = log_ret.index[window - 1: n_windows * window: window]
    rv_proxy = pd.Series(rv_windows, index=dates[:n_windows], name="rv_proxy")
    rv_proxy = rv_proxy[rv_proxy > 1e-12]
    return rv_proxy


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
    parser = argparse.ArgumentParser(description="Calibrate RFSV model from Oxford-Man data or yfinance")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Path to Oxford-Man CSV (default: data/raw/oxfordmanrealizedvolatilityindices.csv)")
    parser.add_argument("--rv-col", default="rv5",
                        help="Realized variance column name (default: rv5)")
    parser.add_argument("--ticker", default=None,
                        help="Filter to specific .SPX2, .FTSE, etc. (Oxford-Man) or yfinance ticker")
    parser.add_argument("--lag-max", type=int, default=20,
                        help="Maximum lag for variogram (default: 20)")
    parser.add_argument("--source", choices=["oxfordman", "yfinance"], default="oxfordman",
                        help="Data source: 'oxfordman' (default) or 'yfinance' (proxy RV from squared returns)")
    args = parser.parse_args()

    if args.source == "yfinance":
        yf_ticker = args.ticker or "^GSPC"
        print(f"Fetching proxy RV from yfinance ({yf_ticker}) ...")
        rv = load_yfinance_rv(yf_ticker)
        print(f"  Loaded {len(rv)} daily obs ({rv.index[0].date()} to {rv.index[-1].date()})")
        print("  NOTE: Using (log-return)^2 as proxy — coarser than 5-min Oxford-Man RV.")
    else:
        if not os.path.exists(args.file):
            print(f"ERROR: Data file not found at {args.file}")
            print(f"Please download the Oxford-Man Realized Library from:")
            print(f"  {DATA_URL}")
            print(f"and place the CSV in {RAW_DIR}/")
            print(f"\nAlternatively, run with --source yfinance for a proxy estimate.")
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
