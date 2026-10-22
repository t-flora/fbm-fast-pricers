"""
Block 1: Calibrate Hurst exponent (H) and vol-of-vol (nu) from Oxford-Man data.

Reads the Oxford-Man Realized Library CSV from data/raw/ (or builds a proxy
from yfinance daily returns), and fits the RFSV log-vol variogram

    E[(log s_{t+D} - log s_t)^2] = nu^2 * D^{2H}

by OLS on a log-log scale: slope = 2H, intercept = 2 log(nu).  D is measured
in years, matching the C++ engine (T = 1, dt = 1/N), so the printed nu can be
pasted into params.hpp directly.

Usage:
    uv run python data/calibrate.py [--ticker .SPX] [--lag-max 20]
    uv run python data/calibrate.py --source yfinance [--ticker ^GSPC]

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


def load_oxford_man(filepath: str, rv_col: str = "rv5", symbol: str = ".SPX") -> pd.Series:
    """
    Load Oxford-Man CSV and return a Series of realized variance for one index.

    The published CSV is in long format: one row per (date, Symbol), with all
    ~30 indices stacked.  It must be filtered to a single Symbol, otherwise the
    variogram would difference across unrelated indices.

    The first (unnamed) column holds local midnight with the exchange's UTC offset,
    e.g. "2000-01-03 00:00:00+01:00".  The index of the result is that local calendar
    date (tz-naive): converting to UTC would move every date east of Greenwich back to
    the previous day.  Missing and non-positive values are dropped, since the variogram
    takes log(rv).
    """
    df = pd.read_csv(filepath, index_col=0)
    if "Symbol" in df.columns:
        symbols = sorted(df["Symbol"].unique())
        if symbol not in symbols:
            raise ValueError(f"Symbol '{symbol}' not found. Available: {symbols}")
        df = df[df["Symbol"] == symbol]
    # Wall-clock date in the exchange's own time zone (offsets change with DST)
    df.index = pd.DatetimeIndex([pd.Timestamp(s).tz_localize(None) for s in df.index])
    df = df.sort_index()
    if rv_col not in df.columns:
        available = [c for c in df.columns if "rv" in c.lower()]
        raise ValueError(f"Column '{rv_col}' not found. Available RV columns: {available}")
    rv = df[rv_col].dropna()
    return rv[rv > 0]


def fit_variogram(log_vol: np.ndarray, step_years: float,
                  lag_max: int = 20) -> tuple[float, float]:
    """
    Estimate (H, nu) from E[(X(t+D) - X(t))^2] = nu^2 D^{2H}.

    step_years: time between consecutive observations, in years.  The lag
    D = lag * step_years must be in the same units as the pricer's time axis,
    otherwise nu is off by a factor (step)^H (e.g. 252^0.1 = 1.74 for days).
    """
    lags = np.arange(1, lag_max + 1)
    variogram = np.array([
        np.mean((log_vol[lag:] - log_vol[:-lag]) ** 2)
        for lag in lags
    ])
    result = stats.linregress(np.log(lags * step_years), np.log(variogram))
    H = result.slope / 2.0
    nu = float(np.exp(result.intercept / 2.0))
    print(f"  Variogram regression: slope={result.slope:.4f}, R²={result.rvalue**2:.4f}")
    return H, nu


def main():
    parser = argparse.ArgumentParser(description="Calibrate RFSV model from Oxford-Man data or yfinance")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Path to Oxford-Man CSV (default: data/raw/oxfordmanrealizedvolatilityindices.csv)")
    parser.add_argument("--rv-col", default="rv5",
                        help="Realized variance column name (default: rv5)")
    parser.add_argument("--ticker", default=None,
                        help="Oxford-Man Symbol (default .SPX) or yfinance ticker (default ^GSPC)")
    parser.add_argument("--lag-max", type=int, default=20,
                        help="Maximum lag for variogram (default: 20)")
    parser.add_argument("--source", choices=["oxfordman", "yfinance"], default="oxfordman",
                        help="Data source: 'oxfordman' (default) or 'yfinance' (proxy RV from squared returns)")
    args = parser.parse_args()

    if args.source == "yfinance":
        yf_ticker = args.ticker or "^GSPC"
        print(f"Fetching proxy RV from yfinance ({yf_ticker}) ...")
        window = 5
        rv = load_yfinance_rv(yf_ticker, window=window)
        step_years = window / 252.0
        print(f"  Loaded {len(rv)} {window}-day windows ({rv.index[0].date()} to {rv.index[-1].date()})")
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
        symbol = args.ticker or ".SPX"
        rv = load_oxford_man(args.file, rv_col=args.rv_col, symbol=symbol)
        step_years = 1.0 / 252.0
        print(f"  Loaded {symbol}: {len(rv)} daily observations ({rv.index[0].date()} to {rv.index[-1].date()})")

    # log-volatility = 0.5 * log(realized_variance)
    log_vol = 0.5 * np.log(rv.values)

    print(f"\nFitting log-vol variogram (lag_max={args.lag_max}) ...")
    H, nu = fit_variogram(log_vol, step_years, lag_max=args.lag_max)
    nu_daily = nu * (1.0 / 252.0) ** H   # same fit with D measured in trading days

    print(f"\n{'='*50}")
    print(f"Calibrated parameters:")
    print(f"  H   = {H:.4f}   (Hurst exponent)")
    print(f"  nu  = {nu:.4f}  (vol-of-vol, time in years: used by the C++ engine)")
    print(f"        {nu_daily:.4f}  (same, time in trading days, as quoted by Gatheral et al.)")
    print(f"{'='*50}")
    print(f"\nPaste into src/common/params.hpp:")
    print(f"  constexpr double H   = {H:.4f};")
    print(f"  constexpr double nu  = {nu:.4f};")


if __name__ == "__main__":
    main()
