"""
Phase 1: Implied Volatility Smile Validation.

Compares RFSV-model European call implied volatilities against SPY market IVs.

Steps:
  1. Download SPY spot + option chain for the two nearest expirations (via yfinance)
  2. Filter to liquid calls with valid bid/ask around ATM (±40% of spot)
  3. Compute market IV via Black-Scholes inversion (scipy.brentq)
  4. Price RFSV European calls (numpy circulant-FFT Monte Carlo)
  5. Convert RFSV prices to IVs via the same BS inversion
  6. Plot: market IV (scatter) vs RFSV IV (line) per expiration

Caveat: RFSV is calibrated to SPX realized variance (H=0.10, nu=0.30).
SPY is used here as the most liquid proxy; dividend effects are small on
short-dated options and are ignored in this pedagogical comparison.

Usage:
    uv run python data/validate_iv.py [--M 5000] [--N 63]

Output:
    plots/validate_iv.png
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_european_call, bs_call_price, bs_implied_vol

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# Calibrated RFSV parameters (from calibrate.py / params.hpp)
H_CALIB = 0.10
NU_CALIB = 0.30
R = 0.0          # simplified: ignore risk-free rate for pedagogical clarity


def fetch_spy_options(n_expirations: int = 2):
    """
    Download SPY spot price and option chains for the nearest n_expirations.
    Returns (spot, list_of_DataFrames_with_columns=[strike, mid, T, expiry]).
    """
    ticker = yf.Ticker("SPY")
    spot_data = ticker.history(period="1d")
    if spot_data.empty:
        raise RuntimeError("Could not download SPY spot price from yfinance.")
    spot = float(spot_data["Close"].iloc[-1])
    print(f"  SPY spot: ${spot:.2f}")

    expirations = ticker.options  # sorted nearest-first
    if not expirations:
        raise RuntimeError("No option expirations returned by yfinance.")

    today = date.today()
    results = []
    for exp_str in expirations:   # scan all until we have enough valid ones
        exp_date = date.fromisoformat(exp_str)
        T = (exp_date - today).days / 365.0
        if T < 7 / 365.0:   # skip weeklies < 1 week out
            continue

        chain = ticker.option_chain(exp_str)
        calls = chain.calls.copy()
        calls["T"] = T
        calls["expiry"] = exp_str
        calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])

        # Filter: valid bid, reasonable spread, within ±40% of spot
        calls = calls[
            (calls["bid"] > 0.01) &
            (calls["ask"] > calls["bid"]) &
            (calls["strike"] >= spot * 0.60) &
            (calls["strike"] <= spot * 1.40)
        ].copy()
        if len(calls) < 5:
            continue

        results.append(calls[["strike", "mid", "T", "expiry"]].reset_index(drop=True))
        if len(results) >= n_expirations:
            break

    return spot, results


def compute_market_ivs(calls_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Compute market IV for each row; drop NaN rows."""
    ivs = []
    for _, row in calls_df.iterrows():
        iv = bs_implied_vol(row["mid"], spot, row["strike"], row["T"], R)
        ivs.append(iv)
    calls_df = calls_df.copy()
    calls_df["market_iv"] = ivs
    return calls_df.dropna(subset=["market_iv"])


def calibrate_mu0(calls_df: pd.DataFrame, spot: float, H: float, nu: float,
                  M: int, N_per_year: int) -> float:
    """
    Estimate log-vol drift mu0 = log(sigma_eff) by matching the RFSV ATM price
    to the market ATM IV.  This decouples vol level from smile shape.
    """
    # Find the row closest to ATM
    atm_idx = (calls_df["strike"] - spot).abs().idxmin()
    atm = calls_df.loc[atm_idx]
    K_atm, T_atm, iv_mkt = atm["strike"], atm["T"], atm["market_iv"]
    if np.isnan(iv_mkt) or iv_mkt <= 0:
        return 0.0

    # Target price at ATM from market IV
    target_price = bs_call_price(spot, K_atm, T_atm, R, iv_mkt)
    N = max(int(round(N_per_year * T_atm)), 5)

    # Binary search over mu0 to match ATM price
    from scipy.optimize import brentq as _brentq

    def residual(mu0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = price_european_call(H=H, nu=nu, K=K_atm, T=T_atm, S0=spot, r=R,
                                    N=N, M=M, seed=42, mu0=mu0)
        return p - target_price

    try:
        # Initial vol level implied by market, used as starting guess
        sigma_mkt = iv_mkt
        mu0_guess = np.log(sigma_mkt)
        mu0 = _brentq(residual, mu0_guess - 3.0, mu0_guess + 3.0, xtol=0.01)
        print(f"  ATM calibration: market_IV={iv_mkt:.3f}  mu0={mu0:.3f}  "
              f"(sigma_0={np.exp(mu0):.3f})")
        return float(mu0)
    except Exception as e:
        print(f"  WARNING: ATM mu0 calibration failed ({e}), using mu0=0")
        return 0.0


def compute_rfsv_ivs(calls_df: pd.DataFrame, spot: float,
                     H: float, nu: float, M: int, N_per_year: int,
                     mu0: float = 0.0) -> pd.DataFrame:
    """Price RFSV European calls and invert to IV for each row."""
    rfsv_ivs = []
    for i, (_, row) in enumerate(calls_df.iterrows()):
        K = row["strike"]
        T = row["T"]
        N = max(int(round(N_per_year * T)), 5)   # at least 5 steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = price_european_call(H=H, nu=nu, K=K, T=T, S0=spot, r=R,
                                    N=N, M=M, seed=42 + i, mu0=mu0)
        iv = bs_implied_vol(p, spot, K, T, R)
        rfsv_ivs.append(iv)
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(calls_df)}] K={K:.0f}  T={T:.3f}  "
                  f"RFSV_price={p:.3f}  RFSV_IV={iv:.3f}")
    calls_df = calls_df.copy()
    calls_df["rfsv_iv"] = rfsv_ivs
    return calls_df


def plot_iv_smiles(dfs: list, spot: float, out_path: str, mu0s: dict = None):  # type: ignore[assignment]
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    mu0s = mu0s or {}

    for ax, df in zip(axes, dfs):
        exp = df["expiry"].iloc[0]
        T   = df["T"].iloc[0]
        mu0 = mu0s.get(exp, 0.0)
        sigma_eff = np.exp(mu0)
        moneyness = df["strike"] / spot

        # Shaded "intrinsic region" where deep-ITM prices give no IV
        rfsv_nan = df[df["rfsv_iv"].isna()]
        if not rfsv_nan.empty:
            cutoff = (rfsv_nan["strike"].max()) / spot
            ax.axvspan(moneyness.min() - 0.01, cutoff,
                       alpha=0.08, color="#e74c3c",
                       label="RFSV IV undefined (intrinsic)")

        # Market IV
        ax.scatter(moneyness, df["market_iv"] * 100, color="#2980b9", s=40, zorder=3,
                   label="Market IV (SPY)")

        # RFSV IV — filter NaN before plotting
        rfsv_ok = df.dropna(subset=["rfsv_iv"])
        if not rfsv_ok.empty:
            ax.plot(rfsv_ok["strike"] / spot, rfsv_ok["rfsv_iv"] * 100,
                    color="#e74c3c", linewidth=2, marker="o", markersize=4,
                    label=f"RFSV IV (H={H_CALIB}, nu={NU_CALIB})")

        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Moneyness  K/S")
        ax.set_ylabel("Implied Volatility (%)")
        ax.set_title(
            f"IV Smile — expiry {exp}  (T={T:.3f} yr)\n"
            f"RFSV scaled to  σ_eff={sigma_eff:.3f}"
        )

        # x-axis: round moneyness ticks
        ax.set_xticks([0.80, 0.90, 1.00, 1.10, 1.20])

        # y-axis: integer % gridlines at 10, 15, 20, 25
        iv_vals = pd.concat([df["market_iv"].dropna(), df["rfsv_iv"].dropna()]) * 100
        if not iv_vals.empty:
            y_lo = max(iv_vals.min() * 0.90, 5.0)
            y_hi = iv_vals.max() * 1.10
            ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.15)

        ax.legend(fontsize=9)

    fig.suptitle(
        f"RFSV Model vs SPY Market Implied Volatility  (H={H_CALIB}, nu={NU_CALIB})",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=3000,
                        help="Monte Carlo paths per option (default: 3000)")
    parser.add_argument("--N", type=int, default=63,
                        help="Time steps per year (default: 63 = quarterly)")
    parser.add_argument("--expirations", type=int, default=2,
                        help="Number of expirations to fetch (default: 2)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    out_path = os.path.join(out_dir, "validate_iv.png")

    print("Fetching SPY option chains ...")
    spot, chain_dfs = fetch_spy_options(n_expirations=args.expirations)

    if not chain_dfs:
        print("ERROR: No valid option chains returned. Check network/yfinance.")
        sys.exit(1)

    enriched_dfs = []
    mu0s = {}
    for df in chain_dfs:
        exp = df["expiry"].iloc[0]
        T = df["T"].iloc[0]
        print(f"\nExpiration {exp} (T={T:.3f} yr, {len(df)} strikes) ...")

        print("  Computing market IVs ...")
        df = compute_market_ivs(df, spot)
        print(f"  {len(df)} options with valid market IV")

        # calibrate_mu0 uses M paths for the ATM price (same as the full run).
        # At M=3000 the ATM price has std ≈ σ_payoff/√M ≈ 35/55 ≈ 0.6, so
        # mu0 uncertainty is roughly ±0.01 in log-vol units.
        print("  Calibrating mu0 (log-vol level) to ATM market IV ...")
        mu0 = calibrate_mu0(df, spot, H_CALIB, NU_CALIB, args.M, args.N)
        mu0s[exp] = mu0

        print("  Pricing RFSV European calls ...")
        df = compute_rfsv_ivs(df, spot, H_CALIB, NU_CALIB, args.M, args.N, mu0=mu0)
        enriched_dfs.append(df)

    print("\nPlotting ...")
    plot_iv_smiles(enriched_dfs, spot, out_path, mu0s=mu0s)

    # Print summary table
    for df in enriched_dfs:
        exp = df["expiry"].iloc[0]
        ok = df.dropna(subset=["market_iv", "rfsv_iv"])
        if ok.empty:
            continue
        atm = ok.iloc[(ok["strike"] - spot).abs().argsort()].iloc[0]
        print(f"\n  {exp}: ATM strike={atm['strike']:.1f}  "
              f"market_IV={atm['market_iv']:.3f}  RFSV_IV={atm['rfsv_iv']:.3f}")


if __name__ == "__main__":
    main()
