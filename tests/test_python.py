"""
Correctness tests for the Python engine and analysis helpers.

Run with:  uv run pytest tests/

Each test targets a property that a past bug broke or would break:
  - gamma(0) = dt^{2H} and the closed-form minimum circulant eigenvalue
    (the stability script once halved gamma(0) and "found" negative eigenvalues)
  - sample covariance of simulated fBM paths matches nu^2 C
  - at nu = 0 the European pricer reproduces Black-Scholes
  - the variogram calibration recovers known (H, nu), with time in years
  - the Python rSVD is close to the optimal truncation
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import calibrate  # noqa: E402
import params  # noqa: E402
import validate_stability as stab  # noqa: E402
from rfsv_model import (  # noqa: E402
    asian_cv_terms,
    bs_call_price,
    build_fgn_eigenvalues,
    price_asian_call,
    price_asian_call_cv,
    price_european_call,
    simulate_log_vol_paths,
)


@pytest.mark.parametrize("H", [0.05, 0.10, 0.30, 0.50])
@pytest.mark.parametrize("N", [64, 252])
def test_circulant_min_eigenvalue_closed_form(N, H):
    dt = 1.0 / N
    assert stab.fgn_cov_row(N, H, dt)[0] == pytest.approx(dt ** (2 * H))
    lam_stab = stab.circulant_eigenvalues(N, H, dt)
    lam_engine = build_fgn_eigenvalues(N, H, dt)
    np.testing.assert_allclose(lam_stab, lam_engine, rtol=1e-10, atol=1e-14)
    closed = dt ** (2 * H) * (N ** (2 * H) - (N - 1) ** (2 * H))
    assert lam_engine.min() > 0
    assert lam_engine.min() == pytest.approx(closed, rel=1e-8)


def test_simulated_paths_have_fbm_covariance():
    N, M, H, nu = 16, 100_000, 0.10, 0.5
    paths = simulate_log_vol_paths(N, M, H, nu, dt=1.0 / N, seed=3)
    S = paths.T @ paths / M
    C = nu ** 2 * stab.fbm_cov_matrix(N, H)
    tol = 6 * np.sqrt(2 / M) * np.abs(C).max()  # ~6 standard errors per entry
    assert np.abs(S - C).max() < tol
    assert np.abs(paths.mean(axis=0)).max() < 6 * nu / np.sqrt(M)


@pytest.mark.parametrize("martingale_correct", [False, True])
def test_european_price_matches_black_scholes_at_zero_vol_of_vol(martingale_correct):
    S0, K, T, sigma, M = 100.0, 100.0, 0.25, 0.20, 200_000
    p = price_european_call(H=0.1, nu=0.0, K=K, T=T, S0=S0, r=0.0, N=20, M=M,
                            seed=5, mu0=np.log(sigma),
                            martingale_correct=martingale_correct)
    se = S0 * sigma * np.sqrt(T) * 0.6 / np.sqrt(M)  # payoff std < 0.6 S0 sigma sqrt(T)
    assert abs(p - bs_call_price(S0, K, T, 0.0, sigma)) < 4 * se


def test_variogram_recovers_known_parameters():
    H_true, nu_true, step = 0.10, 0.30, 1.0 / 252
    log_vol = simulate_log_vol_paths(20_000, 4, H_true, nu_true, dt=step, seed=9)
    fits = [calibrate.fit_variogram(path, step_years=step, lag_max=20) for path in log_vol]
    H_hat, nu_hat = np.mean(fits, axis=0)
    assert H_hat == pytest.approx(H_true, abs=0.02)
    assert nu_hat == pytest.approx(nu_true, rel=0.15)


@pytest.mark.parametrize("k", [4, 16, 64])
def test_rsvd_near_optimal(k):
    C = stab.fbm_cov_matrix(200, 0.10)
    w = np.linalg.eigvalsh(C)[::-1]
    U, S, _ = stab.rsvd(C, k)
    err = np.linalg.norm(C - (U * S) @ U.T)
    assert err / np.sqrt((w[k:] ** 2).sum()) < 1.05


def test_python_params_match_cpp_params():
    import re
    hpp = open(os.path.join(os.path.dirname(__file__), "..", "src", "common", "params.hpp")).read()
    for py_name, cpp_name in [("H", "H"), ("NU", "nu"), ("SIGMA0", "sigma0"), ("S0", "S0"),
                              ("K", "K"), ("T", "T"), ("R", "r")]:
        m = re.search(rf"constexpr double {cpp_name}\s*=\s*([0-9.eE+-]+);", hpp)
        assert m, f"{cpp_name} not found in params.hpp"
        assert getattr(params, py_name) == pytest.approx(float(m.group(1))), py_name


def test_geometric_conditional_expectation_matches_mc():
    # Fix one volatility path; average the geometric payoff over many price shocks
    N, M, dt = 64, 200_000, 1.0 / 64
    one = simulate_log_vol_paths(N, 1, 0.10, 0.52, dt, seed=4) + params.MU0
    V, Y, EY = asian_cv_terms(np.repeat(one, M, axis=0), 100.0, 0.0, dt, 100.0, seed=8)
    assert np.allclose(EY, EY[0])
    assert abs(Y.mean() - EY[0]) < 4 * Y.std() / np.sqrt(M)


def test_control_variate_is_unbiased_and_effective():
    kw = dict(H=params.H, nu=params.NU, K=100.0, N=64, M=50_000)
    res = price_asian_call_cv(seed=3, **kw)
    # The plain part reuses exactly the paths of price_asian_call
    assert res["price_plain"] == pytest.approx(price_asian_call(seed=3, **kw), rel=1e-12)
    # CV and plain agree within error; variance reduction is large
    assert abs(res["price"] - res["price_plain"]) < 4 * res["se_plain"]
    assert (res["se_plain"] / res["se"]) ** 2 > 10
    # Independent runs agree within the (much smaller) CV error
    other = price_asian_call_cv(seed=4, **kw)
    assert abs(res["price"] - other["price"]) < 4 * np.hypot(res["se"], other["se"])


# Long format of the published Oxford-Man file: unnamed date column with the exchange's
# UTC offset (which changes with DST), indices stacked, rows not grouped by date
OXFORD_MAN_CSV = """\
,Symbol,rv5,rv10,bv,rk_parzen,open_time
2000-01-04 00:00:00+00:00,.FTSE,1.1e-04,1.2e-04,1.0e-04,1.3e-04,80000
2000-01-05 00:00:00-05:00,.SPX,2.5e-04,2.6e-04,2.4e-04,2.7e-04,93000
2000-01-04 00:00:00+09:00,.N225,9.0e-05,9.1e-05,8.9e-05,9.2e-05,90000
2000-01-03 00:00:00-05:00,.SPX,1.5e-04,1.6e-04,1.4e-04,1.7e-04,93000
2000-07-03 00:00:00-04:00,.SPX,3.5e-04,3.6e-04,3.4e-04,3.7e-04,93000
2000-01-04 00:00:00-05:00,.SPX,,2.1e-04,1.9e-04,2.2e-04,93000
2000-01-06 00:00:00-05:00,.SPX,0.0,3.1e-04,2.9e-04,3.2e-04,93000
2000-07-03 00:00:00+01:00,.FTSE,4.5e-04,4.6e-04,4.4e-04,4.7e-04,80000
"""


@pytest.fixture
def oxford_man_csv(tmp_path):
    path = tmp_path / "oxfordmanrealizedvolatilityindices.csv"
    path.write_text(OXFORD_MAN_CSV)
    return str(path)


def test_oxford_man_loader_filters_sorts_and_cleans(oxford_man_csv):
    import pandas as pd
    rv = calibrate.load_oxford_man(oxford_man_csv)  # defaults: rv5, .SPX
    # One symbol, sorted by date; the missing (NaN) and zero rv5 rows are dropped
    assert list(rv.index) == list(pd.to_datetime(["2000-01-03", "2000-01-05", "2000-07-03"]))
    np.testing.assert_allclose(rv.values, [1.5e-4, 2.5e-4, 3.5e-4])
    assert rv.index.is_monotonic_increasing and rv.index.tz is None

    rv10 = calibrate.load_oxford_man(oxford_man_csv, rv_col="rv10")
    np.testing.assert_allclose(rv10.values, [1.6e-4, 2.1e-4, 2.6e-4, 3.1e-4, 3.6e-4])


def test_oxford_man_loader_keeps_local_calendar_dates(oxford_man_csv):
    import pandas as pd
    # Local midnight east of Greenwich is the previous day in UTC; the date must not move
    n225 = calibrate.load_oxford_man(oxford_man_csv, symbol=".N225")
    assert list(n225.index) == [pd.Timestamp("2000-01-04")]
    ftse = calibrate.load_oxford_man(oxford_man_csv, symbol=".FTSE")
    assert list(ftse.index) == list(pd.to_datetime(["2000-01-04", "2000-07-03"]))
    np.testing.assert_allclose(ftse.values, [1.1e-4, 4.5e-4])


def test_oxford_man_loader_errors(oxford_man_csv):
    with pytest.raises(ValueError, match=r"Symbol '\.DJI' not found.*\.FTSE.*\.N225.*\.SPX"):
        calibrate.load_oxford_man(oxford_man_csv, symbol=".DJI")
    with pytest.raises(ValueError, match=r"Column 'rv1' not found.*rv5.*rv10"):
        calibrate.load_oxford_man(oxford_man_csv, rv_col="rv1")


def test_oxford_man_loader_single_index_file(tmp_path):
    # Older single-index exports have no Symbol column; the loader must still work
    path = tmp_path / "spx.csv"
    path.write_text(",rv5,bv\n2000-01-04 00:00:00-05:00,2e-4,1e-4\n"
                    "2000-01-03 00:00:00-05:00,1e-4,1e-4\n")
    rv = calibrate.load_oxford_man(str(path), symbol=".IGNORED")
    np.testing.assert_allclose(rv.values, [1e-4, 2e-4])
