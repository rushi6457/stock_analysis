# select_top30.py
import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import timedelta

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = list  # fallback: iterate normally

# ---------- I/O / Loader ----------
def detect_and_read_raw(file_path: str):
    """Read file without headers (header=None) to detect the real header row."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        raw = pd.read_excel(file_path, header=None, engine="openpyxl")
    elif ext == ".csv":
        # try multiple encodings if necessary
        try:
            raw = pd.read_csv(file_path, header=None, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            raw = pd.read_csv(file_path, header=None, encoding="latin1", on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return raw, ext

def find_header_row_with_column(raw_df: pd.DataFrame, col_name: str = "Close"):
    """Find first row index in raw_df that contains a cell equal to col_name."""
    header_row = None
    for i, row in raw_df.iterrows():
        # compare strings after stripping
        vals = [str(v).strip() for v in row.values]
        if col_name in vals:
            header_row = i
            break
    return header_row

def read_with_detected_header(file_path: str):
    raw, ext = detect_and_read_raw(file_path)
    header_row = find_header_row_with_column(raw, "Close")
    if header_row is None:
        raise ValueError("Could not detect header row containing 'Close'. Inspect your file.")
    # reload using header_row
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, header=header_row, engine="openpyxl")
    else:
        # try utf-8 then latin1
        try:
            df = pd.read_csv(file_path, header=header_row, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(file_path, header=header_row, encoding="latin1", on_bad_lines="skip")
    return df

def clean_numeric_columns(df: pd.DataFrame):
    # Trim whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize expected columns existence
    expected = ["Close", "Volume", "Symbol", "Date"]
    # Some files might have lowercase or slightly different names; try common mappings
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if "close" not in cols_lower and "adj close" in cols_lower:
        col_map[cols_lower["adj close"]] = "Close"

    # Apply mapping if any
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure Close & Volume exist
    if "Close" not in df.columns:
        raise KeyError("Expected 'Close' column not found after header detection.")
    if "Volume" not in df.columns:
        # create Volume if missing as NaN (some datasets may not have volume)
        df["Volume"] = np.nan

    # Strip commas and currency symbols from Close & Volume then convert to numeric
    df["Close"] = df["Close"].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip()
    df["Volume"] = df["Volume"].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Date parsing if exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with no Symbol and no Close
    if "Symbol" in df.columns:
        df = df.dropna(subset=["Close", "Symbol"], how="all")
    else:
        # If Symbol missing, try to infer or raise
        raise KeyError("Expected 'Symbol' column not found after header detection.")

    # Reset index
    df = df.reset_index(drop=True)
    return df

def load_consolidated_data(file):
    import pandas as pd
    import os

    # Determine extension
    if isinstance(file, str):
        ext = os.path.splitext(file)[1].lower()
    else:
        ext = getattr(file, "name", "").split(".")[-1].lower()

    # Read file
    if ext == "csv":
        df = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file, engine="openpyxl")
    else:
        return None  # unsupported file

    # Clean numeric columns
    for col in ["Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    return df


# ---------- Basic metrics & metrics helpers ----------
def compute_metrics_for_group(group: pd.DataFrame):
    g = group.sort_values("Date") if "Date" in group.columns else group.sort_index()
    prices = g["Close"].astype(float).values
    if len(prices) < 2:
        return None
    # time span in days (use Date if present)
    if "Date" in g.columns and pd.notna(g["Date"].iloc[0]) and pd.notna(g["Date"].iloc[-1]):
        n_days = (g["Date"].iloc[-1] - g["Date"].iloc[0]).days
    else:
        n_days = len(prices)  # fallback
    if n_days <= 0:
        return None
    years = max(n_days / 365.25, 1/365.25)
    # CAGR
    try:
        cagr = (prices[-1] / prices[0]) ** (1/years) - 1
    except Exception:
        cagr = np.nan
    # daily log returns
    daily_log = np.diff(np.log(prices))
    if daily_log.size == 0:
        return None
    ann_vol = np.std(daily_log, ddof=1) * np.sqrt(252)
    ann_log_return = np.nanmean(daily_log) * 252
    # max drawdown
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cum_max) / cum_max
    max_dd = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0
    return {"CAGR": float(cagr), "AnnVol": float(ann_vol), "AnnLogReturn": float(ann_log_return), "MaxDrawdown": float(max_dd)}

def monte_carlo_gbm(S0, mu, sigma, days=252*5, sims=1000, seed=None):
    """Simulate GBM (returns array shape (sims, days+1)). mu and sigma are annualized."""
    if seed is not None:
        np.random.seed(seed)
    dt = 1/252
    steps = int(days)
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    # generate increments
    increments = np.random.normal(loc=drift, scale=diffusion, size=(sims, steps))
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate([np.zeros((sims, 1)), log_paths], axis=1)
    price_paths = S0 * np.exp(log_paths)
    return price_paths

# ---------- Forecasting & ranking ----------
def rank_with_forecast(df: pd.DataFrame, top_n=30, sims=1000, days=252*5, risk_free_rate=0.04, sims_seed=None, verbose=False):
    """Compute Monte Carlo forecasts per symbol and rank by a composite score."""
    results = []
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    symbols = sorted(df["Symbol"].unique())
    iterator = tqdm(symbols, desc="Forecasting") if (verbose and hasattr(tqdm, "__call__")) else symbols
    for symbol in iterator:
        group = df[df["Symbol"] == symbol].dropna(subset=["Close"]).copy()
        if group.shape[0] < 10:
            continue
        # sort by date if available
        if "Date" in group.columns:
            group = group.sort_values("Date")
        else:
            group = group.sort_values("index", errors="ignore")
        metrics = compute_metrics_for_group(group)
        if metrics is None:
            continue
        prices = group["Close"].astype(float).values
        daily_log_ret = np.diff(np.log(prices))
        if daily_log_ret.size < 2:
            continue
        mu = float(np.nanmean(daily_log_ret) * 252)  # annualized drift (log)
        sigma = float(np.nanstd(daily_log_ret, ddof=1) * np.sqrt(252))  # annualized vol
        if sigma <= 0 or np.isnan(sigma):
            continue
        S0 = float(prices[-1])
        paths = monte_carlo_gbm(S0=S0, mu=mu, sigma=sigma, days=days, sims=sims, seed=sims_seed)
        terminal = paths[:, -1]
        total_return = (terminal / S0) - 1
        years = days / 252
        simulated_cagr = (terminal / S0) ** (1/years) - 1
        median_sim_cagr = float(np.median(simulated_cagr))
        prob_positive = float(np.mean(total_return > 0))
        prob_big_loss = float(np.mean(total_return < -0.3))
        # composite score (tweakable)
        score = (median_sim_cagr * 0.6) + ((1.0 / (metrics["AnnVol"] + 1e-9)) * 0.25) + ((1.0 - prob_big_loss) * 0.15)
        results.append({
            "Symbol": symbol,
            "Hist_CAGR": metrics["CAGR"],
            "AnnVol": metrics["AnnVol"],
            "MaxDrawdown": metrics["MaxDrawdown"],
            "MedianSimCAGR": median_sim_cagr,
            "Prob_Positive_5y": prob_positive,
            "Prob_Loss_>30%": prob_big_loss,
            "Score": score,
            "LastPrice": S0
        })
    out = pd.DataFrame(results)
    if out.empty:
        print("[WARNING] rank_with_forecast produced no results.")
        return out
    out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    return out.head(top_n)

def ultra_conservative_bucket(forecast_df: pd.DataFrame, min_prob_positive=0.7, max_prob_loss30=0.15, max_annvol=None):
    """Filter forecast dataframe into an ultra-conservative bucket."""
    if forecast_df.empty:
        return forecast_df
    df = forecast_df.copy()
    mask = (df["Prob_Positive_5y"] >= min_prob_positive) & (df["Prob_Loss_>30%"] <= max_prob_loss30)
    if max_annvol is not None:
        mask = mask & (df["AnnVol"] <= max_annvol)
    return df[mask].sort_values("Score", ascending=False).reset_index(drop=True)

# ---------- Backtesting ----------
def realized_return_over_period(df: pd.DataFrame, symbol: str, start_date, hold_days: int):
    group = df[df["Symbol"] == symbol].dropna(subset=["Close", "Date"]).sort_values("Date")
    if group.empty:
        return None
    start_date = pd.to_datetime(start_date)
    end_date = start_date + pd.Timedelta(days=hold_days)
    after_start = group[group["Date"] >= start_date]
    if after_start.empty:
        return None
    S0 = float(after_start["Close"].iloc[0])
    after_end = group[group["Date"] <= end_date]
    if after_end.empty:
        return None
    ST = float(after_end["Close"].iloc[-1])
    total_return = (ST / S0) - 1
    years = max(hold_days / 252, 1/252)
    cagr = (ST / S0) ** (1/years) - 1 if S0 > 0 else None
    window_prices = after_start[after_start["Date"] <= end_date]["Close"].values
    if window_prices.size == 0:
        max_dd = None
    else:
        cum_max = np.maximum.accumulate(window_prices)
        drawdowns = (window_prices - cum_max) / cum_max
        max_dd = float(np.min(drawdowns))
    return {"TotalReturn": float(total_return), "CAGR": float(cagr) if cagr is not None else None, "MaxDD": max_dd}

def backtest_selector(df: pd.DataFrame, selector_fn, lookback_years=5, hold_years=5, sims=250, pick_n=10, step_years=1, verbose=True):
    """Rolling backtest of selector_fn. selector_fn(hist_df, top_n, sims) -> forecast df with 'Symbol' column"""
    if "Date" not in df.columns:
        raise ValueError("DataFrame must have a 'Date' column for backtesting.")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError("Invalid Date column.")
    lookback_days = int(lookback_years * 365.25)
    hold_days = int(hold_years * 252)
    step_days = int(step_years * 365.25)
    start_cutoff = min_date + pd.Timedelta(days=lookback_days)
    end_cutoff = max_date - pd.Timedelta(days=hold_years * 365.25)
    if start_cutoff >= end_cutoff:
        raise ValueError("Not enough data to backtest with chosen lookback/hold periods.")
    cutoffs = []
    cur = start_cutoff
    while cur <= end_cutoff:
        cutoffs.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    per_run = []
    iterator = tqdm(cutoffs, desc="Backtest cutoffs") if (verbose and hasattr(tqdm, "__call__")) else cutoffs
    for cutoff in iterator:
        hist = df[df["Date"] <= cutoff].copy()
        if hist.shape[0] < 100:
            continue
        # selector function should accept hist and return forecast DF
        try:
            forecast_df = selector_fn(hist, top_n=pick_n, sims=sims, days=252*5, sims_seed=42, verbose=False)
        except Exception as e:
            print(f"[WARN] selector_fn failed at cutoff {cutoff}: {e}")
            continue
        if forecast_df is None or forecast_df.empty:
            continue
        picks = forecast_df["Symbol"].tolist()[:pick_n]
        realized = []
        for sym in picks:
            rr = realized_return_over_period(df, sym, cutoff + pd.Timedelta(days=1), hold_days)
            if rr is None:
                realized.append({"Symbol": sym, "TotalReturn": np.nan, "CAGR": np.nan, "MaxDD": np.nan})
            else:
                realized.append({"Symbol": sym, **rr})
        rdf = pd.DataFrame(realized)
        valid = rdf.dropna(subset=["CAGR"])
        mean_real = float(valid["CAGR"].mean()) if not valid.empty else np.nan
        hit_rate = float(np.mean(valid["TotalReturn"] > 0)) if not valid.empty else np.nan
        median_real = float(valid["CAGR"].median()) if not valid.empty else np.nan
        avg_maxdd = float(valid["MaxDD"].mean()) if not valid.empty else np.nan
        per_run.append({"CutoffDate": cutoff, "NumPicks": len(picks), "NumValidReal": valid.shape[0],
                        "MeanRealCAGR": mean_real, "MedianRealCAGR": median_real,
                        "HitRate_PosReturn": hit_rate, "AvgMaxDD": avg_maxdd})
    summary_df = pd.DataFrame(per_run).sort_values("CutoffDate").reset_index(drop=True)
    agg = {"Runs": len(summary_df),
           "Avg_MeanRealCAGR": float(summary_df["MeanRealCAGR"].mean()) if not summary_df.empty else np.nan,
           "Avg_HitRate": float(summary_df["HitRate_PosReturn"].mean()) if not summary_df.empty else np.nan,
           "Avg_AvgMaxDD": float(summary_df["AvgMaxDD"].mean()) if not summary_df.empty else np.nan}
    return summary_df, agg

# ---------- Selector wrappers ----------
def selector_for_backtest(hist_df, top_n=10, sims=500, days=252*5, sims_seed=None, verbose=False):
    return rank_with_forecast(hist_df, top_n=top_n, sims=sims, days=days, sims_seed=sims_seed, verbose=verbose)

def selector_conservative(hist_df, top_n=30, sims=2000, days=252*5, sims_seed=None, verbose=False):
    f = rank_with_forecast(hist_df, top_n=top_n*4, sims=sims, days=days, sims_seed=sims_seed, verbose=verbose)
    bucket = ultra_conservative_bucket(f, min_prob_positive=0.7, max_prob_loss30=0.15, max_annvol=1.0)
    return bucket.head(top_n)

# ---------- Main flow ----------
def main():
    # Get input file from argv or default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "Consolidated_Stocks_5Y.xlsx"
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    # Load data
    try:
        df = load_consolidated_data(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Quick forecast (faster defaults)
    print("\n[STEP] Running quick forecast (sims=1000)...")
    forecast_top = rank_with_forecast(df, top_n=50, sims=1000, days=252*5, sims_seed=123, verbose=True)
    print("\n[RESULT] Top forecasted (top 10):")
    if not forecast_top.empty:
        print(forecast_top.head(10).to_string(index=False))
    else:
        print("No forecast results.")
    forecast_top.to_csv("Top50_forecasted.csv", index=False)
    print("[INFO] Saved Top50_forecasted.csv")

    # Ultra-conservative safe bucket
    print("\n[STEP] Building ultra-conservative bucket...")
    safe_bucket = selector_conservative(df, top_n=30, sims=2000)
    if not safe_bucket.empty:
        print("\n[SAFE BUCKET] Top safe picks:")
        print(safe_bucket.head(20).to_string(index=False))
        safe_bucket.to_csv("SafeBucket.csv", index=False)
        print("[INFO] Saved SafeBucket.csv")
    else:
        print("[INFO] No stocks met ultra-conservative criteria.")

    # Run backtest (this can be slow) with conservative defaults for speed
    try:
        print("\n[STEP] Running rolling backtest (this may take a while)...")
        summary_df, agg = backtest_selector(df, selector_for_backtest,
                                           lookback_years=5, hold_years=5,
                                           sims=250, pick_n=10, step_years=1, verbose=True)
        print("\n[BACKTEST] Per-run summary (first 10 rows):")
        if not summary_df.empty:
            print(summary_df.head(10).to_string(index=False))
            summary_df.to_csv("Backtest_Summary.csv", index=False)
            print("[INFO] Saved Backtest_Summary.csv")
            print("\n[BACKTEST AGGREGATE]:", agg)
        else:
            print("[BACKTEST] No backtest runs produced results.")
    except Exception as e:
        print(f"[WARN] Backtest failed or was skipped: {e}")

    print("\n[ALL DONE] Outputs: Top50_forecasted.csv, SafeBucket.csv (if any), Backtest_Summary.csv (if any).")

if __name__ == "__main__":
    main()
