"""
GB half-hourly electricity demand forecasting using NESO open data.

Task: forecast National Demand (ND, MW) for each half-hour settlement period,
24 hours ahead (day-ahead), and compare XGBoost against two baselines.

Run:  python gb_demand_forecast.py            (downloads data on first run)
      python gb_demand_forecast.py --synthetic (code test only, fake data)

Leakage control: every feature for a target half-hour uses only information
available at least 24 hours earlier (lags >= 48 periods). Train/validation/test
are strict chronological splits; no shuffling.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Resource IDs come from https://www.neso.energy/data-portal/historic-demand-data
# Find each year's CSV link on that page and paste it here. The 2023 link is given;
# add the other years you want (2019-2024 recommended; 2020 is a COVID outlier).
URLS = {
    2023: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/"
          "resource/bf5ab335-9b40-4ea4-b93a-ab4af7bce003/download/demanddata_2023.csv",
    # 2022: "<paste link>", 2021: "<paste link>", 2019: "<paste link>", ...
}
DATA_DIR = "data"
OUT_DIR = "results"
STEPS_PER_DAY = 48


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for year, url in URLS.items():
        path = os.path.join(DATA_DIR, f"demanddata_{year}.csv")
        if not os.path.exists(path):
            print(f"Downloading {year} ...")
            urllib.request.urlretrieve(url, path)


def load_real():
    import glob
    download()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "demanddata*.csv")))
    if not files:
        sys.exit("No CSV files found in ./data")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        # Some files carry an actual/forecast flag; keep actual outturn only
        if "FORECAST_ACTUAL_INDICATOR" in d.columns:
            d = d[d["FORECAST_ACTUAL_INDICATOR"] == "A"]
        keep = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "EMBEDDED_SOLAR_GENERATION",
                "EMBEDDED_SOLAR_CAPACITY", "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY"]
        d = d[keep].copy()
        # Files use two date formats (01-JAN-2019 and 2025-01-01). Parse each file with its
        # own explicit format: a generic dayfirst parse silently swaps day and month in ISO dates.
        raw = d["SETTLEMENT_DATE"].astype(str)
        parsed = None
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y"):
            try:
                parsed = pd.to_datetime(raw, format=fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            sys.exit(f"Unrecognised date format in {f}")
        d["SETTLEMENT_DATE"] = parsed
        # Sanity check: dates must advance by 0 or 1 day at a time (catches day/month swaps)
        steps = d["SETTLEMENT_DATE"].diff().dt.days.dropna()
        if steps.min() < 0 or steps.max() > 1:
            sys.exit(f"Date order problem in {f}: check the date format")
        frames.append(d)
        print(f"loaded {os.path.basename(f)}: {len(d)} rows")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["SETTLEMENT_DATE", "ND"])
    # Settlement period 1 = 00:00-00:30 UK local time. On clock-change days there are 46 or 50
    # periods; periods 49-50 on the long day collide with next day's 00:00-00:30 and are dropped.
    df["ts"] = df["SETTLEMENT_DATE"] + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    df = df.rename(columns={"ND": "y"})[["SETTLEMENT_PERIOD", "y", "EMBEDDED_SOLAR_GENERATION",
        "EMBEDDED_SOLAR_CAPACITY", "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY"]]
    # Put data on a complete half-hour grid so lags are true time offsets even across the
    # excluded 2020 gap; rows whose lags fall in a gap are dropped later.
    grid = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(grid)
    df["SETTLEMENT_PERIOD"] = (df.index.hour * 2 + df.index.minute // 30 + 1)
    return df


def load_synthetic():
    idx = pd.date_range("2021-01-01", "2023-12-31 23:30", freq="30min")
    rng = np.random.default_rng(0)
    h = idx.hour + idx.minute / 60
    y = (30000 + 6000 * np.sin((h - 6) / 24 * 2 * np.pi) + 4000 * np.cos(idx.dayofyear / 365 * 2 * np.pi)
         - 3000 * (idx.dayofweek >= 5) + rng.normal(0, 800, len(idx)))
    df = pd.DataFrame({"y": y, "SETTLEMENT_PERIOD": (idx.hour * 2 + idx.minute // 30 + 1)}, index=idx)
    return df


def uk_bank_holidays(index):
    """Approximate GB bank holidays via the `holidays` package if present."""
    try:
        import holidays
        h = holidays.UnitedKingdom(subdiv="ENG", years=sorted(set(index.year)))
        return pd.Series(index.normalize().isin(pd.to_datetime(list(h.keys()))).astype(int), index=index)
    except ImportError:
        print("Note: `holidays` not installed; bank-holiday feature set to 0 (pip install holidays).")
        return pd.Series(0, index=index)


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["period"] = df["SETTLEMENT_PERIOD"]
    X["dow"] = df.index.dayofweek
    X["month"] = df.index.month
    X["doy"] = df.index.dayofyear
    X["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    X["bank_holiday"] = uk_bank_holidays(df.index).values
    # Lags: same time yesterday (48), 2 days (96), last week (336). All >= 24h old.
    for lag in (48, 96, 336):
        X[f"lag_{lag}"] = df["y"].shift(lag)
    # Rolling stats of the previous day, shifted so they stop 24h before the target
    X["roll_mean_48"] = df["y"].shift(48).rolling(48).mean()
    X["roll_mean_336"] = df["y"].shift(48).rolling(336).mean()
    # Embedded (rooftop/small-scale) solar and wind sit inside ND, so they drive its shape.
    # Only information at least 24h old is used (shift 48).
    for c, name in (("EMBEDDED_SOLAR_GENERATION", "solar_gen"), ("EMBEDDED_SOLAR_CAPACITY", "solar_cap"),
                    ("EMBEDDED_WIND_GENERATION", "wind_gen"), ("EMBEDDED_WIND_CAPACITY", "wind_cap")):
        if c in df.columns:
            X[f"{name}_lag48"] = df[c].shift(48)
    return X


def metrics(y, p):
    mape = np.mean(np.abs((y - p) / y)) * 100
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    return {"MAPE_%": round(mape, 2), "MAE_MW": round(mae, 0), "RMSE_MW": round(rmse, 0)}


def main(synthetic=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_synthetic() if synthetic else load_real()
    X = make_features(df)
    data = pd.concat([df["y"], X], axis=1).dropna()

    # Chronological split: last 12 months = test, 6 months before = validation
    end = data.index.max()
    test_start = end - pd.DateOffset(months=12)
    val_start = test_start - pd.DateOffset(months=6)
    train = data[data.index < val_start]
    val = data[(data.index >= val_start) & (data.index < test_start)]
    test = data[data.index >= test_start]
    print(f"train {train.index.min().date()}..{train.index.max().date()}  n={len(train)}")
    print(f"val   {val.index.min().date()}..{val.index.max().date()}  n={len(val)}")
    print(f"test  {test.index.min().date()}..{test.index.max().date()}  n={len(test)}")

    all_feats = [c for c in data.columns if c != "y"]
    embedded = [c for c in all_feats if c.startswith(("solar_", "wind_"))]
    base_feats = [c for c in all_feats if c not in embedded]

    # Baselines
    base_naive_day = test["lag_48"]    # same half-hour yesterday
    base_naive_week = test["lag_336"]  # same half-hour last week (seasonal naive)

    def fit(feats):
        m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=50, n_jobs=-1, random_state=42)
        m.fit(train[feats], train["y"], eval_set=[(val[feats], val["y"])], verbose=False)
        return m, pd.Series(m.predict(test[feats]), index=test.index)

    model_b, pred_b = fit(base_feats)
    # The calendar + demand-lag model is the pre-specified headline model. The embedded
    # solar/wind variant is reported as an ablation, not chosen by looking at test results.
    model, pred, feats = model_b, pred_b, base_feats
    if embedded:
        model_e, pred_e = fit(all_feats)

    rows = {
        "Seasonal naive (last week)": metrics(test["y"], base_naive_week),
        "Naive (yesterday)": metrics(test["y"], base_naive_day),
        "XGBoost (calendar + demand lags)": metrics(test["y"], pred_b),
    }
    if embedded:
        rows["XGBoost (+ embedded solar/wind)"] = metrics(test["y"], pred_e)
    res = pd.DataFrame(rows).T
    print("\nTest-set results\n", res)
    res.to_csv(os.path.join(OUT_DIR, "test_metrics.csv"))

    best_base = min(rows["Seasonal naive (last week)"]["MAPE_%"], rows["Naive (yesterday)"]["MAPE_%"])
    final_key = "XGBoost (calendar + demand lags)"
    gain = (1 - rows[final_key]["MAPE_%"] / best_base) * 100
    print(f"\nXGBoost MAPE is {gain:.1f}% lower than the best baseline.")

    # Error by hour of day and by day type (where does the model struggle?)
    err = (test["y"] - pred).abs() / test["y"] * 100
    by_period = err.groupby(test["period"]).mean()
    by_dow = err.groupby(test["dow"]).mean()
    worst = err.sort_values(ascending=False).head(10)
    worst.to_csv(os.path.join(OUT_DIR, "worst_10_errors.csv"))
    print("\nMAPE by weekday (0=Mon):\n", by_dow.round(2).to_string())
    print("\nWorst 10 half-hours (MAPE %):\n", worst.round(1).to_string())

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feats).sort_values()
    imp.plot.barh(figsize=(6, 4)); plt.title("XGBoost feature importance"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150); plt.close()

    # One representative week: actual vs forecast
    wk = test.index[0] + pd.Timedelta(days=60)
    sl = slice(wk, wk + pd.Timedelta(days=7))
    plt.figure(figsize=(10, 4))
    plt.plot(test["y"][sl], label="Actual"); plt.plot(pred[sl], label="XGBoost (final)")
    plt.plot(base_naive_week[sl], label="Last week", alpha=0.5)
    plt.ylabel("National demand (MW)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "week_forecast.png"), dpi=150); plt.close()

    by_period.plot(figsize=(8, 3)); plt.ylabel("MAPE %"); plt.xlabel("Settlement period")
    plt.title("Error by time of day"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_by_period.png"), dpi=150); plt.close()
    print(f"\nSaved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="test the code on fake data")
    main(ap.parse_args().synthetic)
