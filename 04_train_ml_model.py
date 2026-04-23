"""
SCM Forecasting Accuracy — ML Training Pipeline
================================================
- Loads actuals from forecasting.db
- Engineers lag, rolling, calendar, and static features
- Runs a walk-forward (expanding-window) backtest over the last 52 weeks,
  retraining every 13 weeks (realistic SCM cadence)
- Trains a final model on all history
- Produces a 12-week recursive future forecast with empirical prediction intervals
- Writes real ML forecasts to ml_forecasts and future forecasts to future_forecasts
- Prints a per-category accuracy report

Model: sklearn HistGradientBoostingRegressor
   - Global model (all SKU × warehouse series trained together)
   - Handles NaNs natively (useful at series starts)
   - Fast: ~2 sec per training cycle on this dataset
"""

import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "forecasting.db"
VENV_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"


if (
    VENV_PYTHON.exists()
    and Path(sys.executable) != VENV_PYTHON
    and os.environ.get("SCM_FORECASTING_BOOTSTRAPPED") != "1"
):
    os.environ["SCM_FORECASTING_BOOTSTRAPPED"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
sales = pd.read_sql(
    """
    SELECT s.product_id,
           s.warehouse_id,
           s.week_start_date,
           s.units_sold,
           s.promotion_flag,
           p.category_id,
           p.unit_price,
           p.unit_cost,
           p.safety_stock_units
    FROM   sales_history s
    JOIN   products      p ON s.product_id = p.product_id
    ORDER BY s.product_id, s.warehouse_id, s.week_start_date
    """,
    conn,
    parse_dates=["week_start_date"],
)
print(f"Loaded {len(sales):,} rows of sales history "
      f"({sales['product_id'].nunique()} SKUs × "
      f"{sales['warehouse_id'].nunique()} DCs × "
      f"{sales['week_start_date'].nunique()} weeks).")


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling, and calendar features.
    All lag/rolling features use shift() so they never peek at the target.
    """
    df = (df.sort_values(["product_id", "warehouse_id", "week_start_date"])
            .reset_index(drop=True))

    grp = df.groupby(["product_id", "warehouse_id"])["units_sold"]

    # Lag features — last week, last month, last quarter, last year
    for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling stats of past demand (shift(1) so current week excluded)
    shifted = grp.shift(1)
    for w in [4, 13]:
        df[f"roll_mean_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .rolling(w, min_periods=1).mean().reset_index(drop=True)
        )
        df[f"roll_std_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .rolling(w, min_periods=2).std().reset_index(drop=True)
        )

    # Calendar features
    dt = df["week_start_date"]
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"]       = dt.dt.month
    df["quarter"]     = dt.dt.quarter
    df["week_sin"]    = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["trend_idx"]   = ((dt - dt.min()).dt.days // 7).astype(int)

    return df


data = add_features(sales)

FEATURE_COLS = [
    # Identity / hierarchy (tree models handle integer IDs fine)
    "product_id", "warehouse_id", "category_id",
    # Static product attributes
    "unit_price", "unit_cost", "safety_stock_units",
    # Known-ahead event flag
    "promotion_flag",
    # Lags
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52",
    # Rolling
    "roll_mean_4", "roll_std_4", "roll_mean_13", "roll_std_13",
    # Calendar
    "week_of_year", "month", "quarter",
    "week_sin", "week_cos", "month_sin", "month_cos", "trend_idx",
]
TARGET_COL = "units_sold"


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        max_depth=7,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 3. Walk-forward backtest: last 52 weeks, retrained every 13 weeks
# ---------------------------------------------------------------------------
all_weeks = data["week_start_date"].sort_values().unique()
n_weeks = len(all_weeks)

# 4 quarterly cut points spanning the last year
cut_indices = [n_weeks - 52, n_weeks - 39, n_weeks - 26, n_weeks - 13]

print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (retrain every 13 weeks, expanding window)")
print("=" * 70)

backtest_preds = []
for i, cut_idx in enumerate(cut_indices, start=1):
    cut_date = all_weeks[cut_idx]
    next_idx = min(cut_idx + 13, n_weeks)
    horizon_end_date = all_weeks[next_idx - 1]

    train = data[data["week_start_date"] < cut_date].dropna(subset=["lag_52"])
    test  = data[(data["week_start_date"] >= cut_date) &
                 (data["week_start_date"] <= horizon_end_date)]

    model = new_model()
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    preds = np.clip(model.predict(test[FEATURE_COLS]), 0, None)

    # MAPE on this horizon (excluding zero actuals)
    mask = test[TARGET_COL] > 0
    mape = np.mean(
        np.abs(test.loc[mask, TARGET_COL].values - preds[mask.values])
        / test.loc[mask, TARGET_COL].values
    ) * 100
    print(f"  Retrain #{i}: train≤{pd.Timestamp(cut_date).date()} "
          f"(n={len(train):,})   "
          f"predict {pd.Timestamp(cut_date).date()} → {pd.Timestamp(horizon_end_date).date()}   "
          f"MAPE={mape:.2f}%")

    out = test[["product_id", "warehouse_id", "week_start_date"]].copy()
    out["prediction"] = preds.round().astype(int)
    backtest_preds.append(out)

backtest_df = pd.concat(backtest_preds, ignore_index=True)

# ---------------------------------------------------------------------------
# 4. Final model trained on all available data
# ---------------------------------------------------------------------------
print("\nTraining final model on all history...")
full_train = data.dropna(subset=["lag_52"])
final_model = new_model()
final_model.fit(full_train[FEATURE_COLS], full_train[TARGET_COL])
print(f"  Final model trained on {len(full_train):,} rows.")

# Empirical residuals for prediction intervals
train_pred = final_model.predict(full_train[FEATURE_COLS])
residuals = full_train[TARGET_COL].values - train_pred
resid_std = float(np.std(residuals))
print(f"  Residual std (all series): {resid_std:.2f} units  →  "
      f"±1.44σ used for 85% prediction interval")

# ---------------------------------------------------------------------------
# 5. Recursive 12-week future forecast
# ---------------------------------------------------------------------------
N_FUTURE_WEEKS = 12
last_date = pd.Timestamp(all_weeks[-1])
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1),
                             periods=N_FUTURE_WEEKS, freq="W-MON")

# Static attributes per SKU×DC
static_cols = ["product_id", "warehouse_id", "category_id",
               "unit_price", "unit_cost", "safety_stock_units"]
series = data[static_cols].drop_duplicates().reset_index(drop=True)

# Build a scaffold with a row per (series × future week), units_sold=NaN
future_rows = (series.assign(key=1)
               .merge(pd.DataFrame({"week_start_date": future_dates, "key": 1}), on="key")
               .drop(columns="key"))
future_rows["units_sold"]     = np.nan
future_rows["promotion_flag"] = 0   # assume no planned promos in future

# Concatenate history with future scaffold; features regenerated inside the loop
hist_cols = static_cols + ["week_start_date", "units_sold", "promotion_flag"]
combined = pd.concat([data[hist_cols], future_rows[hist_cols]], ignore_index=True)

print(f"\nGenerating recursive {N_FUTURE_WEEKS}-week future forecast...")
for wk in future_dates:
    # Regenerate features on the combined frame (lags now see predicted values)
    feat = add_features(combined)
    mask = feat["week_start_date"] == wk
    X = feat.loc[mask, FEATURE_COLS]
    yhat = np.clip(final_model.predict(X), 0, None)

    # Write predictions back into combined so they serve as lag inputs next iteration
    idx = combined.index[combined["week_start_date"] == wk]
    combined.loc[idx, "units_sold"] = yhat.round()

future_pred = (combined[combined["week_start_date"] >= future_dates[0]]
               .loc[:, ["product_id", "warehouse_id", "week_start_date", "units_sold"]]
               .rename(columns={"units_sold": "forecasted_units"})
               .assign(forecasted_units=lambda d: d["forecasted_units"].astype(int),
                       lower_bound=lambda d: np.clip(d["forecasted_units"] - 1.44 * resid_std, 0, None).round().astype(int),
                       upper_bound=lambda d: (d["forecasted_units"] + 1.44 * resid_std).round().astype(int),
                       model_name="HistGradientBoosting",
                       prediction_date=last_date.date().isoformat())
               .reset_index(drop=True))

print(f"  Generated {len(future_pred):,} future forecast rows "
      f"({N_FUTURE_WEEKS} weeks × {len(series)} series).")

# ---------------------------------------------------------------------------
# 6. Write results to the database
# ---------------------------------------------------------------------------
print("\nWriting results to database...")

# 6a. Replace ml_forecasts for backtested weeks with REAL predictions.
#     Strategy: delete rows in the backtest period (per SKU×DC×week), insert fresh.
backtest_df["forecast_week_date"] = backtest_df["week_start_date"].dt.date.astype(str)
backtest_df["forecasted_units"]   = backtest_df["prediction"].astype(int)
backtest_df["lower_bound"]        = np.clip(
    (backtest_df["forecasted_units"] - 1.44 * resid_std).round(), 0, None
).astype(int)
backtest_df["upper_bound"]        = (
    (backtest_df["forecasted_units"] + 1.44 * resid_std).round()
).astype(int)
backtest_df["model_name"]         = "HistGradientBoosting"
backtest_df["prediction_date"]    = last_date.date().isoformat()

# Delete the synthetic rows we want to overwrite
conn.executemany(
    "DELETE FROM ml_forecasts WHERE product_id=? AND warehouse_id=? AND forecast_week_date=?",
    [(int(r.product_id), int(r.warehouse_id), r.forecast_week_date)
     for r in backtest_df.itertuples(index=False)],
)
# Re-insert with real predictions; PK gets auto-assigned via NULL
conn.executemany(
    """
    INSERT INTO ml_forecasts
        (product_id, warehouse_id, forecast_week_date, forecasted_units,
         lower_bound, upper_bound, model_name, prediction_date)
    VALUES (?,?,?,?,?,?,?,?)
    """,
    [(int(r.product_id), int(r.warehouse_id), r.forecast_week_date,
      int(r.forecasted_units), int(r.lower_bound), int(r.upper_bound),
      r.model_name, r.prediction_date)
     for r in backtest_df.itertuples(index=False)],
)
conn.commit()
print(f"  Replaced {len(backtest_df):,} ml_forecasts rows with real backtest predictions.")

# 6b. Future forecasts → new table
conn.execute("DROP TABLE IF EXISTS future_forecasts")
conn.execute("""
    CREATE TABLE future_forecasts (
        forecast_id         INTEGER PRIMARY KEY,
        product_id          INTEGER NOT NULL,
        warehouse_id        INTEGER NOT NULL,
        forecast_week_date  DATE NOT NULL,
        forecasted_units    INTEGER NOT NULL,
        lower_bound         INTEGER,
        upper_bound         INTEGER,
        model_name          TEXT,
        prediction_date     DATE,
        FOREIGN KEY (product_id)   REFERENCES products(product_id),
        FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
    )
""")
future_pred["forecast_week_date"] = future_pred["week_start_date"].dt.date.astype(str)
future_pred["forecast_id"] = np.arange(1, len(future_pred) + 1)
future_pred[[
    "forecast_id", "product_id", "warehouse_id", "forecast_week_date",
    "forecasted_units", "lower_bound", "upper_bound", "model_name", "prediction_date"
]].to_sql("future_forecasts", conn, if_exists="append", index=False)
conn.commit()
print(f"  Wrote {len(future_pred):,} future_forecasts rows.")

# ---------------------------------------------------------------------------
# 7. Accuracy report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL ACCURACY REPORT")
print("=" * 70)

report = pd.read_sql("""
    SELECT category_name,
           COUNT(*)                   AS n,
           ROUND(AVG(manual_mape), 2) AS manual_mape_pct,
           ROUND(AVG(ml_mape),     2) AS ml_mape_pct,
           ROUND(AVG(manual_mape) - AVG(ml_mape), 2) AS improvement_pp
    FROM   v_forecast_accuracy
    WHERE  manual_mape IS NOT NULL AND ml_mape IS NOT NULL
      AND  week_start_date >= ?
    GROUP  BY category_name
    ORDER  BY manual_mape_pct DESC
""", conn, params=[pd.Timestamp(all_weeks[cut_indices[0]]).date().isoformat()])
print("\nBacktest period only (last 52 weeks):")
print(report.to_string(index=False))

overall = pd.read_sql("""
    SELECT ROUND(AVG(manual_mape), 2) AS manual_mape,
           ROUND(AVG(ml_mape),     2) AS ml_mape
    FROM   v_forecast_accuracy
    WHERE  manual_mape IS NOT NULL AND ml_mape IS NOT NULL
      AND  week_start_date >= ?
""", conn, params=[pd.Timestamp(all_weeks[cut_indices[0]]).date().isoformat()]).iloc[0]

print(f"\nOverall backtest: "
      f"Manual MAPE {overall['manual_mape']}%  →  "
      f"ML MAPE {overall['ml_mape']}%  "
      f"(−{overall['manual_mape'] - overall['ml_mape']:.2f} pp, "
      f"{(overall['manual_mape'] - overall['ml_mape']) / overall['manual_mape'] * 100:.1f}% relative)")

# Feature importance proxy: permutation-free gain via split points
# HistGradientBoostingRegressor exposes it indirectly; we'll just report a sklearn-style summary
print("\nTop 10 features (by training-time feature_importances_ if available):")
try:
    # HistGB doesn't expose feature_importances_ — use sample-based permutation on a tiny slice
    from sklearn.inspection import permutation_importance
    sample = full_train.sample(min(2000, len(full_train)), random_state=0)
    pi = permutation_importance(final_model, sample[FEATURE_COLS], sample[TARGET_COL],
                                n_repeats=3, random_state=0, n_jobs=1)
    imp = (pd.Series(pi.importances_mean, index=FEATURE_COLS)
           .sort_values(ascending=False).head(10))
    for name, val in imp.items():
        print(f"  {name:<20}  {val:>8.3f}")
except Exception as e:
    print(f"  (skipped: {e})")

conn.close()
print("\n✓ ML training pipeline complete.")
