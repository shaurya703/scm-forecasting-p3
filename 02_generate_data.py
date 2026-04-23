"""
SCM Forecasting Accuracy — Sample Data Generator
=================================================
Generates 3 years of weekly data for 30 SKUs across 4 Indian warehouses.
Writes everything to forecasting.db (SQLite).

Design goals:
- Realistic demand patterns: base level + seasonality + trend + noise + promos
- SKU-specific seasonality (winter jackets peak in Jan, swimwear in May, etc.)
- Manual forecasts have systematic BIAS (the problem we're solving)
- Synthetic ML forecasts are closer to truth (simulating good model output;
  the real model is trained in step 3 of the project)
- Inventory & stockouts flow naturally from forecast errors
"""

import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "forecasting.db"
SCHEMA_PATH = BASE_DIR / "01_schema.sql"

# ---------------------------------------------------------------------------
# Time axis: 156 Mondays starting 2 Jan 2023 → 3 full years of weekly data
# ---------------------------------------------------------------------------
START_DATE = pd.Timestamp("2023-01-02")
NUM_WEEKS  = 156
weeks = pd.date_range(START_DATE, periods=NUM_WEEKS, freq="W-MON")

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
categories = [
    (1, "Electronics",       "Consumer Tech"),
    (2, "Apparel",           "Fashion"),
    (3, "Home Goods",        "Household"),
    (4, "Food & Beverage",   "Groceries"),
    (5, "Sports & Outdoor",  "Lifestyle"),
]

# Product master + demand-generation parameters.
# Columns:
#   pid, sku, name, cat_id, price, cost, lead_time, safety_stock,
#   base_demand, seasonality_amplitude (0..1), peak_week (0..51),
#   yearly_trend_pct, noise_sigma, manual_forecast_bias_pct
products = [
    # Electronics — mild seasonality, holiday spike (week 45-50)
    (1,  "ELC-001", "Wireless Earbuds",   1, 2499, 1400, 14,  80, 120, 0.25, 50,  0.20, 0.15,  0.15),
    (2,  "ELC-002", "Smart Watch",        1, 8999, 5200, 21,  60,  80, 0.30, 50,  0.10, 0.20,  0.25),
    (3,  "ELC-003", "Bluetooth Speaker",  1, 3499, 1900, 14,  50,  70, 0.20, 30, -0.05, 0.18, -0.10),
    (4,  "ELC-004", "Phone Charger",      1,  799,  350,  7, 150, 200, 0.05,  0,  0.08, 0.12,  0.05),
    (5,  "ELC-005", "Laptop Stand",       1, 1599,  800, 10,  40,  50, 0.10, 10,  0.15, 0.22,  0.20),
    (6,  "ELC-006", "USB-C Cable",        1,  399,  150,  7, 250, 350, 0.03,  0,  0.05, 0.10, -0.15),
    # Apparel — strong seasonality
    (7,  "APP-001", "Winter Jacket",      2, 4999, 2800, 28,  30,  80, 0.80,  2,  0.00, 0.25,  0.35),
    (8,  "APP-002", "Summer T-Shirt",     2,  899,  400, 10, 100, 180, 0.70, 20,  0.05, 0.20, -0.20),
    (9,  "APP-003", "Running Shoes",      2, 3999, 2100, 21,  70, 100, 0.25, 14,  0.12, 0.18,  0.10),
    (10, "APP-004", "Jeans",              2, 2499, 1100, 14,  80, 110, 0.15, 40,  0.02, 0.15,  0.05),
    (11, "APP-005", "Wool Scarf",         2,  699,  250, 14,  20,  40, 0.95,  1,  0.00, 0.30,  0.40),
    (12, "APP-006", "Swimwear",           2, 1299,  550, 14,  25,  60, 0.90, 18,  0.08, 0.28, -0.25),
    # Home Goods
    (13, "HOM-001", "Coffee Maker",       3, 5999, 3300, 21,  40,  50, 0.15, 45,  0.10, 0.17,  0.15),
    (14, "HOM-002", "Bed Sheets",         3, 1499,  700, 10,  90, 130, 0.20, 45,  0.03, 0.14, -0.12),
    (15, "HOM-003", "Dinner Plate Set",   3, 2299, 1100, 14,  35,  45, 0.30, 44,  0.00, 0.22,  0.20),
    (16, "HOM-004", "Cookware Set",       3, 4499, 2400, 21,  25,  30, 0.35, 44, -0.05, 0.24,  0.30),
    (17, "HOM-005", "Desk Lamp",          3, 1299,  600, 10,  60,  80, 0.10, 35,  0.12, 0.16, -0.08),
    (18, "HOM-006", "Throw Pillow",       3,  699,  280,  7, 100, 120, 0.15, 40,  0.05, 0.18,  0.10),
    # Food & Beverage
    (19, "FNB-001", "Ice Cream Tub",      4,  299,  120,  5, 200, 400, 0.85, 18, -0.02, 0.15, -0.30),
    (20, "FNB-002", "Hot Chocolate Mix",  4,  399,  180,  7,  80, 150, 0.80,  2,  0.00, 0.17,  0.25),
    (21, "FNB-003", "Energy Drink",       4,  150,   70,  5, 300, 500, 0.10,  0,  0.15, 0.12,  0.05),
    (22, "FNB-004", "Protein Bar",        4,  120,   55,  7, 400, 600, 0.08, 10,  0.20, 0.14, -0.10),
    (23, "FNB-005", "Coffee Beans",       4,  899,  450, 14, 100, 140, 0.05, 10,  0.08, 0.11,  0.03),
    (24, "FNB-006", "Sparkling Water",    4,  199,   80,  5, 250, 400, 0.35, 20,  0.10, 0.15, -0.15),
    # Sports & Outdoor
    (25, "SPT-001", "Yoga Mat",           5, 1499,  700, 14,  50,  70, 0.20,  2,  0.18, 0.20,  0.15),
    (26, "SPT-002", "Tennis Racket",      5, 3999, 2100, 21,  20,  30, 0.60, 14,  0.05, 0.25,  0.30),
    (27, "SPT-003", "Ski Gloves",         5, 1999,  900, 21,  15,  40, 0.95,  2,  0.00, 0.35,  0.45),
    (28, "SPT-004", "Bike Helmet",        5, 2499, 1300, 14,  30,  50, 0.45, 14,  0.10, 0.22, -0.20),
    (29, "SPT-005", "Camping Tent",       5, 6999, 3800, 28,  15,  25, 0.70, 22, -0.08, 0.28,  0.25),
    (30, "SPT-006", "Dumbbells 10kg",     5, 2999, 1500, 21,  40,  50, 0.10,  2,  0.12, 0.18,  0.08),
]

# 4 Indian DCs (user is in Bengaluru — relatable context)
warehouses = [
    (1, "WH-BLR", "Bengaluru DC", "South", "Bengaluru", 50_000),
    (2, "WH-MUM", "Mumbai DC",    "West",  "Mumbai",    60_000),
    (3, "WH-DEL", "Delhi DC",     "North", "Delhi",     55_000),
    (4, "WH-KOL", "Kolkata DC",   "East",  "Kolkata",   40_000),
]

# Mumbai & Delhi = larger consumer markets
wh_multiplier = {1: 1.00, 2: 1.20, 3: 1.15, 4: 0.70}

planners = ["R. Sharma", "A. Iyer", "P. Mehta", "S. Nair"]

# ---------------------------------------------------------------------------
# Build the database
# ---------------------------------------------------------------------------
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA_PATH.read_text())

conn.executemany("INSERT INTO categories VALUES (?,?,?)", categories)
conn.executemany(
    "INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
    [p[:8] for p in products],
)
conn.executemany("INSERT INTO warehouses VALUES (?,?,?,?,?,?)", warehouses)
conn.commit()

# ---------------------------------------------------------------------------
# Generate time-series rows
# ---------------------------------------------------------------------------
sales_rows, manual_rows, ml_rows, inv_rows, stockout_rows = [], [], [], [], []
sale_id = mfc_id = mlfc_id = inv_id = so_id = 0

for p in products:
    (pid, sku, name, cat_id, price, cost, lt, ss,
     base_dem, amp, peak_wk, trend_yr, noise_sigma, manual_bias) = p

    for wh in warehouses:
        whid = wh[0]
        mult = wh_multiplier[whid]

        t = np.arange(NUM_WEEKS)
        # Yearly seasonality peaking at peak_wk
        seasonal = 1 + amp * np.cos(2 * np.pi * (t - peak_wk) / 52)
        seasonal = np.maximum(seasonal, 0.10)
        # Linear trend
        trend = 1 + trend_yr * (t / 52)
        trend = np.maximum(trend, 0.30)
        expected = base_dem * mult * seasonal * trend

        # Actuals = expected × lognormal noise × promo multiplier
        noise = np.exp(np.random.normal(0, noise_sigma, NUM_WEEKS))
        promo_flags = (np.random.random(NUM_WEEKS) < 0.03).astype(int)
        promo_mult = np.where(
            promo_flags == 1, np.random.uniform(2.0, 3.5, NUM_WEEKS), 1.0
        )
        actuals = np.maximum(
            np.round(expected * noise * promo_mult).astype(int), 0
        )

        # Manual forecast: biased + its own noise, doesn't anticipate promos
        manual_noise = np.exp(np.random.normal(0, 0.18, NUM_WEEKS))
        manual_fc = np.maximum(
            np.round(expected * (1 + manual_bias) * manual_noise).astype(int), 0
        )
        planner = planners[(pid + whid) % len(planners)]

        # Synthetic ML forecast: unbiased + tight noise (real model in notebook later)
        ml_noise = np.exp(np.random.normal(0, 0.08, NUM_WEEKS))
        ml_fc = np.maximum(np.round(expected * ml_noise).astype(int), 0)

        # Simple inventory loop: replenish to manual_forecast + safety_stock
        on_hand = ss + int(expected[0])
        for wi, wk in enumerate(weeks):
            wk_iso    = wk.date().isoformat()
            pred_iso  = (wk - timedelta(days=7)).date().isoformat()

            sale_id += 1
            sales_rows.append((
                sale_id, pid, whid, wk_iso,
                int(actuals[wi]), float(actuals[wi] * price), int(promo_flags[wi]),
            ))

            mfc_id += 1
            manual_rows.append((
                mfc_id, pid, whid, wk_iso,
                int(manual_fc[wi]), planner, pred_iso,
            ))

            mlfc_id += 1
            ml_rows.append((
                mlfc_id, pid, whid, wk_iso,
                int(ml_fc[wi]),
                max(0, int(ml_fc[wi] * 0.85)),
                int(ml_fc[wi] * 1.15),
                "SARIMA+XGBoost", pred_iso,
            ))

            # Inventory: replenish up to forecast + SS
            target = int(manual_fc[wi]) + ss
            replenish = max(0, target - on_hand)
            on_hand += replenish

            # Consume demand (stockout if short)
            if on_hand >= actuals[wi]:
                on_hand -= int(actuals[wi])
                shortage = 0
            else:
                shortage = int(actuals[wi]) - on_hand
                on_hand = 0

            inv_id += 1
            inv_rows.append((
                inv_id, pid, whid, wk_iso, int(on_hand), int(replenish),
            ))

            if shortage > 0:
                so_id += 1
                stockout_rows.append((
                    so_id, pid, whid, wk_iso,
                    int(shortage), float(shortage * price),
                ))

# Bulk insert
conn.executemany("INSERT INTO sales_history       VALUES (?,?,?,?,?,?,?)",   sales_rows)
conn.executemany("INSERT INTO manual_forecasts    VALUES (?,?,?,?,?,?,?)",   manual_rows)
conn.executemany("INSERT INTO ml_forecasts        VALUES (?,?,?,?,?,?,?,?,?)", ml_rows)
conn.executemany("INSERT INTO inventory_snapshots VALUES (?,?,?,?,?,?)",     inv_rows)
conn.executemany("INSERT INTO stockout_events     VALUES (?,?,?,?,?,?)",     stockout_rows)
conn.commit()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def n(sql):
    return conn.execute(sql).fetchone()[0]

print("=" * 60)
print("DATA GENERATION SUMMARY")
print("=" * 60)
print(f"Categories ........... {n('SELECT COUNT(*) FROM categories'):>10,}")
print(f"Products (SKUs) ...... {n('SELECT COUNT(*) FROM products'):>10,}")
print(f"Warehouses ........... {n('SELECT COUNT(*) FROM warehouses'):>10,}")
print(f"Sales rows ........... {n('SELECT COUNT(*) FROM sales_history'):>10,}")
print(f"Manual forecasts ..... {n('SELECT COUNT(*) FROM manual_forecasts'):>10,}")
print(f"ML forecasts ......... {n('SELECT COUNT(*) FROM ml_forecasts'):>10,}")
print(f"Inventory snapshots .. {n('SELECT COUNT(*) FROM inventory_snapshots'):>10,}")
print(f"Stockout events ...... {n('SELECT COUNT(*) FROM stockout_events'):>10,}")

print("\nForecast accuracy (all SKUs, all warehouses, all weeks):")
row = conn.execute("""
    SELECT ROUND(AVG(manual_mape), 2),
           ROUND(AVG(ml_mape),     2)
    FROM   v_forecast_accuracy
    WHERE  manual_mape IS NOT NULL AND ml_mape IS NOT NULL
""").fetchone()
manual_mape, ml_mape = row
print(f"  Manual MAPE .... {manual_mape}%")
print(f"  ML MAPE ........ {ml_mape}%")
print(f"  Improvement .... {round(manual_mape - ml_mape, 2)} percentage points"
      f"  ({round(100 * (manual_mape - ml_mape) / manual_mape, 1)}% relative)")

total_lost_rev = n("SELECT COALESCE(SUM(lost_revenue), 0) FROM stockout_events")
print(f"\nTotal lost revenue from stockouts (3 yrs): ₹{total_lost_rev:,.0f}")

print(f"\n✓ Database written to: {DB_PATH}")
conn.close()
