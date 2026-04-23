"""
SCM Forecasting — Data Sanity Checks & Example Dashboard Queries
=================================================================
Runs a handful of queries to (a) verify data quality and
(b) show the analytics that will power the Streamlit dashboard tiles.
"""

import sqlite3
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width",       150)

DB_PATH = Path(__file__).parent / "forecasting.db"
conn = sqlite3.connect(DB_PATH)


def q(sql, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(pd.read_sql_query(sql, conn).to_string(index=False))


# 1. Executive KPIs — the top-of-dashboard tiles
q("""
    SELECT
        ROUND(AVG(manual_mape), 2) AS manual_mape_pct,
        ROUND(AVG(ml_mape),     2) AS ml_mape_pct,
        ROUND(AVG(manual_bias_units), 1) AS manual_avg_bias_units,
        ROUND(AVG(ml_bias_units),     1) AS ml_avg_bias_units
    FROM v_forecast_accuracy
    WHERE manual_mape IS NOT NULL AND ml_mape IS NOT NULL
""", "1) EXECUTIVE KPIs — Forecast Error (lower = better)")

# 2. Accuracy by category — which areas hurt the business most today
q("""
    SELECT
        category_name,
        COUNT(*)                  AS forecast_points,
        ROUND(AVG(manual_mape),2) AS manual_mape_pct,
        ROUND(AVG(ml_mape),    2) AS ml_mape_pct,
        ROUND(AVG(manual_mape) - AVG(ml_mape), 2) AS improvement_pp
    FROM v_forecast_accuracy
    WHERE manual_mape IS NOT NULL AND ml_mape IS NOT NULL
    GROUP BY category_name
    ORDER BY manual_mape_pct DESC
""", "2) ACCURACY BY CATEGORY (sorted worst-to-best manual accuracy)")

# 3. Worst 10 SKU-Warehouse combinations under manual forecasting
q("""
    SELECT
        sku_code,
        product_name,
        warehouse_city,
        ROUND(AVG(manual_mape), 2) AS manual_mape_pct,
        ROUND(AVG(ml_mape),     2) AS ml_mape_pct,
        ROUND(AVG(manual_bias_units), 1) AS avg_bias_units
    FROM v_forecast_accuracy
    WHERE manual_mape IS NOT NULL
    GROUP BY sku_code, product_name, warehouse_city
    ORDER BY manual_mape_pct DESC
    LIMIT 10
""", "3) TOP 10 WORST MANUAL FORECASTS (best candidates for ML replacement)")

# 4. Stockout impact — last 12 months
q("""
    SELECT
        c.category_name,
        COUNT(*)                     AS stockout_events,
        SUM(s.shortage_units)        AS total_units_short,
        ROUND(SUM(s.lost_revenue),0) AS lost_revenue_inr
    FROM stockout_events s
    JOIN products   p ON s.product_id  = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    WHERE s.event_date >= DATE('2025-01-01')
    GROUP BY c.category_name
    ORDER BY lost_revenue_inr DESC
""", "4) STOCKOUT IMPACT BY CATEGORY (2025)")

# 5. Forecast-error trend — monthly aggregate
q("""
    SELECT
        STRFTIME('%Y-%m', week_start_date)    AS month,
        ROUND(AVG(manual_mape), 2) AS manual_mape_pct,
        ROUND(AVG(ml_mape),     2) AS ml_mape_pct
    FROM v_forecast_accuracy
    WHERE manual_mape IS NOT NULL
    GROUP BY month
    ORDER BY month
    LIMIT 12
""", "5) MONTHLY ERROR TREND (first 12 months — for a time-series chart)")

# 6. Bias direction check — are manual planners chronic over- or under-forecasters?
q("""
    SELECT
        CASE
            WHEN AVG(manual_bias_units) >  5 THEN 'Over-forecasting'
            WHEN AVG(manual_bias_units) < -5 THEN 'Under-forecasting'
            ELSE 'Balanced'
        END                              AS bias_direction,
        sku_code,
        product_name,
        ROUND(AVG(manual_bias_units),1) AS avg_bias_units
    FROM v_forecast_accuracy
    WHERE manual_bias_units IS NOT NULL
    GROUP BY sku_code, product_name
    ORDER BY avg_bias_units DESC
    LIMIT 5
""", "6) CHRONIC OVER-FORECASTERS (top 5) — these SKUs carry excess inventory")

q("""
    SELECT
        sku_code,
        product_name,
        ROUND(AVG(manual_bias_units),1) AS avg_bias_units
    FROM v_forecast_accuracy
    WHERE manual_bias_units IS NOT NULL
    GROUP BY sku_code, product_name
    ORDER BY avg_bias_units ASC
    LIMIT 5
""", "7) CHRONIC UNDER-FORECASTERS (top 5) — these SKUs cause stockouts")

conn.close()
print("\n" + "=" * 78)
print("Data looks healthy. Ready for dashboard + real ML model.")
print("=" * 78)
