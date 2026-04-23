-- ==========================================================================
-- SCM Forecasting Accuracy — Database Schema (SQLite)
-- ==========================================================================
-- Star-like schema centered on the core forecasting story:
--   actuals (sales_history) vs manual_forecasts vs ml_forecasts
--   with downstream impact tracked via inventory_snapshots + stockout_events.
-- ==========================================================================

-- Product hierarchy
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE,
    department    TEXT NOT NULL
);

CREATE TABLE products (
    product_id         INTEGER PRIMARY KEY,
    sku_code           TEXT NOT NULL UNIQUE,
    product_name       TEXT NOT NULL,
    category_id        INTEGER NOT NULL,
    unit_price         REAL NOT NULL,    -- selling price in INR
    unit_cost          REAL NOT NULL,    -- COGS in INR
    lead_time_days     INTEGER NOT NULL, -- supplier lead time
    safety_stock_units INTEGER NOT NULL, -- policy buffer
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Network / locations
CREATE TABLE warehouses (
    warehouse_id   INTEGER PRIMARY KEY,
    warehouse_code TEXT NOT NULL UNIQUE,
    warehouse_name TEXT NOT NULL,
    region         TEXT NOT NULL,
    city           TEXT NOT NULL,
    capacity_units INTEGER NOT NULL
);

-- Actual weekly sales (ground truth)
CREATE TABLE sales_history (
    sale_id         INTEGER PRIMARY KEY,
    product_id      INTEGER NOT NULL,
    warehouse_id    INTEGER NOT NULL,
    week_start_date DATE NOT NULL,
    units_sold      INTEGER NOT NULL,
    revenue         REAL NOT NULL,
    promotion_flag  INTEGER DEFAULT 0,   -- 1 if week was on promo
    FOREIGN KEY (product_id)   REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE (product_id, warehouse_id, week_start_date)
);

-- Baseline manual forecasts (biased — this is the problem we're solving)
CREATE TABLE manual_forecasts (
    forecast_id        INTEGER PRIMARY KEY,
    product_id         INTEGER NOT NULL,
    warehouse_id       INTEGER NOT NULL,
    forecast_week_date DATE NOT NULL,
    forecasted_units   INTEGER NOT NULL,
    created_by         TEXT NOT NULL,    -- planner name
    created_date       DATE NOT NULL,    -- when the forecast was made
    FOREIGN KEY (product_id)   REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE (product_id, warehouse_id, forecast_week_date)
);

-- ML-generated forecasts (our solution)
CREATE TABLE ml_forecasts (
    forecast_id        INTEGER PRIMARY KEY,
    product_id         INTEGER NOT NULL,
    warehouse_id       INTEGER NOT NULL,
    forecast_week_date DATE NOT NULL,
    forecasted_units   INTEGER NOT NULL,
    lower_bound        INTEGER,          -- 85% prediction interval
    upper_bound        INTEGER,
    model_name         TEXT NOT NULL,
    prediction_date    DATE NOT NULL,
    FOREIGN KEY (product_id)   REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE (product_id, warehouse_id, forecast_week_date, model_name)
);

-- Weekly inventory position (downstream impact of forecast accuracy)
CREATE TABLE inventory_snapshots (
    snapshot_id     INTEGER PRIMARY KEY,
    product_id      INTEGER NOT NULL,
    warehouse_id    INTEGER NOT NULL,
    snapshot_date   DATE NOT NULL,
    on_hand_units   INTEGER NOT NULL,
    on_order_units  INTEGER DEFAULT 0,
    FOREIGN KEY (product_id)   REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE (product_id, warehouse_id, snapshot_date)
);

-- Stockouts = hard business cost of forecast errors
CREATE TABLE stockout_events (
    event_id        INTEGER PRIMARY KEY,
    product_id      INTEGER NOT NULL,
    warehouse_id    INTEGER NOT NULL,
    event_date      DATE NOT NULL,
    shortage_units  INTEGER NOT NULL,
    lost_revenue    REAL NOT NULL,
    FOREIGN KEY (product_id)   REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

-- Indexes for common dashboard queries
CREATE INDEX idx_sales_product_week     ON sales_history(product_id, week_start_date);
CREATE INDEX idx_sales_warehouse_week   ON sales_history(warehouse_id, week_start_date);
CREATE INDEX idx_manual_fc_product_week ON manual_forecasts(product_id, forecast_week_date);
CREATE INDEX idx_ml_fc_product_week     ON ml_forecasts(product_id, forecast_week_date);
CREATE INDEX idx_inv_product_date       ON inventory_snapshots(product_id, snapshot_date);
CREATE INDEX idx_stockout_product_date  ON stockout_events(product_id, event_date);

-- ==========================================================================
-- View: one-row-per-week side-by-side comparison of actuals, manual & ML
-- The dashboard's main fact table.
-- ==========================================================================
CREATE VIEW v_forecast_accuracy AS
SELECT
    s.product_id,
    p.sku_code,
    p.product_name,
    c.category_name,
    s.warehouse_id,
    w.warehouse_code,
    w.city AS warehouse_city,
    s.week_start_date,
    s.units_sold        AS actual_units,
    mf.forecasted_units AS manual_forecast,
    ml.forecasted_units AS ml_forecast,
    (mf.forecasted_units - s.units_sold) AS manual_bias_units,
    (ml.forecasted_units - s.units_sold) AS ml_bias_units,
    ABS(s.units_sold - mf.forecasted_units) AS manual_abs_error,
    ABS(s.units_sold - ml.forecasted_units) AS ml_abs_error,
    CASE WHEN s.units_sold > 0
         THEN ROUND(100.0 * ABS(s.units_sold - mf.forecasted_units) / s.units_sold, 2)
         ELSE NULL END AS manual_mape,
    CASE WHEN s.units_sold > 0
         THEN ROUND(100.0 * ABS(s.units_sold - ml.forecasted_units) / s.units_sold, 2)
         ELSE NULL END AS ml_mape,
    s.promotion_flag
FROM      sales_history     s
JOIN      products           p  ON s.product_id   = p.product_id
JOIN      categories         c  ON p.category_id  = c.category_id
JOIN      warehouses         w  ON s.warehouse_id = w.warehouse_id
LEFT JOIN manual_forecasts   mf ON s.product_id   = mf.product_id
                                AND s.warehouse_id = mf.warehouse_id
                                AND s.week_start_date = mf.forecast_week_date
LEFT JOIN ml_forecasts       ml ON s.product_id   = ml.product_id
                                AND s.warehouse_id = ml.warehouse_id
                                AND s.week_start_date = ml.forecast_week_date;
