"""
SCM Forecasting Accuracy — Interactive Dashboard
=================================================
Run locally with:
    pip install -r requirements.txt
    streamlit run 05_dashboard.py

Tabs:
    1. Executive Overview     — headline KPIs, manual-vs-ML improvement
    2. Forecast Explorer      — per-SKU time-series deep-dive
    3. Category & DC Analysis — where the errors and losses concentrate
    4. Future Forecasts       — next 12 weeks ML forecast with prediction bands
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import streamlit as st

DB_PATH = Path(__file__).parent / "forecasting.db"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SCM Forecasting Accuracy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    [data-testid="stMetricValue"]   { font-size: 28px; }
    [data-testid="stMetricLabel"]   { font-size: 13px; color: #555; }
    [data-testid="stMetricDelta"] svg { display: none; }
    h1 { font-size: 1.9rem !important; margin-bottom: 0 !important; }
    .subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all():
    conn = sqlite3.connect(DB_PATH)
    acc = pd.read_sql("SELECT * FROM v_forecast_accuracy", conn, parse_dates=["week_start_date"])
    products   = pd.read_sql("SELECT * FROM products", conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)
    warehouses = pd.read_sql("SELECT * FROM warehouses", conn)
    stockouts  = pd.read_sql("SELECT * FROM stockout_events", conn, parse_dates=["event_date"])
    inventory  = pd.read_sql("SELECT * FROM inventory_snapshots", conn, parse_dates=["snapshot_date"])
    future     = pd.read_sql("SELECT * FROM future_forecasts", conn, parse_dates=["forecast_week_date"])
    conn.close()
    return acc, products, categories, warehouses, stockouts, inventory, future


acc, products, categories, warehouses, stockouts, inventory, future = load_all()

# Backtest period = where we have honest ML predictions
BACKTEST_START = acc["week_start_date"].max() - pd.Timedelta(weeks=52)

# ---------------------------------------------------------------------------
# Sidebar: global filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

cat_options = ["All"] + sorted(acc["category_name"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", cat_options, index=0)

wh_options = ["All"] + sorted(acc["warehouse_city"].unique().tolist())
selected_wh = st.sidebar.selectbox("Warehouse", wh_options, index=0)

min_date, max_date = acc["week_start_date"].min(), acc["week_start_date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(BACKTEST_START.date(), max_date.date()),
    min_value=min_date.date(), max_value=max_date.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = BACKTEST_START, max_date

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard compares **manual** vs **ML-generated** weekly "
    "demand forecasts. ML predictions over the last year are genuine "
    "walk-forward backtest outputs from a HistGradientBoosting model."
)

# Apply filters
f = acc[(acc["week_start_date"] >= start_date) &
        (acc["week_start_date"] <= end_date)].copy()
if selected_cat != "All":
    f = f[f["category_name"] == selected_cat]
if selected_wh != "All":
    f = f[f["warehouse_city"] == selected_wh]


def kpi_block(df):
    """Return manual_mape, ml_mape, improvement_pct, lost_rev_inr for a slice of acc."""
    m = df["manual_mape"].mean()
    ml = df["ml_mape"].mean()
    impr = m - ml
    rel = 100 * impr / m if m else 0
    return m, ml, impr, rel


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 SCM Forecasting Accuracy")
st.markdown(
    '<p class="subtitle">Quantifying the business impact of replacing manual '
    'forecasts with an ML-driven pipeline</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Executive Overview",
    "🔍 Forecast Explorer",
    "🗂️ Category & DC Analysis",
    "🔮 Future Forecasts",
])

# ===========================================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ===========================================================================
with tab1:
    st.subheader("Headline KPIs")
    m, ml, impr, rel = kpi_block(f.dropna(subset=["manual_mape", "ml_mape"]))

    # Stockout cost in the same filter slice
    so = stockouts.merge(products, on="product_id")
    so = so.merge(warehouses.rename(columns={"city": "warehouse_city"}), on="warehouse_id")
    so_f = so[(so["event_date"] >= start_date) & (so["event_date"] <= end_date)]
    if selected_cat != "All":
        so_f = so_f[so_f["category_id"].isin(
            categories.loc[categories["category_name"] == selected_cat, "category_id"]
        )]
    if selected_wh != "All":
        so_f = so_f[so_f["warehouse_city"] == selected_wh]
    total_lost = so_f["lost_revenue"].sum()
    total_events = len(so_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Manual Forecast MAPE", f"{m:.1f}%", help="Mean Absolute Percentage Error of planner forecasts")
    c2.metric("ML Forecast MAPE", f"{ml:.1f}%", delta=f"−{impr:.1f} pp", delta_color="inverse")
    c3.metric("Relative Improvement", f"{rel:.0f}%", help="(Manual − ML) / Manual")
    c4.metric("Stockout Events", f"{total_events:,}")
    c5.metric("Lost Revenue", f"₹{total_lost/1e7:.2f} Cr", help="Revenue lost to stockouts in filtered slice")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    # Monthly trend of MAPE — manual vs ML
    with col_a:
        st.markdown("**Forecast Error Trend (monthly MAPE)**")
        trend = (f.dropna(subset=["manual_mape", "ml_mape"])
                 .assign(month=lambda d: d["week_start_date"].dt.to_period("M").dt.to_timestamp())
                 .groupby("month")[["manual_mape", "ml_mape"]].mean().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["month"], y=trend["manual_mape"],
            mode="lines+markers", name="Manual",
            line=dict(color="#ef4444", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=trend["month"], y=trend["ml_mape"],
            mode="lines+markers", name="ML",
            line=dict(color="#10b981", width=3),
        ))
        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="MAPE (%)", xaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bias direction distribution
    with col_b:
        st.markdown("**Manual Forecast Bias (units vs actual)**")
        bias_dist = (f.dropna(subset=["manual_bias_units"])
                     .groupby(["sku_code"])["manual_bias_units"].mean()
                     .reset_index()
                     .assign(kind=lambda d: np.where(d["manual_bias_units"] > 5, "Over-forecasting",
                                            np.where(d["manual_bias_units"] < -5, "Under-forecasting", "Balanced"))))
        fig2 = px.histogram(
            bias_dist, x="manual_bias_units", nbins=20, color="kind",
            color_discrete_map={
                "Over-forecasting":  "#f59e0b",
                "Under-forecasting": "#ef4444",
                "Balanced":          "#10b981",
            },
        )
        fig2.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Avg bias (units/week)", yaxis_title="# SKUs",
            legend_title_text="", bargap=0.05,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**Business Case Summary**")
    st.info(
        f"Replacing manual forecasts with the ML model reduces error from "
        f"**{m:.1f}%** to **{ml:.1f}%** on the filtered slice — a **{rel:.0f}% relative "
        f"improvement**. Applied company-wide, this directly addresses the "
        f"**₹{total_lost/1e7:.1f} Cr** in lost revenue from stockouts and reduces "
        f"excess-inventory carrying cost on the over-forecasted SKUs."
    )

# ===========================================================================
# TAB 2 — FORECAST EXPLORER
# ===========================================================================
with tab2:
    st.subheader("Per-SKU Forecast Deep-Dive")

    left, right = st.columns([1, 3])

    with left:
        sku_options = (products.merge(categories, on="category_id")
                       .sort_values("sku_code")[["sku_code", "product_name", "category_name"]])
        if selected_cat != "All":
            sku_options = sku_options[sku_options["category_name"] == selected_cat]

        sku_label_map = {
            f"{r.sku_code} · {r.product_name}": r.sku_code
            for r in sku_options.itertuples(index=False)
        }
        sku_display = st.selectbox("SKU", list(sku_label_map.keys()))
        sku = sku_label_map[sku_display]

        city_options = sorted(acc["warehouse_city"].unique())
        sku_city = st.selectbox("Warehouse", city_options, index=0)

        sku_f = acc[(acc["sku_code"] == sku) & (acc["warehouse_city"] == sku_city)].copy()

        # SKU-level KPIs
        m2, ml2, impr2, rel2 = kpi_block(
            sku_f[sku_f["week_start_date"] >= BACKTEST_START].dropna(subset=["manual_mape", "ml_mape"])
        )
        st.metric("Manual MAPE (last yr)", f"{m2:.1f}%")
        st.metric("ML MAPE (last yr)", f"{ml2:.1f}%", delta=f"−{impr2:.1f} pp", delta_color="inverse")

        # Over / under forecaster flag
        avg_bias = sku_f["manual_bias_units"].mean()
        if avg_bias > 5:
            st.warning(f"⚠️ Chronic **over**-forecaster (+{avg_bias:.0f} units/wk)")
        elif avg_bias < -5:
            st.warning(f"⚠️ Chronic **under**-forecaster ({avg_bias:.0f} units/wk)")
        else:
            st.success(f"✓ Balanced ({avg_bias:+.0f} units/wk)")

    with right:
        st.markdown(f"**{sku_display} at {sku_city}** — actuals, manual forecast, ML forecast")
        sku_f_sorted = sku_f.sort_values("week_start_date")

        # Promo markers
        promos = sku_f_sorted[sku_f_sorted["promotion_flag"] == 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sku_f_sorted["week_start_date"], y=sku_f_sorted["actual_units"],
            mode="lines", name="Actual", line=dict(color="#1f2937", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=sku_f_sorted["week_start_date"], y=sku_f_sorted["manual_forecast"],
            mode="lines", name="Manual forecast", line=dict(color="#ef4444", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=sku_f_sorted["week_start_date"], y=sku_f_sorted["ml_forecast"],
            mode="lines", name="ML forecast", line=dict(color="#10b981", width=2),
        ))
        if len(promos):
            fig.add_trace(go.Scatter(
                x=promos["week_start_date"], y=promos["actual_units"],
                mode="markers", name="Promotion", marker=dict(color="#f59e0b", size=10, symbol="star"),
            ))
        # Shade backtest period
        fig.add_vrect(
            x0=BACKTEST_START, x1=sku_f_sorted["week_start_date"].max(),
            fillcolor="#3b82f6", opacity=0.06, layer="below", line_width=0,
            annotation_text="ML backtest window", annotation_position="top left",
        )
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None, yaxis_title="Units/week",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error comparison — small histogram
        st.markdown("**Weekly error distribution (backtest period)**")
        err_df = (sku_f_sorted[sku_f_sorted["week_start_date"] >= BACKTEST_START]
                  .melt(id_vars=["week_start_date"],
                        value_vars=["manual_bias_units", "ml_bias_units"],
                        var_name="method", value_name="bias")
                  .assign(method=lambda d: d["method"].map(
                      {"manual_bias_units": "Manual", "ml_bias_units": "ML"})))
        fig_h = px.histogram(
            err_df, x="bias", color="method", barmode="overlay", nbins=20,
            color_discrete_map={"Manual": "#ef4444", "ML": "#10b981"},
        )
        fig_h.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Forecast − Actual (units)", yaxis_title="# weeks",
            legend_title_text="",
        )
        fig_h.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_h, use_container_width=True)

# ===========================================================================
# TAB 3 — CATEGORY & DC ANALYSIS
# ===========================================================================
with tab3:
    st.subheader("Where does the ML model help most?")

    col1, col2 = st.columns(2)

    # Accuracy by category
    with col1:
        st.markdown("**MAPE by Category**")
        by_cat = (f.dropna(subset=["manual_mape", "ml_mape"])
                  .groupby("category_name")[["manual_mape", "ml_mape"]].mean().reset_index()
                  .sort_values("manual_mape", ascending=True))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=by_cat["category_name"], x=by_cat["manual_mape"],
            orientation="h", name="Manual", marker_color="#ef4444",
        ))
        fig.add_trace(go.Bar(
            y=by_cat["category_name"], x=by_cat["ml_mape"],
            orientation="h", name="ML", marker_color="#10b981",
        ))
        fig.update_layout(
            barmode="group", height=350, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="MAPE (%)", yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Accuracy by warehouse
    with col2:
        st.markdown("**MAPE by Warehouse**")
        by_wh = (f.dropna(subset=["manual_mape", "ml_mape"])
                 .groupby("warehouse_city")[["manual_mape", "ml_mape"]].mean().reset_index()
                 .sort_values("manual_mape", ascending=True))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=by_wh["warehouse_city"], x=by_wh["manual_mape"],
            orientation="h", name="Manual", marker_color="#ef4444",
        ))
        fig.add_trace(go.Bar(
            y=by_wh["warehouse_city"], x=by_wh["ml_mape"],
            orientation="h", name="ML", marker_color="#10b981",
        ))
        fig.update_layout(
            barmode="group", height=350, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="MAPE (%)", yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Worst 15 SKU × Warehouse combinations (by manual MAPE)**")
    worst = (f.dropna(subset=["manual_mape"])
             .groupby(["sku_code", "product_name", "category_name", "warehouse_city"])
             .agg(manual_mape=("manual_mape", "mean"),
                  ml_mape=("ml_mape", "mean"),
                  avg_bias=("manual_bias_units", "mean"))
             .reset_index()
             .sort_values("manual_mape", ascending=False)
             .head(15))
    worst["improvement_pp"] = (worst["manual_mape"] - worst["ml_mape"]).round(1)
    worst = worst.round({"manual_mape": 1, "ml_mape": 1, "avg_bias": 1})
    worst.columns = ["SKU", "Product", "Category", "Warehouse",
                     "Manual MAPE", "ML MAPE", "Avg bias (u/wk)", "Δ (pp)"]

    st.dataframe(
        worst,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Manual MAPE":     st.column_config.NumberColumn(format="%.1f%%"),
            "ML MAPE":         st.column_config.NumberColumn(format="%.1f%%"),
            "Avg bias (u/wk)": st.column_config.NumberColumn(format="%+.0f"),
            "Δ (pp)":          st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown("---")
    st.markdown("**Stockout Cost by Category (filtered period)**")
    so_by_cat = (stockouts
                 .merge(products, on="product_id")
                 .merge(categories, on="category_id")
                 .merge(warehouses.rename(columns={"city": "warehouse_city"}), on="warehouse_id")
                 .query("event_date >= @start_date and event_date <= @end_date"))
    if selected_cat != "All":
        so_by_cat = so_by_cat[so_by_cat["category_name"] == selected_cat]
    if selected_wh != "All":
        so_by_cat = so_by_cat[so_by_cat["warehouse_city"] == selected_wh]

    by_cat_so = (so_by_cat.groupby("category_name")
                 .agg(events=("event_id", "count"),
                      units=("shortage_units", "sum"),
                      revenue=("lost_revenue", "sum"))
                 .reset_index()
                 .sort_values("revenue", ascending=False))
    fig = px.bar(
        by_cat_so, x="category_name", y="revenue",
        labels={"revenue": "Lost revenue (₹)", "category_name": ""},
        color="revenue", color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 4 — FUTURE FORECASTS
# ===========================================================================
with tab4:
    st.subheader("Next 12 Weeks — ML Forecast with Prediction Bands")
    st.caption("Recursively generated by the final HistGradientBoosting model "
               "trained on all available history.")

    fut = future.merge(products[["product_id", "sku_code", "product_name"]], on="product_id")
    fut = fut.merge(warehouses.rename(columns={"city": "warehouse_city"}), on="warehouse_id")

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        cat_sel = st.selectbox("Category", ["All"] + sorted(categories["category_name"].tolist()),
                               key="future_cat")
    with col_f2:
        wh_sel = st.selectbox("Warehouse", ["All"] + sorted(warehouses["city"].tolist()),
                              key="future_wh")

    fut_view = fut.copy()
    if cat_sel != "All":
        sku_in_cat = products[products["category_id"] == int(
            categories.loc[categories["category_name"] == cat_sel, "category_id"].iloc[0]
        )]["product_id"]
        fut_view = fut_view[fut_view["product_id"].isin(sku_in_cat)]
    if wh_sel != "All":
        fut_view = fut_view[fut_view["warehouse_city"] == wh_sel]

    # Aggregated total forecast with uncertainty
    agg = (fut_view.groupby("forecast_week_date")
           .agg(pred=("forecasted_units", "sum"),
                lo=("lower_bound", "sum"),
                hi=("upper_bound", "sum"))
           .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["forecast_week_date"], y=agg["hi"], mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=agg["forecast_week_date"], y=agg["lo"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(16,185,129,0.18)",
        name="85% prediction band",
    ))
    fig.add_trace(go.Scatter(
        x=agg["forecast_week_date"], y=agg["pred"], mode="lines+markers",
        line=dict(color="#10b981", width=3), name="ML forecast",
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Units/week (total)", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-SKU table
    st.markdown("**Forecast by SKU × Warehouse (next 12 weeks total)**")
    per_sku = (fut_view.groupby(["sku_code", "product_name", "warehouse_city"])
               .agg(forecast_units=("forecasted_units", "sum"),
                    low=("lower_bound", "sum"),
                    high=("upper_bound", "sum"))
               .reset_index()
               .sort_values("forecast_units", ascending=False)
               .head(50))
    per_sku.columns = ["SKU", "Product", "Warehouse",
                       "12-wk forecast", "Low (85%)", "High (85%)"]
    st.dataframe(
        per_sku, use_container_width=True, hide_index=True,
        column_config={
            "12-wk forecast": st.column_config.NumberColumn(format="%d"),
            "Low (85%)":      st.column_config.NumberColumn(format="%d"),
            "High (85%)":     st.column_config.NumberColumn(format="%d"),
        },
    )

    st.caption("Use the ranges above as recommended replenishment volumes; order "
               "toward the upper bound for high-service-level SKUs and toward the "
               "point estimate for cost-sensitive ones.")
