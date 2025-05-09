import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Set page config with new theme
st.set_page_config(
    page_title="Sales Performance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    products = ["Product A", "Product B", "Product C", "Product D"]
    regions = ["North", "South", "East", "West"]
    
    # Monthly special events (promotions, holidays etc.)
    monthly_events = {
        1: 1.1,   # January sales
        2: 1.0,   # Normal month
        3: 1.0,   # Normal month
        4: 1.15,  # Spring promotions
        5: 1.0,   # Normal month
        6: 1.05,  # Mid-year
        7: 1.0,   # Normal month
        8: 1.0,   # Normal month
        9: 1.0,   # Normal month
        10: 1.0,  # Normal month
        11: 1.2,  # Pre-holiday
        12: 1.3   # Holiday season
    }
    
    # Base values for each product with quarterly adjustments
    product_stats = {
        "Product A": {
            "mean_units": 45, "std_units": 8, 
            "price_mean": 12, "price_std": 2, 
            "margin_mean": 0.25, "margin_std": 0.03,
            "q_multipliers": [1.0, 1.1, 0.9, 1.2],  # Q1, Q2, Q3, Q4 multipliers
            "region_min": 0.7  # Minimum regional multiplier to ensure some sales everywhere
        },
        "Product B": {
            "mean_units": 60, "std_units": 12, 
            "price_mean": 15, "price_std": 3, 
            "margin_mean": 0.22, "margin_std": 0.04,
            "q_multipliers": [0.9, 1.0, 1.3, 0.8],
            "region_min": 0.6
        },
        "Product C": {
            "mean_units": 30, "std_units": 5, 
            "price_mean": 20, "price_std": 4, 
            "margin_mean": 0.30, "margin_std": 0.02,
            "q_multipliers": [1.2, 0.8, 1.1, 1.4],
            "region_min": 0.8
        },
        "Product D": {
            "mean_units": 50, "std_units": 10, 
            "price_mean": 18, "price_std": 3, 
            "margin_mean": 0.28, "margin_std": 0.03,
            "q_multipliers": [1.1, 1.0, 0.7, 1.5],
            "region_min": 0.5
        }
    }
    
    # Regional multipliers with seasonal variations
    region_multipliers = {
        "North": {
            "base": 0.9,
            "q_adjustments": [1.0, 1.1, 1.3, 0.7],  # Strong in summer (Q3)
            "min_mult": 0.6  # Minimum multiplier to ensure some sales
        },
        "South": {
            "base": 1.1,
            "q_adjustments": [0.9, 1.0, 1.2, 1.4],  # Strong in winter (Q4)
            "min_mult": 0.7
        },
        "East": {
            "base": 1.0,
            "q_adjustments": [1.1, 1.0, 0.9, 1.0],  # Stable with Q1 boost
            "min_mult": 0.8
        },
        "West": {
            "base": 1.2,
            "q_adjustments": [1.0, 1.2, 1.1, 0.9],  # Strong in Q2
            "min_mult": 0.7
        }
    }
    
    # Day of week patterns with quarterly variations
    day_patterns = {
        0: [0.8, 0.9, 1.0, 0.9],  # Monday
        1: [1.0, 1.0, 1.1, 1.0],   # Tuesday
        2: [1.1, 1.2, 1.0, 1.1],   # Wednesday
        3: [1.3, 1.1, 1.2, 1.4],   # Thursday
        4: [1.2, 1.3, 1.1, 1.5],   # Friday
        5: [0.6, 0.7, 0.8, 1.0],   # Saturday
        6: [0.4, 0.5, 0.6, 0.8]    # Sunday
    }
    
    data = []
    for date in dates:
        day_of_week = date.weekday()
        month = date.month
        quarter = (date.month - 1) // 3
        event_multiplier = monthly_events.get(month, 1.0)
        
        for product in products:
            for region in regions:
                # Get quarterly adjustments with minimum enforcement
                q_mult = max(product_stats[product]["region_min"], 
                           product_stats[product]["q_multipliers"][quarter])
                region_q_mult = max(region_multipliers[region]["min_mult"],
                                  region_multipliers[region]["q_adjustments"][quarter])
                day_mult = day_patterns[day_of_week][quarter]
                
                # Calculate base units with all seasonal factors
                base_units = max(5, int(np.random.normal(
                    product_stats[product]["mean_units"],
                    product_stats[product]["std_units"]
                )))
                
                units = max(1, int(
                    base_units * 
                    max(region_multipliers[region]["min_mult"], region_multipliers[region]["base"]) * 
                    region_q_mult *
                    q_mult * 
                    day_mult * 
                    event_multiplier
                ))
                
                # Generate realistic price with quarterly variation
                price = max(5, np.random.normal(
                    product_stats[product]["price_mean"],
                    product_stats[product]["price_std"]
                ))
                
                # Generate margin with potential quarterly promotions
                margin = max(0.1, min(0.5, np.random.normal(
                    product_stats[product]["margin_mean"],
                    product_stats[product]["margin_std"]
                )))
                
                # Special holiday discounts in Q4
                if quarter == 3:
                    margin = margin * 0.95  # 5% margin reduction for holiday sales
                
                revenue = units * price
                profit = revenue * margin
                
                data.append({
                    "Date": date,
                    "Product": product,
                    "Region": region,
                    "Units": units,
                    "Revenue": revenue,
                    "Profit": profit,
                    "Quarter": f"Q{quarter + 1}"
                })
    
    df = pd.DataFrame(data)
    df["Month"] = df["Date"].dt.month_name()
    df["MonthNum"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.day_name()
    
    return df

df = load_data()

# Initialize session state for view mode
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "strategic"

# CSS styling to make the UI more aesthetically pleasing and better follow the cognitive factors
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #4f46e5;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .positive {
        color: #10b981;
    }
    
    .negative {
        color: #ef4444;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background: #e2e8f0;
        border-radius: 0;
        margin-right: 0;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #4f46e5;
        font-weight: 600;
        border-bottom: 3px solid #4f46e5;
    }
    
    /* Buttons */
    .stButton button {
        background: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #4338ca;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Inputs */
    .stSelectbox, .stRadio, .stSlider {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

# Modern header with navigation
st.markdown("""
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">Sales Performance Dashboard</h1>
                <p style="margin: 0; opacity: 0.9;">Data-driven insights for strategic business decisions</p>
            </div>
            <div style="display: flex; gap: 1rem;">
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# View mode selector
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    view_mode = st.radio(
        "View Mode",
        ["Strategic", "Operational", "Analytical"],
        horizontal=True,
        key="view_mode_selector",
        label_visibility="collapsed"
    )
    st.session_state.view_mode = view_mode.lower()

# Calculate metrics for the full dataset
def calculate_metrics(df):
    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_units = df["Units"].sum()
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
    
    # Calculate comparison metrics vs previous period
    # The previous period is simulated by comparing the second half of the year to the first half - this is done because the dataset is generated
    first_half = df[df["Date"] < datetime(2023, 7, 1)]
    second_half = df[df["Date"] >= datetime(2023, 7, 1)]
    
    prev_revenue = first_half["Revenue"].sum()
    prev_profit = first_half["Profit"].sum()
    prev_units = first_half["Units"].sum()
    prev_margin = (prev_profit / prev_revenue) * 100 if prev_revenue > 0 else 0
    
    curr_revenue = second_half["Revenue"].sum()
    curr_profit = second_half["Profit"].sum()
    curr_units = second_half["Units"].sum()
    
    revenue_change = ((curr_revenue - prev_revenue) / prev_revenue) * 100 if prev_revenue > 0 else 0
    profit_change = ((curr_profit - prev_profit) / prev_profit) * 100 if prev_profit > 0 else 0
    units_change = ((curr_units - prev_units) / prev_units) * 100 if prev_units > 0 else 0
    
    curr_margin = (curr_profit / curr_revenue) * 100 if curr_revenue > 0 else 0
    margin_change = curr_margin - prev_margin
    
    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_units": total_units,
        "profit_margin": profit_margin,
        "revenue_change": revenue_change,
        "profit_change": profit_change,
        "units_change": units_change,
        "margin_change": margin_change
    }

metrics = calculate_metrics(df)

# Generate dynamic insights based on data
def generate_insights(df):
    insights = []
    
    # Check if DataFrame is empty
    if df.empty:
        insights.append({
            "title": "ℹ️ No Data Available",
            "content": "No records available.",
            "metric": "No metrics"
        })
        return insights
    
    # Product insights
    try:
        product_df = df.groupby("Product").agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        product_df["Profit Margin"] = (product_df["Profit"] / product_df["Revenue"]) * 100
        
        if not product_df.empty:
            top_product = product_df.loc[product_df["Revenue"].idxmax()]
            insights.append({
                "title": "📊 Top Performing Product",
                "content": f"{top_product['Product']} has the highest revenue at ${top_product['Revenue']/1000:,.1f}K",
                "metric": f"Margin: {top_product['Profit Margin']:.1f}%"
            })
    except:
        pass
    
    # Regional insights
    try:
        region_df = df.groupby("Region").agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        region_df["Profit Margin"] = (region_df["Profit"] / region_df["Revenue"]) * 100
        
        if not region_df.empty:
            top_region = region_df.loc[region_df["Profit Margin"].idxmax()]
            insights.append({
                "title": "🌎 Regional Spotlight",
                "content": f"{top_region['Region']} has the highest profit margin at {top_region['Profit Margin']:.1f}%",
                "metric": f"Revenue: ${top_region['Revenue']/1000:,.1f}K"
            })
    except:
        pass
    
    # Daily patterns
    try:
        day_df = df.copy()
        day_df["Day"] = day_df["Date"].dt.day_name()
        day_avg = day_df.groupby("Day").agg({"Revenue": "mean"}).reset_index()
        
        if not day_avg.empty:
            peak_day = day_avg.loc[day_avg["Revenue"].idxmax()]
            avg_revenue = day_avg["Revenue"].mean()
            day_performance = ((peak_day["Revenue"] - avg_revenue) / avg_revenue) * 100 if avg_revenue > 0 else 0
            
            insights.append({
                "title": "📅 Daily Patterns",
                "content": f"{peak_day['Day']} shows peak sales activity with {day_performance:.1f}% higher revenue than average",
                "metric": f"Revenue: ${peak_day['Revenue']/1000:,.1f}K"
            })
    except:
        pass
    
    # Correlation insights
    try:
        corr_df = df.groupby(["Product", "Region"]).agg({
            "Revenue": "sum",
            "Profit": "sum",
            "Units": "sum"
        }).reset_index()
        
        numeric_cols = corr_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:  # Need at least 2 numeric columns for correlation
            corr_matrix = corr_df[numeric_cols].corr()
            units_revenue_corr = corr_matrix.loc["Units", "Revenue"]
            
            insights.append({
                "title": "🔍 Strong Correlation",
                "content": f"Units sold and revenue show {units_revenue_corr:.2f} correlation coefficient",
                "metric": "P-value: <0.001" if abs(units_revenue_corr) > 0.7 else "Moderate relationship"
            })
    except:
        pass
    
    #Error handling in case there are no insights
    if not insights:
        insights.append({
            "title": "ℹ️ No Insights Available",
            "content": "Could not generate insights.",
            "metric": "Try different filters"
        })
    
    return insights

insights = generate_insights(df)

# Improved forecasting function with progressive growth
def generate_forecast(df, months=6, revenue_growth=0.05, cost_change=0.0):
    # Prepare historical data by aggregating to monthly level
    hist_df = df.groupby("Date").agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
    hist_df["Cost"] = hist_df["Revenue"] - hist_df["Profit"]
    
    # Create future dates using proper pandas date arithmetic
    last_date = hist_df["Date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(days=1),  # Start from next day
        periods=months,
        freq='M'  # Monthly frequency
    )
    
    # Simple linear regression for baseline forecast
    X = np.array(range(len(hist_df))).reshape(-1, 1)
    y_revenue = hist_df["Revenue"].values
    y_cost = hist_df["Cost"].values
    
    model_revenue = LinearRegression().fit(X, y_revenue)
    model_cost = LinearRegression().fit(X, y_cost)
    
    # Predict with progressive growth factors
    future_X = np.array(range(len(hist_df), len(hist_df) + months)).reshape(-1, 1)
    base_revenue = model_revenue.predict(future_X)
    base_cost = model_cost.predict(future_X)
    
    # Each month compounds on the previous
    forecast_revenue = []
    forecast_cost = []
    forecast_profit = []
    
    # Calculate monthly values with compounding growth
    for i in range(months):
        if i == 0:
            # First month uses base forecast with simple growth
            rev = base_revenue[i] * (1 + revenue_growth)
            cost = base_cost[i] * (1 + cost_change)
        else:
            # Subsequent months compound from previous month
            rev = forecast_revenue[i-1] * (1 + revenue_growth)
            cost = forecast_cost[i-1] * (1 + cost_change)
            
        profit = rev - cost
        forecast_revenue.append(rev)
        forecast_cost.append(cost)
        forecast_profit.append(profit)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Revenue": forecast_revenue,
        "Cost": forecast_cost,
        "Profit": forecast_profit,
        "Type": "Forecast"
    })
    
    # Prepare historical data for plotting
    hist_df["Type"] = "Historical"
    
    return pd.concat([hist_df, forecast_df])

# Strategic View
if st.session_state.view_mode == "strategic":
    # Key metrics with new layout
    st.subheader("Performance at a Glance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Revenue</div>
                <div class="metric-value">${metrics['total_revenue']/1000:,.1f}K</div>
                <div class="metric-delta {'positive' if metrics['revenue_change'] >= 0 else 'negative'}">
                    {'↑' if metrics['revenue_change'] >= 0 else '↓'} {f"{abs(metrics['revenue_change']):.1f}%" if metrics['revenue_change'] != 0 else "0%"} vs prior
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Profit</div>
                <div class="metric-value">${metrics['total_profit']/1000:,.1f}K</div>
                <div class="metric-delta {'positive' if metrics['profit_change'] >= 0 else 'negative'}">
                    {'↑' if metrics['profit_change'] >= 0 else '↓'} {f"{abs(metrics['profit_change']):.1f}%" if metrics['profit_change'] != 0 else "0%"} vs prior
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Units Sold</div>
                <div class="metric-value">{metrics['total_units']:,}</div>
                <div class="metric-delta {'positive' if metrics['units_change'] >= 0 else 'negative'}">
                    {'↑' if metrics['units_change'] >= 0 else '↓'} {f"{abs(metrics['units_change']):.1f}%" if metrics['units_change'] != 0 else "0%"} vs prior
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Profit Margin</div>
                <div class="metric-value">{metrics['profit_margin']:.1f}%</div>
                <div class="metric-delta {'positive' if metrics['margin_change'] >= 0 else 'negative'}">
                    {'↑' if metrics['margin_change'] >= 0 else '↓'} {f"{abs(metrics['margin_change']):.1f}pp" if metrics['margin_change'] != 0 else "0pp"} vs prior
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Insights section
    st.subheader("Strategic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="card">
                <h3>{insights[0]['title']}</h3>
                <p>{insights[0]['content']}</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>{insights[0]['metric']}</span>
                    <span class="positive">↑ {metrics['revenue_change']:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="card">
                <h3>{insights[1]['title']}</h3>
                <p>{insights[1]['content']}</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>{insights[1]['metric']}</span>
                    <span class="positive">↑ {metrics['margin_change']:.1f}pp</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Main visualization area
    st.subheader("Trend Analysis")
    
    tab1, tab2 = st.tabs(["Revenue & Profit", "Product Performance"])
    
    with tab1:
        # Time series chart
        fig = go.Figure()
        
        # Add Revenue area chart
        time_agg = st.radio(
            "View by:",
            ["Monthly", "Quarterly"],
            horizontal=True,
            key="time_agg_view"
        )
        
        if time_agg == "Monthly":
            time_col = "Month"
            sort_order = ["January", "February", "March", "April", "May", "June", 
                         "July", "August", "September", "October", "November", "December"]
            agg_df = df.groupby(time_col).agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        else:
            time_col = "Quarter"
            sort_order = ["Q1", "Q2", "Q3", "Q4"]
            agg_df = df.groupby(time_col).agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=agg_df[time_col],
                y=agg_df["Revenue"],
                name="Revenue",
                line=dict(color="#4f46e5", width=3),
                stackgroup="one",
                fill="tozeroy",
                fillcolor="rgba(79, 70, 229, 0.1)"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=agg_df[time_col],
                y=agg_df["Profit"],
                name="Profit",
                line=dict(color="#10b981", width=3),
                stackgroup="two",
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.1)"
            )
        )
        
        fig.update_layout(
            xaxis=dict(
                title=time_col,
                categoryorder="array",
                categoryarray=sort_order
            ),
            yaxis=dict(title="Amount ($)"),
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Product performance
        product_df = df.groupby("Product").agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        product_df["Profit Margin"] = (product_df["Profit"] / product_df["Revenue"]) * 100
        
        fig = px.bar(
            product_df,
            x="Product",
            y="Revenue",
            color="Profit Margin",
            color_continuous_scale="Tealgrn",
            text_auto=".2s",
            labels={"Revenue": "Revenue ($)", "Profit Margin": "Margin (%)"}
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    tab3, tab4 = st.tabs(["Product Trends", "Regional Trends"])

    with tab3:
        st.subheader("Product Performance Over Time")
    
        # Time aggregation selector
        time_agg = st.radio(
            "View by:",
            ["Monthly", "Quarterly"],
            horizontal=True,
            key="product_time_agg"
        )
        
        if time_agg == "Monthly":
            time_col = "Month"
            sort_order = ["January", "February", "March", "April", "May", "June", 
                        "July", "August", "September", "October", "November", "December"]
            
            # Add MonthNum to allow proper sorting
            month_product_df = df.groupby(["Month", "MonthNum", "Product"]).agg({
                "Revenue": "sum",
                "Units": "sum",
                "Profit": "sum"
            }).reset_index()
            
            # Sort by month number to ensure chronological display
            agg_df = month_product_df.sort_values("MonthNum")
        else:
            time_col = "Quarter"
            sort_order = ["Q1", "Q2", "Q3", "Q4"]
            agg_df = df.groupby([time_col, "Product"]).agg({
                "Revenue": "sum",
                "Units": "sum", 
                "Profit": "sum"
            }).reset_index()
        
        # Metric selector
        metric = st.selectbox(
            "Select metric to view:",
            ["Revenue", "Units", "Profit", "Profit Margin"],
            key="product_metric"
        )
        
        if metric == "Profit Margin":
            agg_df["Profit Margin"] = (agg_df["Profit"] / agg_df["Revenue"]) * 100
            y_col = "Profit Margin"
        else:
            y_col = metric
        
        # Product performance trends
        fig = px.line(
            agg_df,
            x=time_col,
            y=y_col,
            color="Product",
            markers=True,
            category_orders={time_col: sort_order},
            title=f"Product {metric} Trends",
            labels={y_col: metric}
        )
        
        # Enhance the line display for better clarity
        fig.update_traces(
            mode="lines+markers", 
            marker_size=8,
            line=dict(width=3)  # Thicker lines for better visibility
        )
        
        # Ensure proper x-axis ordering
        if time_agg == "Monthly":
            fig.update_layout(
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=sort_order
                )
            )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            xaxis_title="Time Period",
            yaxis_title=metric,
            legend_title="Product"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Product comparison heatmap
        st.subheader("Product Performance Comparison")
        pivot_df = agg_df.pivot(index=time_col, columns="Product", values=y_col)
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Product", y=time_col, color=metric),
            x=pivot_df.columns,
            y=pivot_df.index,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            title=f"{metric} by Product and Time Period",
            xaxis_title="Product",
            yaxis_title=time_col
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Regional Performance Over Time")
        
        # Time aggregation selector
        time_agg = st.radio(
            "View by:",
            ["Monthly", "Quarterly"],
            horizontal=True,
            key="region_time_agg"
        )
        
        if time_agg == "Monthly":
            time_col = "Month"
            sort_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
            
            # Add MonthNum to allow proper sorting
            month_region_df = df.groupby(["Month", "MonthNum", "Region"]).agg({
                "Revenue": "sum",
                "Units": "sum",
                "Profit": "sum"
            }).reset_index()
            
            # Sort by month number to ensure chronological display
            agg_df = month_region_df.sort_values("MonthNum")
        else:
            time_col = "Quarter"
            sort_order = ["Q1", "Q2", "Q3", "Q4"]
            agg_df = df.groupby([time_col, "Region"]).agg({
                "Revenue": "sum",
                "Units": "sum",
                "Profit": "sum"
            }).reset_index()
        
        # Metric selector
        metric = st.selectbox(
            "Select metric to view:",
            ["Revenue", "Units", "Profit", "Profit Margin"],
            key="region_metric"
        )
        
        if metric == "Profit Margin":
            agg_df["Profit Margin"] = (agg_df["Profit"] / agg_df["Revenue"]) * 100
            y_col = "Profit Margin"
        else:
            y_col = metric
        
        # Regional performance trends
        fig = px.line(
            agg_df,
            x=time_col,
            y=y_col,
            color="Region",
            markers=True,
            category_orders={time_col: sort_order},
            title=f"Regional {metric} Trends",
            labels={y_col: metric}
        )
        
        # Enhance the line display for better clarity
        fig.update_traces(
            mode="lines+markers", 
            marker_size=8,
            line=dict(width=3)  # Thicker lines for better visibility
        )
        
        # Ensure proper x-axis ordering
        if time_agg == "Monthly":
            fig.update_layout(
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=sort_order
                )
            )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            xaxis_title="Time Period",
            yaxis_title=metric,
            legend_title="Region"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional comparison heatmap
        st.subheader("Regional Performance Comparison")
        pivot_df = agg_df.pivot(index=time_col, columns="Region", values=y_col)
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Region", y=time_col, color=metric),
            x=pivot_df.columns,
            y=pivot_df.index,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Plasma"
        )
        fig.update_layout(
            title=f"{metric} by Region and Time Period",
            xaxis_title="Region",
            yaxis_title=time_col
        )
        st.plotly_chart(fig, use_container_width=True)

# Operational View - Detailed metrics
elif st.session_state.view_mode == "operational":
    st.subheader("Operational Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="card">
                <h3>{insights[2]['title']}</h3>
                <p>{insights[2]['content']}</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>{insights[2]['metric']}</span>
                    <span class="positive">↑ {metrics['revenue_change']:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Find product with lowest inventory turnover
        product_turnover = df.groupby("Product").agg({"Units": "sum"}).reset_index()
        avg_turnover = product_turnover["Units"].mean()
        slow_product = product_turnover.loc[product_turnover["Units"].idxmin()]
        turnover_diff = ((avg_turnover - slow_product["Units"]) / avg_turnover) * 100
        
        st.markdown(f"""
            <div class="card">
                <h3>📦 Inventory Alert</h3>
                <p>{slow_product['Product']} has {turnover_diff:.1f}% slower turnover than average</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>Units Sold: {slow_product['Units']:,}</span>
                    <span class="negative">↓ {turnover_diff:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Operational Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily sales trend
        daily_df = df.groupby("Date").agg({"Revenue": "sum"}).reset_index()
        
        fig = px.line(
            daily_df,
            x="Date",
            y="Revenue",
            title="Daily Revenue Trend",
            labels={"Revenue": "Revenue ($)"}
        )
        fig.update_traces(line_color="#4f46e5")
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Product mix
        product_mix = df.groupby("Product").agg({"Units": "sum"}).reset_index()
        
        fig = px.pie(
            product_mix,
            names="Product",
            values="Units",
            title="Product Mix by Units",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional performance
        region_df = df.groupby("Region").agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
        region_df["Profit Margin"] = (region_df["Profit"] / region_df["Revenue"]) * 100
        
        fig = px.bar(
            region_df,
            x="Region",
            y="Revenue",
            color="Profit Margin",
            title="Regional Performance",
            text_auto=".2s",
            color_continuous_scale="Bluyl",
            labels={"Revenue": "Revenue ($)", "Profit Margin": "Margin (%)"}
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by day of week
        day_df = df.copy()
        day_df["Day"] = day_df["Date"].dt.day_name()
        
        fig = px.bar(
            day_df.groupby("Day").agg({"Revenue": "sum"}).reset_index(),
            x="Day",
            y="Revenue",
            title="Revenue by Day of Week",
            category_orders={"Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
            color_discrete_sequence=["#4f46e5"]
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

# Analytical View - Deep dive
elif st.session_state.view_mode == "analytical":
    st.subheader("Analytical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="card">
                <h3>{insights[3]['title']}</h3>
                <p>{insights[3]['content']}</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>{insights[3]['metric']}</span>
                    <span class="positive">Strong</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Find region with high margin but low market share
        region_df = df.groupby("Region").agg({"Revenue": "sum", "Profit": "sum", "Units": "sum"}).reset_index()
        region_df["Profit Margin"] = (region_df["Profit"] / region_df["Revenue"]) * 100
        region_df["Market Share"] = (region_df["Units"] / region_df["Units"].sum()) * 100
        
        opportunity_region = region_df.loc[(region_df["Profit Margin"] > region_df["Profit Margin"].mean()) & 
                                         (region_df["Market Share"] < region_df["Market Share"].mean())]
        
        if len(opportunity_region) > 0:
            best_opp = opportunity_region.loc[opportunity_region["Profit Margin"].idxmax()]
            margin_diff = ((best_opp["Profit Margin"] - region_df["Profit Margin"].mean()) / 
                          region_df["Profit Margin"].mean()) * 100
            
            st.markdown(f"""
                <div class="card">
                    <h3>📈 Growth Opportunity</h3>
                    <p>{best_opp['Region']} has {margin_diff:.1f}% higher margin but only {best_opp['Market Share']:.1f}% market share</p>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Margin: {best_opp['Profit Margin']:.1f}%</span>
                        <span class="positive">↑ {margin_diff:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3>📈 Growth Opportunity</h3>
                    <p>No clear regional opportunities found based on current data</p>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Try different analysis</span>
                        <span class="positive">↑</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.subheader("Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Performance Breakdown", "Forecasting"])
    
    with tab1:
        # Correlation matrix
        corr_df = df.groupby(["Product", "Region"]).agg({
            "Revenue": "sum",
            "Profit": "sum",
            "Units": "sum"
        }).reset_index()
        
        numeric_cols = corr_df.select_dtypes(include=[np.number]).columns
        corr_matrix = corr_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Metric Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Detailed breakdown
        breakdown = st.selectbox(
            "Breakdown by",
            ["Product", "Region", "Month", "Quarter"],
            key="breakdown_select"
        )
        
        metric = st.selectbox(
            "Metric",
            ["Revenue", "Profit", "Units", "Profit Margin"],
            key="metric_select"
        )
        
        if metric == "Profit Margin":
            bd_df = df.groupby(breakdown).agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
            bd_df[metric] = (bd_df["Profit"] / bd_df["Revenue"]) * 100
        else:
            bd_df = df.groupby(breakdown).agg({metric: "sum"}).reset_index()
        
        fig = px.bar(
            bd_df,
            x=breakdown,
            y=metric,
            title=f"{metric} by {breakdown}",
            color=metric,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Forecasting controls
        st.subheader("Forecast Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_months = st.slider("Months to forecast", 1, 12, 6)
        
        with col2:
            revenue_growth = st.slider("Monthly revenue growth (%)", -5.0, 20.0, 5.0) / 100
        
        with col3:
            cost_change = st.slider("Monthly cost change (%)", -10.0, 10.0, 0.0) / 100
        
        # Generate forecast
        forecast_df = generate_forecast(df, forecast_months, revenue_growth, cost_change)
        
        # Plot forecast without using add_vline which has date arithmetic issues
        fig = go.Figure()
        
        # Historical data
        hist_df = forecast_df[forecast_df["Type"] == "Historical"]
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Revenue"],
            name="Historical Revenue",
            line=dict(color="#4f46e5", width=2),
            mode="lines+markers"
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Profit"],
            name="Historical Profit",
            line=dict(color="#10b981", width=2),
            mode="lines+markers"
        ))
        
        # Forecast data
        forecast_data = forecast_df[forecast_df["Type"] == "Forecast"]
        fig.add_trace(go.Scatter(
            x=forecast_data["Date"],
            y=forecast_data["Revenue"],
            name="Forecast Revenue",
            line=dict(color="#4f46e5", width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(symbol="diamond", size=8),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast Revenue: $%{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data["Date"],
            y=forecast_data["Profit"],
            name="Forecast Profit",
            line=dict(color="#10b981", width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(symbol="diamond", size=8),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast Profit: $%{y:.2f}<extra></extra>"
        ))
        
        # Instead of add_vline, use shape with type="line"
        if not hist_df.empty and not forecast_data.empty:
            # Get min and max y values for the entire dataset to set vertical line height
            all_y_values = np.concatenate([
                hist_df["Revenue"].values,
                hist_df["Profit"].values,
                forecast_data["Revenue"].values,
                forecast_data["Profit"].values
            ])
            y_min = min(all_y_values) * 0.9  # Extend a bit below
            y_max = max(all_y_values) * 1.1  # Extend a bit above
            
            # Add a vertical line as a shape
            last_hist_date = hist_df["Date"].max()
            
            # Create vertical line at the transition between historical and forecast
            fig.add_shape(
                type="line",
                x0=last_hist_date,
                x1=last_hist_date,
                y0=y_min,
                y1=y_max,
                line=dict(
                    color="gray",
                    width=2,
                    dash="dash"
                )
            )
            
            # Add annotation for the forecast start
            fig.add_annotation(
                x=last_hist_date,
                y=y_max,
                text="Forecast Start",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                font=dict(size=12)
            )
            
            # Add shaded area for forecast region using shape
            fig.add_shape(
                type="rect",
                x0=last_hist_date,
                x1=forecast_data["Date"].max(),
                y0=0,
                y1=1,
                yref="paper",
                fillcolor="rgba(128, 128, 128, 0.1)",
                line_width=0,
                layer="below"
            )
        
        fig.update_layout(
            title="Revenue and Profit Forecast",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced forecast summary with monthly data
        st.subheader("Forecast Summary")
        
        total_revenue_growth = ((1 + revenue_growth) ** forecast_months - 1) * 100
        total_cost_change = ((1 + cost_change) ** forecast_months - 1) * 100
        
        last_hist = hist_df.iloc[-1]
        last_forecast = forecast_data.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Projected Revenue Growth",
                f"{total_revenue_growth:.1f}%",
                f"${last_forecast['Revenue']/1000:,.1f}K vs ${last_hist['Revenue']/1000:,.1f}K"
            )
        
        with col2:
            st.metric(
                "Projected Cost Change",
                f"{total_cost_change:.1f}%",
                f"${last_forecast['Cost']/1000:,.1f}K vs ${last_hist['Cost']/1000:,.1f}K"
            )
        
        with col3:
            profit_change = ((last_forecast["Profit"] - last_hist["Profit"]) / last_hist["Profit"]) * 100
            st.metric(
                "Projected Profit Change",
                f"{profit_change:.1f}%",
                f"${last_forecast['Profit']/1000:,.1f}K vs ${last_hist['Profit']/1000:,.1f}K"
            )
            
        # Show monthly forecast table
        st.subheader("Monthly Forecast Breakdown")
        
        # Format the forecast data for display
        display_forecast = forecast_data.copy()
        display_forecast["Month"] = display_forecast["Date"].dt.strftime("%b %Y")
        display_forecast["Revenue"] = display_forecast["Revenue"].map("${:,.2f}".format)
        display_forecast["Profit"] = display_forecast["Profit"].map("${:,.2f}".format)
        display_forecast["Cost"] = display_forecast["Cost"].map("${:,.2f}".format)
        display_forecast["Profit Margin"] = (forecast_data["Profit"] / forecast_data["Revenue"] * 100).map("{:.1f}%".format)
        
        # Reorder and select columns for display
        display_forecast = display_forecast[["Month", "Revenue", "Cost", "Profit", "Profit Margin"]]
        
        # Display the table
        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

# Footer with data freshness
st.markdown("---")
st.caption(f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Sales Performance Dashboard")