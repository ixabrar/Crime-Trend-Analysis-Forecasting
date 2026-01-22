"""
Crime Pattern Analysis Dashboard

A Streamlit-based visualization dashboard for exploring crime trends 
and forecasts in India (2001-2028).

Key Features:
- Historical crime trends (2001-2014)
- ARIMA-based forecasts (2015-2028)
- State-level clustering analysis
- All models pre-trained and loaded from disk

Ethical Compliance:
- Uses only aggregated NCRB/data.gov.in data
- No personal data or individual profiling
- Academic research purpose only
- Models trained offline
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# ========================
# Configuration
# ========================

st.set_page_config(
    page_title="Crime Pattern Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================
# Data Loading Functions
# ========================

@st.cache_data
def load_processed_data():
    """Load preprocessed crime datasets."""
    data_path = Path("data/processed")
    
    core_ipc = pd.read_csv(data_path / "core_crime" / "core_ipc_standardized.csv")
    women_crime = pd.read_csv(data_path / "women_crime" / "women_crime_standardized.csv")
    children_crime = pd.read_csv(data_path / "children_crime" / "children_crime_standardized.csv")
    
    return core_ipc, women_crime, children_crime


@st.cache_resource
def load_arima_models():
    """Load pre-trained ARIMA models from disk and prepare data."""
    model_path = Path("models")
    
    # Load processed data
    core_ipc, women_crime, children_crime = load_processed_data()
    
    # Prepare time series data
    # IPC crimes
    yearly_ipc = aggregate_yearly(core_ipc, "total_ipc_crimes")
    yearly_ipc["year"] = pd.to_datetime(yearly_ipc["year"], format="%Y")
    yearly_ipc = yearly_ipc.set_index("year").asfreq("YS")
    ts_ipc = yearly_ipc["total_ipc_crimes"]
    
    # Women crimes
    yearly_women = women_crime.groupby("year", as_index=False).sum(numeric_only=True)
    yearly_women["total_women_crimes"] = yearly_women.drop(columns=["year"], errors="ignore").sum(axis=1)
    yearly_women = yearly_women[["year", "total_women_crimes"]]
    yearly_women["year"] = pd.to_datetime(yearly_women["year"], format="%Y")
    yearly_women = yearly_women.set_index("year").asfreq("YS")
    ts_women = yearly_women["total_women_crimes"]
    
    # Children crimes
    yearly_children = children_crime.groupby("year", as_index=False)["total"].sum()
    yearly_children.rename(columns={"total": "total_children_crimes"}, inplace=True)
    yearly_children["year"] = pd.to_datetime(yearly_children["year"], format="%Y")
    yearly_children = yearly_children.set_index("year").asfreq("YS")
    ts_children = yearly_children["total_children_crimes"]
    
    # Load and fit models
    with open(model_path / "ipc_crime_arima_model.pkl", "rb") as f:
        ipc_model = pickle.load(f)
    ipc_result = ipc_model.fit()
    
    with open(model_path / "women_crime_arima_model.pkl", "rb") as f:
        women_model = pickle.load(f)
    women_result = women_model.fit()
    
    with open(model_path / "children_crime_arima_model.pkl", "rb") as f:
        children_model = pickle.load(f)
    children_result = children_model.fit()
    
    return ipc_result, women_result, children_result


@st.cache_data
def load_clusters():
    """Load pre-computed clustering results."""
    clusters_path = Path("data/processed/clusters.csv")
    
    if clusters_path.exists():
        return pd.read_csv(clusters_path)
    else:
        st.warning("⚠️ Clustering results not found. Run `python models/kmeans_clustering.py` first.")
        return None


@st.cache_resource
def load_kmeans_model():
    """Load pre-trained KMeans model."""
    model_path = Path("models/kmeans_model.pkl")
    
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


# ========================
# Helper Functions
# ========================

def aggregate_yearly(df, value_col):
    """Aggregate data by year."""
    return df.groupby("year", as_index=False)[value_col].sum()


def create_forecast_data(historical_years, historical_values, forecast_years, forecast_values, 
                          forecast_lower, forecast_upper):
    """Prepare forecast visualization data."""
    return {
        "historical_years": historical_years,
        "historical_values": historical_values,
        "forecast_years": forecast_years,
        "forecast_values": forecast_values,
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper
    }


def plot_forecast(data, title, ylabel):
    """Create interactive forecast plot with Plotly."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data["historical_years"],
        y=data["historical_values"],
        mode="lines+markers",
        name="Historical",
        line=dict(color="blue", width=2),
        marker=dict(size=6)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=data["forecast_years"],
        y=data["forecast_values"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(data["forecast_years"]) + list(data["forecast_years"][::-1]),
        y=list(data["forecast_upper"]) + list(data["forecast_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(255, 0, 0, 0.2)",
        line=dict(color="rgba(255, 0, 0, 0)"),
        name="95% Confidence Interval",
        showlegend=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    return fig


# ========================
# Tab 1: Overview (IPC Crimes)
# ========================

def tab_overview():
    st.header("📈 Total IPC Crimes - Overview")
    
    st.markdown("""
    This section displays historical trends and forecasts for total IPC (Indian Penal Code) crimes 
    across India from 2001 to 2028.
    """)
    
    # Load data
    core_ipc, _, _ = load_processed_data()
    ipc_result, _, _ = load_arima_models()
    
    # Aggregate by year
    yearly_ipc = aggregate_yearly(core_ipc, "total_ipc_crimes")
    
    # Generate forecast
    forecast_steps = 15  # 2015-2028 (if data ends at 2014)
    forecast = ipc_result.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    
    # Prepare data
    forecast_years = pd.date_range(
        start=str(yearly_ipc["year"].max() + 1), 
        periods=forecast_steps, 
        freq="YS"
    ).year
    
    data = create_forecast_data(
        historical_years=yearly_ipc["year"].values,
        historical_values=yearly_ipc["total_ipc_crimes"].values,
        forecast_years=forecast_years,
        forecast_values=forecast_df["mean"].values,
        forecast_lower=forecast_df["mean_ci_lower"].values,
        forecast_upper=forecast_df["mean_ci_upper"].values
    )
    
    # Plot
    fig = plot_forecast(
        data, 
        "Total IPC Crimes: Historical & Forecast (2001-2028)",
        "Total IPC Crimes"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Observed Year", int(yearly_ipc["year"].max()))
    
    with col2:
        st.metric("Last Observed Value", f"{int(yearly_ipc['total_ipc_crimes'].iloc[-1]):,}")
    
    with col3:
        st.metric("Forecast 2028", f"{int(forecast_df['mean'].iloc[-1]):,}")
    
    with col4:
        change = ((forecast_df["mean"].iloc[-1] - yearly_ipc["total_ipc_crimes"].iloc[-1]) 
                  / yearly_ipc["total_ipc_crimes"].iloc[-1] * 100)
        st.metric("% Change (2014→2028)", f"{change:.1f}%")
    
    # Insights
    with st.expander("📊 View Detailed Statistics"):
        st.subheader("Forecast Summary (2015-2028)")
        forecast_display = pd.DataFrame({
            "Year": forecast_years,
            "Predicted Value": forecast_df["mean"].values.astype(int),
            "Lower Bound (95%)": forecast_df["mean_ci_lower"].values.astype(int),
            "Upper Bound (95%)": forecast_df["mean_ci_upper"].values.astype(int)
        })
        st.dataframe(forecast_display, use_container_width=True)


# ========================
# Tab 2: Women Crime
# ========================

def tab_women_crime():
    st.header("👩 Crimes Against Women")
    
    st.markdown("""
    Analysis of crimes against women, including trends, forecasts, and proportion of total IPC crimes.
    """)
    
    # Load data
    core_ipc, women_crime, _ = load_processed_data()
    _, women_result, _ = load_arima_models()
    
    # Aggregate by year
    yearly_ipc = aggregate_yearly(core_ipc, "total_ipc_crimes")
    
    yearly_women = women_crime.groupby("year", as_index=False).sum(numeric_only=True)
    yearly_women["total_women_crimes"] = yearly_women.drop(
        columns=["year"], errors="ignore"
    ).sum(axis=1)
    yearly_women = yearly_women[["year", "total_women_crimes"]]
    
    # Generate forecast
    forecast_steps = 15
    forecast = women_result.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    
    forecast_years = pd.date_range(
        start=str(yearly_women["year"].max() + 1),
        periods=forecast_steps,
        freq="YS"
    ).year
    
    data = create_forecast_data(
        historical_years=yearly_women["year"].values,
        historical_values=yearly_women["total_women_crimes"].values,
        forecast_years=forecast_years,
        forecast_values=forecast_df["mean"].values,
        forecast_lower=forecast_df["mean_ci_lower"].values,
        forecast_upper=forecast_df["mean_ci_upper"].values
    )
    
    # Plot
    fig = plot_forecast(
        data,
        "Crimes Against Women: Historical & Forecast (2001-2028)",
        "Total Women Crimes"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Observed Year", int(yearly_women["year"].max()))
    
    with col2:
        st.metric("Last Observed Value", f"{int(yearly_women['total_women_crimes'].iloc[-1]):,}")
    
    with col3:
        st.metric("Forecast 2028", f"{int(forecast_df['mean'].iloc[-1]):,}")
    
    with col4:
        # Calculate % share of total IPC in last observed year
        merged = yearly_ipc.merge(yearly_women, on="year")
        share = (merged["total_women_crimes"].iloc[-1] / merged["total_ipc_crimes"].iloc[-1] * 100)
        st.metric("% of Total IPC (2014)", f"{share:.1f}%")
    
    # === SPECIFIC CRIME TYPES ANALYSIS ===
    st.markdown("---")
    st.subheader("🔍 Crime Type Breakdown")
    
    # Crime columns (excluding metadata)
    crime_cols = ['rape', 'kidnapping_abduction', 'dowry_deaths', 'assault_on_women', 
                  'insult_to_modesty_of_women', 'cruelty_by_husband_or_relatives', 
                  'importation_of_girls']
    
    # Aggregate by crime type over years
    crime_type_trends = women_crime.groupby('year')[crime_cols].sum()
    
    # Create sub-tabs for different visualizations
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Trends Over Time", "🥧 Distribution", "🏆 Top States"])
    
    with sub_tab1:
        st.markdown("**Yearly Trends for Each Crime Type**")
        
        # Line chart showing all crime types
        fig_trends = go.Figure()
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#16a085', '#2980b9', '#8e44ad', '#c0392b']
        
        for i, col in enumerate(crime_cols):
            fig_trends.add_trace(go.Scatter(
                x=crime_type_trends.index,
                y=crime_type_trends[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(width=2, color=colors[i]),
                marker=dict(size=6)
            ))
        
        fig_trends.update_layout(
            title="Women Crime Types: Yearly Trends (2001-2014)",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with sub_tab2:
        st.markdown("**Crime Type Distribution (2001-2014 Total)**")
        
        # Pie chart for total distribution
        crime_totals = women_crime[crime_cols].sum().sort_values(ascending=False)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=[label.replace('_', ' ').title() for label in crime_totals.index],
            values=crime_totals.values,
            hole=0.3,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_pie.update_layout(
            title="Distribution of Women Crime Types (2001-2014)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Show actual numbers
        st.markdown("**Total Cases by Crime Type:**")
        crime_stats = pd.DataFrame({
            'Crime Type': [label.replace('_', ' ').title() for label in crime_totals.index],
            'Total Cases': crime_totals.values,
            'Percentage': (crime_totals.values / crime_totals.sum() * 100).round(2)
        })
        st.dataframe(crime_stats, use_container_width=True, hide_index=True)
    
    with sub_tab3:
        st.markdown("**Top 10 States by Crime Type**")
        
        # Select crime type
        selected_crime = st.selectbox(
            "Select Crime Type:",
            options=crime_cols,
            format_func=lambda x: x.replace('_', ' ').title(),
            key='women_crime_select'
        )
        
        # Aggregate by state
        state_crime = women_crime.groupby('state_ut')[selected_crime].sum().sort_values(ascending=False).head(10)
        
        fig_bar = go.Figure(data=[go.Bar(
            x=state_crime.values,
            y=state_crime.index,
            orientation='h',
            marker=dict(color='#e74c3c', line=dict(color='#c0392b', width=1))
        )])
        
        fig_bar.update_layout(
            title=f"Top 10 States: {selected_crime.replace('_', ' ').title()} (2001-2014)",
            xaxis_title="Total Cases",
            yaxis_title="State/UT",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Trend Analysis
    with st.expander("📊 View Proportion Trend"):
        merged = yearly_ipc.merge(yearly_women, on="year")
        merged["women_share_pct"] = (merged["total_women_crimes"] / merged["total_ipc_crimes"] * 100)
        
        fig_share = go.Figure()
        fig_share.add_trace(go.Scatter(
            x=merged["year"],
            y=merged["women_share_pct"],
            mode="lines+markers",
            name="Women Crime Share",
            line=dict(color="purple", width=2),
            marker=dict(size=6)
        ))
        
        fig_share.update_layout(
            title="Women Crimes as % of Total IPC Crimes",
            xaxis_title="Year",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_share, use_container_width=True)


# ========================
# Tab 3: Children Crime
# ========================

def tab_children_crime():
    st.header("👶 Crimes Against Children")
    
    st.markdown("""
    Analysis of crimes against children with historical trends and future forecasts.
    """)
    
    # Load data
    _, _, children_crime = load_processed_data()
    _, _, children_result = load_arima_models()
    
    # Aggregate by year
    yearly_children = children_crime.groupby("year", as_index=False)["total"].sum()
    yearly_children.rename(columns={"total": "total_children_crimes"}, inplace=True)
    
    # Generate forecast
    forecast_steps = 15
    forecast = children_result.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    
    forecast_years = pd.date_range(
        start=str(yearly_children["year"].max() + 1),
        periods=forecast_steps,
        freq="YS"
    ).year
    
    data = create_forecast_data(
        historical_years=yearly_children["year"].values,
        historical_values=yearly_children["total_children_crimes"].values,
        forecast_years=forecast_years,
        forecast_values=forecast_df["mean"].values,
        forecast_lower=forecast_df["mean_ci_lower"].values,
        forecast_upper=forecast_df["mean_ci_upper"].values
    )
    
    # Plot
    fig = plot_forecast(
        data,
        "Crimes Against Children: Historical & Forecast (2001-2028)",
        "Total Children Crimes"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Observed Year", int(yearly_children["year"].max()))
    
    with col2:
        st.metric("Last Observed Value", f"{int(yearly_children['total_children_crimes'].iloc[-1]):,}")
    
    with col3:
        st.metric("Forecast 2028", f"{int(forecast_df['mean'].iloc[-1]):,}")
    
    with col4:
        change = ((forecast_df["mean"].iloc[-1] - yearly_children["total_children_crimes"].iloc[-1])
                  / yearly_children["total_children_crimes"].iloc[-1] * 100)
        st.metric("% Change (2013→2028)", f"{change:.1f}%")
    
    # === SPECIFIC CRIME TYPES ANALYSIS ===
    st.markdown("---")
    st.subheader("🔍 Crime Type Breakdown")
    
    # Crime columns (excluding metadata and total)
    crime_cols = ['murder', 'rape', 'kidnapping_abduction', 'foeticide', 'abetment_of_suicide',
                  'exposure_and_abandonment', 'procuration_of_minor_girls', 
                  'buying_of_girls_for_prostitution', 'selling_of_girls_for_prostitution',
                  'prohibition_of_child_marriage_act', 'other_crimes']
    
    # Aggregate by crime type over years
    crime_type_trends = children_crime.groupby('year')[crime_cols].sum()
    
    # Create sub-tabs for different visualizations
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Trends Over Time", "🥧 Distribution", "🏆 Top States"])
    
    with sub_tab1:
        st.markdown("**Yearly Trends for Each Crime Type**")
        
        # Line chart showing all crime types
        fig_trends = go.Figure()
        
        colors = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f', 
                  '#16a085', '#2980b9', '#8e44ad', '#9b59b6', '#34495e', '#95a5a6']
        
        for i, col in enumerate(crime_cols):
            fig_trends.add_trace(go.Scatter(
                x=crime_type_trends.index,
                y=crime_type_trends[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(width=2, color=colors[i]),
                marker=dict(size=6)
            ))
        
        fig_trends.update_layout(
            title="Children Crime Types: Yearly Trends (2001-2013)",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with sub_tab2:
        st.markdown("**Crime Type Distribution (2001-2013 Total)**")
        
        # Pie chart for total distribution
        crime_totals = children_crime[crime_cols].sum().sort_values(ascending=False)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=[label.replace('_', ' ').title() for label in crime_totals.index],
            values=crime_totals.values,
            hole=0.3,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_pie.update_layout(
            title="Distribution of Children Crime Types (2001-2013)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Show actual numbers
        st.markdown("**Total Cases by Crime Type:**")
        crime_stats = pd.DataFrame({
            'Crime Type': [label.replace('_', ' ').title() for label in crime_totals.index],
            'Total Cases': crime_totals.values,
            'Percentage': (crime_totals.values / crime_totals.sum() * 100).round(2)
        })
        st.dataframe(crime_stats, use_container_width=True, hide_index=True)
    
    with sub_tab3:
        st.markdown("**Top 10 States by Crime Type**")
        
        # Select crime type
        selected_crime = st.selectbox(
            "Select Crime Type:",
            options=crime_cols,
            format_func=lambda x: x.replace('_', ' ').title(),
            key='children_crime_select'
        )
        
        # Aggregate by state
        state_crime = children_crime.groupby('state_ut')[selected_crime].sum().sort_values(ascending=False).head(10)
        
        fig_bar = go.Figure(data=[go.Bar(
            x=state_crime.values,
            y=state_crime.index,
            orientation='h',
            marker=dict(color='#3498db', line=dict(color='#2980b9', width=1))
        )])
        
        fig_bar.update_layout(
            title=f"Top 10 States: {selected_crime.replace('_', ' ').title()} (2001-2013)",
            xaxis_title="Total Cases",
            yaxis_title="State/UT",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Insights
    with st.expander("📊 View Detailed Statistics"):
        st.subheader("Forecast Summary (2014-2028)")
        forecast_display = pd.DataFrame({
            "Year": forecast_years,
            "Predicted Value": forecast_df["mean"].values.astype(int),
            "Lower Bound (95%)": forecast_df["mean_ci_lower"].values.astype(int),
            "Upper Bound (95%)": forecast_df["mean_ci_upper"].values.astype(int)
        })
        st.dataframe(forecast_display, use_container_width=True)


# ========================
# Tab 4: Area Pattern (KMeans)
# ========================

def tab_area_pattern():
    st.header("🗺️ State-Level Crime Pattern Clustering")
    
    st.markdown("""
    This section uses **KMeans clustering** to group Indian states based on crime patterns (2001-2014).
    
    **Features used:**
    - Total IPC Crimes
    - Crimes Against Women
    - Crimes Against Children
    
    **Note:** Clustering is performed offline. Models are loaded from disk.
    """)
    
    # Load clustering data
    clusters_df = load_clusters()
    kmeans_info = load_kmeans_model()
    
    if clusters_df is None:
        st.error("❌ Clustering data not found. Please run: `python models/kmeans_clustering.py`")
        return
    
    # Sidebar: State selection
    st.sidebar.subheader("🔍 Explore by State")
    selected_state = st.sidebar.selectbox(
        "Select a State/UT:",
        options=sorted(clusters_df["state_ut"].unique())
    )
    
    # State Details
    state_info = clusters_df[clusters_df["state_ut"] == selected_state].iloc[0]
    
    st.subheader(f"📍 {selected_state}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cluster ID", int(state_info["cluster"]))
        st.metric("Cluster Category", state_info["cluster_description"])
    
    with col2:
        st.metric("Total IPC Crimes (2001-2014)", f"{int(state_info['total_ipc_crimes']):,}")
        st.metric("Women Crimes (2001-2014)", f"{int(state_info['total_women_crimes']):,}")
    
    # Modern Bubble Map - Clean & Zoomable to District Level
    st.subheader("🗺️ India Crime Pattern Clusters - Interactive Bubble Map")
    
    # Define color mapping for clusters
    cluster_colors = {
        0: 'rgba(52, 168, 83, 0.6)',   # Green - Low Crime
        1: 'rgba(251, 188, 5, 0.6)',   # Yellow - Medium-Low Crime
        2: 'rgba(255, 109, 0, 0.6)',   # Orange - Medium-High Crime
        3: 'rgba(234, 67, 53, 0.6)'    # Red - High Crime
    }
    
    # Add color column to clusters_df
    clusters_df['color'] = clusters_df['cluster'].map(cluster_colors)
    
    # Load district-level data for granularity
    core_ipc_full = load_processed_data()[0]
    
    # Aggregate at district level for detailed zoom
    district_data = core_ipc_full.groupby(['state_ut', 'district'], as_index=False).agg({
        'total_ipc_crimes': 'sum'
    })
    
    # Merge with cluster info
    district_data = district_data.merge(
        clusters_df[['state_ut', 'cluster', 'cluster_description', 'color']],
        on='state_ut',
        how='left'
    )
    
    # District coordinates (approximate - for major districts)
    district_coords = {}
    
    # Add state-level fallback coordinates
    state_coords = {
        'Andhra Pradesh': (15.9129, 79.7400), 'Arunachal Pradesh': (28.2180, 94.7278),
        'Assam': (26.2006, 92.9376), 'Bihar': (25.0961, 85.3131),
        'Chhattisgarh': (21.2787, 81.8661), 'Goa': (15.2993, 74.1240),
        'Gujarat': (22.2587, 71.1924), 'Haryana': (29.0588, 76.0856),
        'Himachal Pradesh': (31.1048, 77.1734), 'Jharkhand': (23.6102, 85.2799),
        'Karnataka': (15.3173, 75.7139), 'Kerala': (10.8505, 76.2711),
        'Madhya Pradesh': (22.9734, 78.6569), 'Maharashtra': (19.7515, 75.7139),
        'Manipur': (24.6637, 93.9063), 'Meghalaya': (25.4670, 91.3662),
        'Mizoram': (23.1645, 92.9376), 'Nagaland': (26.1584, 94.5624),
        'Odisha': (20.9517, 85.0985), 'Punjab': (31.1471, 75.3412),
        'Rajasthan': (27.0238, 74.2179), 'Sikkim': (27.5330, 88.5122),
        'Tamil Nadu': (11.1271, 78.6569), 'Telangana': (18.1124, 79.0193),
        'Tripura': (23.9408, 91.9882), 'Uttar Pradesh': (26.8467, 80.9462),
        'Uttarakhand': (30.0668, 79.0193), 'West Bengal': (22.9868, 87.8550),
        'Andaman & Nicobar Islands': (11.7401, 92.6586), 'Chandigarh': (30.7333, 76.7794),
        'Dadra & Nagar Haveli': (20.1809, 73.0169), 'Daman & Diu': (20.4283, 72.8397),
        'Delhi': (28.7041, 77.1025), 'Jammu & Kashmir': (33.7782, 76.5762),
        'Lakshadweep': (10.5667, 72.6417), 'Puducherry': (11.9416, 79.8083)
    }
    
    # Assign coordinates (use state coords + small offset for districts)
    import numpy as np
    np.random.seed(42)
    
    district_data['lat'] = district_data['state_ut'].map(lambda x: state_coords.get(x, (20, 78))[0])
    district_data['lon'] = district_data['state_ut'].map(lambda x: state_coords.get(x, (20, 78))[1])
    
    # Add small random offset for districts within same state
    district_data['lat'] = district_data['lat'] + np.random.uniform(-0.5, 0.5, len(district_data))
    district_data['lon'] = district_data['lon'] + np.random.uniform(-0.5, 0.5, len(district_data))
    
    # Calculate bubble size (proportional to crime)
    district_data['bubble_size'] = np.sqrt(district_data['total_ipc_crimes']) / 10
    district_data['bubble_size'] = district_data['bubble_size'].clip(5, 60)  # Min-max size
    
    # Create bubble map
    fig_bubble = go.Figure()
    
    # Color mapping
    color_map = {
        'Low Crime Intensity': 'rgba(184, 230, 184, 0.6)',
        'Medium-Low Crime Intensity': 'rgba(255, 230, 109, 0.6)',
        'Medium-High Crime Intensity': 'rgba(255, 140, 66, 0.6)',
        'High Crime Intensity': 'rgba(255, 51, 102, 0.7)'
    }
    
    for cluster in district_data['cluster_description'].unique():
        if pd.isna(cluster):
            continue
            
        cluster_data = district_data[district_data['cluster_description'] == cluster]
        
        fig_bubble.add_trace(go.Scattermapbox(
            lat=cluster_data['lat'],
            lon=cluster_data['lon'],
            mode='markers',
            marker=dict(
                size=cluster_data['bubble_size'],
                color=color_map.get(cluster, 'rgba(150, 150, 150, 0.6)'),
                sizemode='diameter',
                opacity=0.7
            ),
            text=cluster_data['district'],
            customdata=cluster_data[['state_ut', 'total_ipc_crimes']],
            hovertemplate=(
                '<b>%{text}</b><br>' +
                '<b>State:</b> %{customdata[0]}<br>' +
                '<b>Total Crimes:</b> %{customdata[1]:,.0f}<br>' +
                '<extra></extra>'
            ),
            name=cluster,
            showlegend=True
        ))
    
    # Update layout with clean, minimal style
    fig_bubble.update_layout(
        mapbox=dict(
            style='carto-positron',  # Clean light background like reference
            center=dict(lat=22.5, lon=82.5),
            zoom=4,
            bearing=0,
            pitch=0
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(150, 150, 150, 0.5)',
            borderwidth=1,
            font=dict(size=11, color='#333333')
        ),
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#F8F8F8',
        font=dict(family='Arial', color='#333333')
    )
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Interactive instructions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🔍 **Zoom In** - Scroll or pinch to zoom to district level")
    with col2:
        st.info("🎈 **Bubble Size** - Larger bubbles = higher crime volume")
    with col3:
        st.info("🎨 **Colors** - Cluster intensity (green to red)")
    with col4:
        st.info("🖱️ **Hover** - See district details | **Drag** to pan")
    
    # Cluster Distribution Summary
    with st.expander("📊 View Cluster Distribution Summary"):
        cluster_counts = clusters_df["cluster_description"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Number of States"]
        
        fig_dist = px.bar(
            cluster_counts,
            x="Cluster",
            y="Number of States",
            color="Cluster",
            title="Number of States per Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Cluster Centroids Comparison
    st.subheader("🎯 Cluster Centroids Comparison")
    
    if kmeans_info:
        # Get original scale centroids
        centroids_scaled = kmeans_info["cluster_centers"]
        scaler = kmeans_info["scaler"]
        centroids_original = scaler.inverse_transform(centroids_scaled)
        
        # Create dataframe
        centroid_df = pd.DataFrame(
            centroids_original,
            columns=["Total IPC Crimes", "Women Crimes", "Children Crimes"]
        )
        centroid_df["Cluster"] = [f"Cluster {i}" for i in range(len(centroid_df))]
        
        # Melt for plotting
        centroid_melted = centroid_df.melt(
            id_vars="Cluster",
            var_name="Crime Type",
            value_name="Average Value"
        )
        
        fig_centroids = px.bar(
            centroid_melted,
            x="Cluster",
            y="Average Value",
            color="Crime Type",
            barmode="group",
            title="Average Crime Statistics by Cluster",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        st.plotly_chart(fig_centroids, use_container_width=True)
        
        # Model Metrics
        with st.expander("📈 View Clustering Metrics"):
            st.write(f"**Silhouette Score:** {kmeans_info['silhouette_score']:.3f}")
            st.write(f"**Inertia:** {kmeans_info['inertia']:.2f}")
            st.write(f"**Number of Clusters:** {len(centroid_df)}")
    
    # Full data table
    with st.expander("📋 View All States"):
        display_df = clusters_df[[
            "state_ut", "cluster", "cluster_description",
            "total_ipc_crimes", "total_women_crimes", "total_children_crimes"
        ]].sort_values("total_ipc_crimes", ascending=False)
        
        st.dataframe(display_df, use_container_width=True, height=400)


# ========================
# Main App
# ========================

def main():
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("---")
    
    # Tab selection
    tab_selection = st.sidebar.radio(
        "Select Analysis:",
        ["📈 Overview (IPC)", "👩 Women Crime", "👶 Children Crime", "🗺️ Area Pattern"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About This Dashboard:**
    
    - Data: NCRB / data.gov.in (2001-2014)
    - Models: ARIMA, KMeans
    - Purpose: Academic research
    - All models pre-trained offline
    """)
    
    # Title
    st.title("🚨 Crime Pattern Analysis & Forecasting Dashboard")
    st.markdown("*India (2001-2028) - Aggregated State-Level Analysis*")
    st.markdown("---")
    
    # Render selected tab
    if tab_selection == "📈 Overview (IPC)":
        tab_overview()
    elif tab_selection == "👩 Women Crime":
        tab_women_crime()
    elif tab_selection == "👶 Children Crime":
        tab_children_crime()
    elif tab_selection == "🗺️ Area Pattern":
        tab_area_pattern()


if __name__ == "__main__":
    main()
