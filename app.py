""""
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
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import plotly.graph_objects as go
import plotly.express as px


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
        
        # User-selectable state for district bar chart
        st.markdown("**Top 10 Districts by Crime Type (State Filter)**")
        state_options = ['All India'] + sorted(women_crime['state_ut'].dropna().unique())
        selected_state = st.selectbox(
            "Select State/UT for District Ranking:",
            options=state_options,
            key='women_district_state_select',
            index=state_options.index('All India') if 'All India' in state_options else 0
        )

        if selected_state == 'All India':
            district_crime = women_crime.groupby('district')[selected_crime].sum().sort_values(ascending=False).head(10)
        else:
            district_crime = women_crime[women_crime['state_ut'] == selected_state].groupby('district')[selected_crime].sum().sort_values(ascending=False).head(10)

        fig_bar = go.Figure(data=[go.Bar(
            x=district_crime.values,
            y=district_crime.index,
            orientation='h',
            marker=dict(color='#e74c3c', line=dict(color='#c0392b', width=1))
        )])

        fig_bar.update_layout(
            title=f"Top 10 Districts in {selected_state}: {selected_crime.replace('_', ' ').title()} (2001-2014)",
            xaxis_title="Total Cases",
            yaxis_title="District",
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

        # User-selectable state for district bar chart
        st.markdown("**Top 10 Districts by Crime Type (State Filter)**")
        state_options = ['All India'] + sorted(children_crime['state_ut'].dropna().unique())
        selected_state = st.selectbox(
            "Select State/UT for District Ranking:",
            options=state_options,
            key='children_district_state_select',
            index=state_options.index('All India') if 'All India' in state_options else 0
        )

        if selected_state == 'All India':
            district_crime = children_crime.groupby('district')[selected_crime].sum().sort_values(ascending=False).head(10)
        else:
            district_crime = children_crime[children_crime['state_ut'] == selected_state].groupby('district')[selected_crime].sum().sort_values(ascending=False).head(10)

        fig_bar2 = go.Figure(data=[go.Bar(
            x=district_crime.values,
            y=district_crime.index,
            orientation='h',
            marker=dict(color='#16a085', line=dict(color='#145a32', width=1))
        )])
        fig_bar2.update_layout(
            title=f"Top 10 Districts in {selected_state}: {selected_crime.replace('_', ' ').title()} (2001-2013)",
            xaxis_title="Total Cases",
            yaxis_title="District",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_bar2, use_container_width=True)
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





def area_tab_pattern():

    st.set_page_config(page_title="District Crime Clustering", layout="wide")
    st.title("District Crime Clustering with Real Coordinates")

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    def load_crime_data():
        return pd.read_csv("data/processed/core_crime/core_ipc_standardized.csv")

    def load_latlong_data():
        return pd.read_csv("archive/India Districts Latlongs.csv")

    crime_df = load_crime_data()
    latlong_df = load_latlong_data()

    # ------------------------------------------------------------------
    # 2. Preprocess and Merge Coordinates
    # ------------------------------------------------------------------
    def extract_district_state(place):
        if pd.isna(place):
            return "", ""
        place = str(place)
        tokens = [t.strip() for t in place.split(",")]
        if len(tokens) >= 2:
            state = tokens[-2]
            district = tokens[-3] if len(tokens) >= 3 else tokens[0]
            return district.upper(), state.upper()
        return place.upper(), ""

    latlong_df["district_key"] = latlong_df["Place Name"].apply(
        lambda x: extract_district_state(x)[0]
    )
    latlong_df["state_key"] = latlong_df["Place Name"].apply(
        lambda x: extract_district_state(x)[1]
    )

    crime_df["district_key"] = (
        crime_df["district"]
        .str.upper()
        .str.replace(" DISTRICT", "")
        .str.replace(" ", "")
    )
    crime_df["state_key"] = crime_df["state_ut"].str.upper().str.replace(" ", "")

    latlong_df["district_key"] = latlong_df["district_key"].str.replace(" ", "")
    latlong_df["state_key"] = latlong_df["state_key"].str.replace(" ", "")

    merged = pd.merge(
        crime_df,
        latlong_df,
        how="left",
        left_on=["district_key", "state_key"],
        right_on=["district_key", "state_key"],
    )

    # ------------------------------------------------------------------
    # 3. Aggregate by District
    # ------------------------------------------------------------------
    district_agg = (
        merged.groupby(["state_ut", "district", "Latitude", "Longitude"])
        .agg(
            {
                "total_ipc_crimes": "sum",
                "murder": "sum",
                "rape": "sum",
                "robbery": "sum",
            }
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. KMeans Clustering
    # ------------------------------------------------------------------
    st.sidebar.header("KMeans Clustering Settings")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 4)

    features = ["total_ipc_crimes", "murder", "rape", "robbery"]
    scaler = StandardScaler()
    X = scaler.fit_transform(district_agg[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    district_agg["cluster"] = labels

    # ------------------------------------------------------------------
    # 5. Cluster Quality (Silhouette Score)
    # ------------------------------------------------------------------
    if len(set(labels)) > 1:
        sil_score = silhouette_score(X, labels)
        st.sidebar.success(f"Silhouette Score: {sil_score:.3f}")
    else:
        st.sidebar.warning("Only one cluster found. Silhouette score unavailable.")

    # ------------------------------------------------------------------
    # 6. Visualization – Bubble Map (Green → Red)
    # ------------------------------------------------------------------
    st.subheader("District Crime Clusters (Bubble Map)")

    # Green → Red (8 transitions)
    green_to_red_8 = [
        "rgba(0, 255, 0, 0.55)",     # Bright Green
        "rgba(102, 255, 0, 0.55)",   # Yellow-Green
        "rgba(204, 255, 0, 0.55)",   # Lime-Yellow
        "rgba(255, 255, 0, 0.55)",   # Yellow
        "rgba(255, 204, 0, 0.55)",   # Orange-Yellow
        "rgba(255, 102, 0, 0.55)",   # Orange
        "rgba(255, 51, 0, 0.55)",    # Orange-Red
        "rgba(255, 0, 0, 0.55)"      # Red
    ]

    fig = go.Figure()

    cluster_ids = sorted(district_agg["cluster"].unique())
    num_clusters = len(cluster_ids)
    color_map = green_to_red_8[:num_clusters]

    for idx, cluster in enumerate(cluster_ids):
        cluster_data = district_agg[district_agg["cluster"] == cluster]

        fig.add_trace(
            go.Scattermapbox(
                lat=cluster_data["Latitude"],
                lon=cluster_data["Longitude"],
                mode="markers",
                marker=dict(
                    size=np.log1p(cluster_data["total_ipc_crimes"]) * 4,
                    color=color_map[idx],
                    opacity=0.7,
                    sizemode="diameter",
                ),
                text=cluster_data["district"],
                customdata=np.column_stack(
                    [
                        cluster_data["district"],
                        cluster_data["state_ut"],
                        cluster_data["total_ipc_crimes"],
                        cluster_data["murder"],
                        cluster_data["rape"],
                        cluster_data["robbery"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<b>State:</b> %{customdata[1]}<br>"
                    "<b>Total Crimes:</b> %{customdata[2]:,.0f}<br>"
                    "Murder: %{customdata[3]:,.0f}<br>"
                    "Rape: %{customdata[4]:,.0f}<br>"
                    "Robbery: %{customdata[5]:,.0f}<br>"
                    "<extra></extra>"
                ),
                name=f"Cluster {cluster}",
                showlegend=True,
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=22.5, lon=82.5),
            zoom=4.2,
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(title="Cluster", font=dict(size=12)),
        paper_bgcolor="#f5f5f5",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 7. Data Table
    # ------------------------------------------------------------------
    st.subheader("Clustered District Data Table")
    st.dataframe(district_agg, use_container_width=True)

    # ------------------------------------------------------------------
    # 8. Download Option
    # ------------------------------------------------------------------
    st.download_button(
        "Download Clustered Data as CSV",
        district_agg.to_csv(index=False),
        "district_clusters.csv",
    )

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
        [
            "📈 Overview (IPC)",
            "👩 Women Crime",
            "👶 Children Crime",
            "🗺️ Area Pattern",
        ]
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
        area_tab_pattern()



if __name__ == "__main__":
    main()
