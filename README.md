# Crime Pattern Analysis and Trend Forecasting System

A comprehensive machine learning system for analyzing and forecasting crime patterns in India using public NCRB data (2001-2014).

## Overview

This project implements a complete crime analytics platform combining time series forecasting, unsupervised clustering, and interactive visualization to support data-driven policy making and law enforcement resource allocation.

## Key Features

- **Time Series Forecasting**: ARIMA models for predicting crime trends through 2028
- **Pattern Recognition**: KMeans clustering to identify state-level crime patterns
- **Interactive Dashboard**: Streamlit-based visualization system
- **Crime Type Analysis**: Detailed breakdowns for women and children crimes
- **Geospatial Visualization**: District-level interactive maps

## Technology Stack

- **Python 3.13**: Core programming language
- **Statsmodels**: ARIMA time series modeling
- **Scikit-learn**: KMeans clustering and preprocessing
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data manipulation and analysis

## System Architecture

### Data Pipeline
1. Raw data ingestion from NCRB datasets (2001-2014)
2. Standardization and preprocessing
3. Feature engineering and aggregation
4. Model training (offline)
5. Results persistence for dashboard loading

### Models
- **ARIMA(1,2,1)**: Three separate models for IPC crimes, women crimes, and children crimes
- **KMeans(k=4)**: State clustering based on crime patterns
- **Silhouette Score**: 0.618 (indicating good cluster separation)

### Dashboard Tabs
1. **IPC Crimes Overview**: Total crimes with 15-year forecast
2. **Crimes Against Women**: Detailed analysis of 7 crime categories
3. **Crimes Against Children**: Analysis of 11 crime categories
4. **Area Patterns**: Interactive state clustering and district-level maps

## Installation

### Prerequisites
- Python 3.13 or higher
- Virtual environment tool (venv)
- Git

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/ixabrar/Crime-Trend-Analysis-Forecasting.git
cd Crime-Trend-Analysis-Forecasting

# Create virtual environment
python -m venv crime
crime\Scripts\activate  # Windows
source crime/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify data files exist
ls data/processed/core_crime/
ls data/processed/women_crime/
ls data/processed/children_crime/
ls models/
```

## Usage

### Running the Dashboard

```bash
# Activate virtual environment
crime\Scripts\activate

# Launch Streamlit dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Training Models (Optional)

Models are pre-trained and included. To retrain:

```bash
# Prepare standardized datasets
python prepare_data.py

# Train clustering model
python models/kmeans_clustering.py

# Train ARIMA models (use Jupyter notebook)
jupyter notebook notebooks/eda.ipynb
```

## Data Sources

- **NCRB (National Crime Records Bureau)**: Official Indian government crime statistics
- **Data.gov.in**: Public data portal
- **Coverage**: 2001-2014 across all Indian states and union territories
- **Granularity**: District-level data

## Model Performance

### ARIMA Models
- **IPC Crimes**: AIC = 261.23, RMSE validation performed
- **Women Crimes**: AIC = 265.47, stationary after second differencing
- **Children Crimes**: AIC = 263.91, 15-year forecast horizon

### Clustering Model
- **Algorithm**: KMeans with StandardScaler preprocessing
- **Clusters**: 4 (Low, Medium-Low, Medium-High, High crime intensity)
- **Silhouette Score**: 0.618
- **States Clustered**: 69 entities

## Ethical Considerations

- **Aggregated Data Only**: No individual-level information used
- **Academic Purpose**: Research and educational use only
- **No Profiling**: System does not target specific demographics
- **Transparency**: All methodologies documented
- **Public Data**: Uses only officially released government statistics

## Project Structure

```
Crime-Trend-Analysis-Forecasting/
├── app.py                          # Main Streamlit dashboard
├── prepare_data.py                 # Data preprocessing pipeline
├── requirements.txt                # Python dependencies
├── data/
│   ├── processed/                  # Standardized datasets
│   └── dataset/                    # Raw NCRB CSV files
├── models/
│   ├── kmeans_clustering.py        # Clustering implementation
│   ├── arima_ipc.pkl               # Trained IPC model
│   ├── arima_women.pkl             # Trained women crimes model
│   └── arima_children.pkl          # Trained children crimes model
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis and training
└── utils/
    └── preprocessing.py            # Data cleaning utilities
```

## Results and Insights

### Forecast Highlights
- IPC crimes show declining trend through 2028
- Women crimes projected to decrease by 12% (2014-2028)
- Children crimes forecasted to stabilize

### Clustering Insights
- Cluster 0 (Low): 23 states with minimal crime rates
- Cluster 1 (Medium-Low): 18 states with moderate activity
- Cluster 2 (Medium-High): 15 states requiring attention
- Cluster 3 (High): 13 states needing priority intervention

### Top Crime Types
**Women**: Cruelty by relatives (38%), kidnapping (24%), assault (19%)
**Children**: Kidnapping (31%), rape (28%), other crimes (22%)

## Contributing

This is an academic project. For improvements or suggestions:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## License

This project is intended for educational and research purposes. Data sourced from public government repositories.

## Acknowledgments

- National Crime Records Bureau (NCRB) for public data access
- Data.gov.in platform for dataset hosting
- Open source community for tools and libraries

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

## Citation

If using this work for research, please cite:
```
Crime Pattern Analysis and Trend Forecasting System Using Public Data
GitHub: https://github.com/ixabrar/Crime-Trend-Analysis-Forecasting
Year: 2026
```
