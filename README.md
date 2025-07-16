# SDG_dashboard
# UN SDG Progress Tracker (as of July 2025)

## Overview

The UN SDG Progress Tracker is a python-based dashboard for monitoring and analyzing progress towards the United Nations Sustainable Development Goals (SDGs). This tool provides:

- Global overviews of SDG progress
- Country-level performance analysis
- Cross-country comparisons
- ARIMA-based forecasting to 2030 targets
- Regional performance breakdowns

## Features

### 1. Global Overview Dashboard
- Interactive world maps showing SDG performance by country
- Key performance metrics and statistics
- Top/bottom performer analysis
- Regional performance comparisons
- Historical trend animations

<img width="817" height="360" alt="image" src="https://github.com/user-attachments/assets/0c5fd8a7-d0ef-4b4e-a572-8ec480b21181" />


### 2. Country Performance Analysis
- Radar charts showing all 17 SDG scores
- Time series trends with forecasting
- Waterfall charts showing year-over-year changes
- Component indicator analysis for each SDG
- Performance benchmarking

<img width="847" height="202" alt="image" src="https://github.com/user-attachments/assets/47de140c-004a-459e-9e99-1d0837c1d035" />
<img width="868" height="262" alt="image" src="https://github.com/user-attachments/assets/114c39e3-00ac-4222-aa82-849183b9f9f2" />
<img width="892" height="216" alt="image" src="https://github.com/user-attachments/assets/03b00ca4-acee-4b8a-abf4-3f7dfe966238" />

### 3. Cross-Country Comparison
- Side-by-side performance comparisons
- Multi-country trend analysis
- Relative ranking tables
- Progress rate comparisons

<img width="895" height="338" alt="image" src="https://github.com/user-attachments/assets/f5213b53-6299-4521-8895-5dfc80eb301e" />

### 4. 2030 Forecast Module
- ARIMA time series forecasting to 2030 targets
- Performance status indicators (On Track/Off Track)
- Confidence intervals for predictions
- Actionable insights and recommendations
- Multi-country forecast comparisons

<img width="808" height="426" alt="image" src="https://github.com/user-attachments/assets/ecc9df11-ba35-4b67-bcc4-9cd9e6d27fb0" />

## Data Sources

### Primary Data
- **SDG Dataset**: Main indicator data from [Sustainable Development Report Dashboards](https://dashboards.sdgindex.org/downloads)
  - Contains normalized scores (0-100 scale) for SDG indicators
  - Includes country-level data from 2000-2024
  - Goal-level scores and SDG Index scores

### Supplementary Data
- **World Development Indicators**: Metadata and additional indicators from [World Bank DataBank](https://databank.worldbank.org/source/world-development-indicators)
- **UN SDG Metadata**: Indicator descriptions and technical details

## Data Processing

The raw data has undergone significant cleaning and preprocessing:

1. **Year-wise Imputation**:
   - Linear interpolation for missing years where appropriate
   - Carrying forward last known values for stable indicators
   - No imputation for rapidly changing metrics

2. **Geographic Handling**:
   - Blank values for indicators that don't apply (e.g., SDG 14 for landlocked countries)
   - Regional aggregation for analysis

3. **Normalization**:
   - All indicators scaled to 0-100 range for comparability
   - Directionality adjusted so higher values always indicate better performance

4. **Validation**:
   - Range checking for all normalized values
   - Cross-validation with original sources

## Technical Implementation

### Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Statsmodels (for ARIMA forecasting)
- Kaleido (for image export)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/oneekat/SDG_dashboard.git
   cd sdg-tracker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download data files:
   - Place `SDG_dataset.xlsx` in the root directory
   - Place `metadata.xlsx` (metadata) in the root directory

### Running the Application
```bash
streamlit run app.py
```

For the forecasting module:
```bash
streamlit run forecast_2030.py
```

## File Structure
```
sdg-tracker/
├── app.py                 # Main dashboard application
├── forecast_2030.py       # ARIMA forecasting module
├── SDG_dataset.xlsx       # Primary dataset
├── metadata.xlsx          # Metadata file
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── icons/                 # contains icons and images
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- United Nations Sustainable Development Goals
- Sustainable Development Solutions Network
- World Bank World Development Indicators
- Streamlit for the dashboard framework
