import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import imageio
import kaleido
import io
from statsmodels.tsa.arima.model import ARIMA
import textwrap
import pycountry_convert as pc
import re 

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UN SDG Progress Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern professional theme
st.markdown("""
<style>
    :root {
        --primary: #1a73e8;
        --primary-dark: #0d47a1;
        --primary-light: #4285f4;
        --secondary: #34a853;
        --accent: #fbbc05;
        --dark-bg: #121212;
        --dark-surface: #1e1e1e;
        --dark-card: #252525;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --divider: #333333;
    }
    
    * {
        font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-sizing: border-box;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text-primary);
        margin: 0;
        padding: 0;
        line-height: 1.6;
    }
    
    .header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-bottom: 4px solid var(--secondary);
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 700px;
        margin: 0 auto;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .card {
        background: var(--dark-card);
        padding: 0;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid var(--divider);
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    .card-header {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        padding: 1.2rem;
        color: white;
    }
    
    .card-header h2 {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .card-content {
        padding: 1.5rem;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.2rem;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: var(--dark-surface);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--divider);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        border-color: var(--primary);
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary);
    }
    
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-bottom: 0.8rem;
    }
    
    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.3rem 0;
        line-height: 1.2;
    }
    
    .metric-subtext {
        font-size: 0.85rem;
        color: var(--text-secondary);
        opacity: 0.8;
        margin-top: 0.2rem;
    }
    
    .trend-up {
        color: var(--secondary) !important;
    }
    
    .trend-down {
        color: #ff5252 !important;
    }
    
    .performer-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    .performer-card {
        flex: 1 1 0;
        min-width: 0;
        max-width: 160px;
        background: var(--dark-surface);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }
    
    .performer-card:hover {
        transform: translateY(-2px);
        background: #2a2a2a;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .performer-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.3rem;
        font-weight: 500;
        color: var(--text-primary);
        gap: 8px;
        font-size: 0.9rem;
    }
    
    .performer-value {
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin: 0.2rem 0;
    }
    
    .stSelectbox, .stSlider, .stRadio, .stMultiselect {
        margin-bottom: 1.2rem;
    }
    
    .stSlider .thumb {
        background: var(--primary) !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 0.5rem 1rem;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem !important;
        border-radius: 8px !important;
        background: var(--dark-surface) !important;
        transition: all 0.3s ease !important;
        font-weight: 500;
        border: 1px solid var(--divider) !important;
        color: var(--text-secondary) !important;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #2a2a2a !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3) !important;
        border: none !important;
    }
    
    .country-title {
        font-size: 1.7rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: var(--primary-light);
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--divider);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.8rem 0 1.2rem 0;
        color: var(--text-primary);
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--divider);
    }
    
    .chart-container {
        background: var(--dark-surface);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border: 1px solid var(--divider);
    }
    
    .chart-container:hover {
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    
    .grid-layout {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    
    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border: 1px solid var(--divider);
    }
    
    .control-group {
        background: rgba(26, 115, 232, 0.08);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--divider);
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1.2rem;
    }
    
    .info-tooltip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        color: var(--text-secondary);
        font-size: 0.7rem;
        margin-left: 5px;
        cursor: help;
        vertical-align: middle;
    }
    
    .progress-container {
        margin: 1rem 0;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    
    progress {
        width: 100%;
        height: 10px;
        border-radius: 5px;
        background-color: var(--dark-surface);
    }
    
    progress::-webkit-progress-bar {
        background-color: var(--dark-surface);
        border-radius: 5px;
    }
    
    progress::-webkit-progress-value {
        background-color: var(--primary);
        border-radius: 5px;
    }
    
    progress::-moz-progress-bar {
        background-color: var(--primary);
        border-radius: 5px;
    }
    
    .sdg-colors {
        display: flex;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin: 1.2rem auto 0;
        max-width: 700px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    
    .sdg-colors span {
        flex: 1;
    }
    
    /* SDG Colors */
    .sdg-1 { background: #e5243b; }
    .sdg-2 { background: #DDA63A; }
    .sdg-3 { background: #4C9F38; }
    .sdg-4 { background: #C5192D; }
    .sdg-5 { background: #FF3A21; }
    .sdg-6 { background: #26BDE2; }
    .sdg-7 { background: #FCC30B; }
    .sdg-8 { background: #A21942; }
    .sdg-9 { background: #FD6925; }
    .sdg-10 { background: #DD1367; }
    .sdg-11 { background: #FD9D24; }
    .sdg-12 { background: #BF8B2E; }
    .sdg-13 { background: #3F7E44; }
    .sdg-14 { background: #0A97D9; }
    .sdg-15 { background: #56C02B; }
    .sdg-16 { background: #00689D; }
    .sdg-17 { background: #19486A; }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
        .header p {
            font-size: 1rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr 1fr;
        }
        
        .grid-layout {
            grid-template-columns: 1fr;
        }
        
        .stats-container {
            grid-template-columns: 1fr;
        }
    }
    
    /* Plotly chart background fix */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background: transparent !important;
    }
    
    /* Table styling */
    .dataframe {
        background-color: var(--dark-surface) !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe th {
        background-color: var(--primary-dark) !important;
        color: white !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #252525 !important;
    }
    
    .dataframe tr:hover {
        background-color: #2a2a2a !important;
    }
</style>
""", unsafe_allow_html=True)

def wrap_title(text, width=40):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) <= width:
            line += " " + word
        else:
            lines.append(line.strip())
            line = word
    lines.append(line.strip())
    return "<br>".join(lines)


def forecast_arima(series, end_year=2030):
    """Forecast yearly values using ARIMA until December 2030"""
    try:
        series = series.dropna()
        
        if len(series) < 4:  # Need at least 4 data points for ARIMA(1,1,1)
            print("Not enough data for ARIMA")
            return None

        # Ensure index is numeric (years)
        if not np.issubdtype(series.index.dtype, np.integer):
            try:
                series.index = pd.to_datetime(series.index).year
            except:
                series.index = series.index.astype(int)
        
        # Sort by index (years)
        series = series.sort_index()
        
        # Check if we have enough recent data
        last_year = series.index.max()
        if last_year < 2015:  # If data is too old, forecasting might not be reliable
            print("Data too old for reliable forecasting")
            return None
            
        steps = end_year - last_year
        if steps <= 0:
            print(f"ARIMA skipped: last year is {last_year}, cannot forecast to {end_year}")
            return None

        # Fit ARIMA model
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.get_forecast(steps=steps).predicted_mean
        forecast.index = range(last_year + 1, end_year + 1)
        
        return forecast
        
    except Exception as e:
        print(f"ARIMA forecasting failed: {str(e)}")
        return None

def create_time_series_with_forecast(country_data, metric, show_forecast=False):
    country_data = country_data[['year', metric]].dropna().sort_values('year')
    series = country_data.set_index('year')[metric]

    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1e88e5', width=3.5)
    ))

    # Forecast
    if show_forecast and len(series) >= 4:  # Only show forecast if we have enough data
        forecast = forecast_arima(series)
        if forecast is not None:
            goal_met = forecast.iloc[-1] >= 100 if hasattr(forecast, 'iloc') else forecast[-1] >= 100
            forecast_color = '#00c853' if goal_met else '#ff5252'
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color=forecast_color, dash='dot', width=2.5),
                marker=dict(symbol='circle-open')
            ))

    fig.update_layout(
        title=f"<b>{get_metric_display_name(metric)} Over Time</b>",
        xaxis_title="Year",
        yaxis_title=get_metric_display_name(metric),
        height=450,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor='#333', linecolor='#444', tickfont_color='white'),
        yaxis=dict(gridcolor='#333', linecolor='#444', tickfont_color='white'),
    )
    return fig


@st.cache_data
def load_data():
    """Load and preprocess the SDG data"""
    try:
        # Load the Excel file
        df = pd.read_excel('SDG_dataset.xlsx')
        
        # Load metadata
        metadata = pd.read_excel('new_list.xlsx', sheet_name='Codebook')
        
        # Clean country names
        df['Country'] = df['Country'].str.strip()
        
        # The normalized columns already exist in the data with 'n_' prefix
        # No need to recalculate them, just ensure they're properly formatted
        
        # Ensure normalized columns are clipped to 0-100 range
        for _, row in metadata.iterrows():
            ind_code = row['IndCode']
            norm_col = f"n_{ind_code}"
            
            if norm_col in df.columns:
                # Clip values to 0-100 range to handle any outliers
                df[norm_col] = df[norm_col].clip(0, 100)
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    
@st.cache_data
def get_metadata():
    """Load and return indicator metadata"""
    try:
        return pd.read_excel('new_list.xlsx', sheet_name='Codebook')
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return pd.DataFrame()

def get_indicator_metadata(indicator_code):
    """Get metadata for a specific indicator"""
    try:
        metadata = get_metadata()
        base_code = indicator_code[2:]  # Remove 'n_' prefix
        row = metadata[metadata['IndCode'] == base_code]
        return row.iloc[0] if not row.empty else pd.Series(dtype=object)
    except Exception as e:
        print(f"Error getting metadata: {str(e)}")
        return pd.Series(dtype=object)


def get_metric_options():
    """Define available metrics using metadata and existing columns in data"""
    metrics = {}
    metadata = get_metadata()
    available_columns = load_data().columns  # Get columns from SDG_dataset

    # Only include normalized indicators that exist in the data
    for _, row in metadata.iterrows():
        ind_code = row['IndCode']
        norm_col = f"n_{ind_code}"
        if norm_col in available_columns:
            indicator_name = row['Indicator']
            display_name = f"SDG {row['SDG']}: {indicator_name}"

            # If RED > GREEN, lower is better; add note
            red = row['Red threshold']
            green = row['Green threshold']
            if pd.notna(red) and pd.notna(green) and red > green:
                display_name += " (progress towards target)"

            metrics[norm_col] = display_name

    # Add SDG Index and goal-level scores if they exist
    if 'sdgi_s' in available_columns:
        metrics['sdgi_s'] = 'SDG Index Score'
    
    for i in range(1, 18):
        goal_col = f'goal{i}'
        if goal_col in available_columns:
            metrics[goal_col] = f'SDG {i}'

    return metrics



def create_world_map(df, metric, year):
    """Create an interactive world map with proper title wrapping"""
    # Get display name and wrap it if too long
    display_name = get_metric_display_name(metric)
    if len(display_name) > 50:  # Adjust threshold as needed
        display_name = "<br>".join(textwrap.wrap(display_name, width=40))
    
    # Filter data for the selected year
    year_data = df[df['year'] == year].copy()
    
    # Create hover text
    hover_text = []
    for _, row in year_data.iterrows():
        if pd.isna(row[metric]):
            hover_text.append(f"<b>{row['Country']}</b><br>Data unavailable for {year}")
        else:
            hover_text.append(f"<b>{row['Country']}</b><br>{display_name}: {row[metric]:.2f}<br>Year: {year}")
    
    year_data['hover_text'] = hover_text
    
    # Determine color scale direction
    base_code = metric[2:]
    meta = get_indicator_metadata(metric)
    color_scale = 'RdYlGn'
    
    # Create the map
    fig = px.choropleth(
        year_data,
        locations='Country',
        locationmode='country names',
        color=metric,
        hover_name='Country',
        color_continuous_scale=color_scale,
        custom_data=['hover_text']
    )
    
    # Update layout with proper title handling
    fig.update_layout(
        title={
            'text': f'<b>{display_name} - {year}</b>',
            'y':0.95,  # Position from top
            'x':0.5,   # Center horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 18,
                'color': 'white'
            }
        },
        title_font_size=18,  # Slightly smaller font
        title_x=0.5,        # Centered
        margin=dict(t=100, b=0, l=0, r=0),  # More top margin for title
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='rgba(0,0,0,0)',
            landcolor='#2a2a2a',
            lakecolor='#121212',
            oceancolor='#121212'
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Segoe UI, sans-serif",
            color='white'
        ),
        coloraxis_colorbar=dict(
            title_font_color='white',
            tickfont_color='white',
            title_text=''  # Remove redundant title from colorbar
        )
    )
    
    # Update traces for custom hover
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker_line_color='white',
        marker_line_width=0.8
    )
    
    return fig

def get_metric_display_name(metric):
    """Get display name for metric"""
    metric_options = get_metric_options()
    return metric_options.get(metric, metric)

def create_sdg_radar_chart(country_data, year):
    """Create a radar chart for all SDG goals with dark theme"""
    
    sdg_goals = [f'goal{i}' for i in range(1, 18)]
    goal_names = [f'SDG {i}' for i in range(1, 18)]
    
    # Get values for the selected year
    values = []
    for goal in sdg_goals:
        if goal in country_data.columns:
            goal_data = country_data[country_data['year'] == year][goal]
            if not goal_data.empty and not pd.isna(goal_data.iloc[0]):
                values.append(goal_data.iloc[0])
            else:
                values.append(0)  # Use 0 for missing data
        else:
            values.append(0)
    
    # Create radar chart with dark theme
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=goal_names,
        fill='toself',
        name=f'{year}',
        line_color='#1e88e5',
        fillcolor='rgba(30, 136, 229, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#444',
                angle=90,
                tickfont_color='white'
            ),
            angularaxis=dict(
                gridcolor='#444',
                linecolor='#444',
                tickfont_color='white'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title="<b>SDG Goals Performance</b>",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Segoe UI, sans-serif",
            color="white"
        )
    )
    
    return fig

def create_time_series_chart(country_data, metric):
    """Create a time series chart for a specific metric with dark theme"""
    
    # Sort by year
    country_data = country_data.sort_values('year')
    
    # Determine line color based on trend
    if len(country_data) > 1:
        first_val = country_data[metric].iloc[0]
        last_val = country_data[metric].iloc[-1]
        line_color = '#00c853' if last_val > first_val else '#ff5252'
    else:
        line_color = '#1e88e5'
    
    fig = px.line(
        country_data,
        x='year',
        y=metric,
        title=f'<b>{get_metric_display_name(metric)} Over Time</b>',
        line_shape='spline',
        markers=True
    )
    
    # Update layout for dark theme
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=get_metric_display_name(metric),
        height=450,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Segoe UI, sans-serif",
            color="white"
        ),
        xaxis=dict(
            gridcolor='#333',
            linecolor='#444',
            tickfont_color='white'
        ),
        yaxis=dict(
            gridcolor='#333',
            linecolor='#444',
            tickfont_color='white'
        ),
        legend=dict(
            font=dict(
                color="white"
            )
        )
    )
    
    fig.update_traces(
        line=dict(color=line_color, width=3.5),
        marker=dict(size=9, color=line_color)
    )
    
    return fig

def create_comparison_chart(df, countries, metric):
    """Create comparison line chart for multiple countries with dark theme"""
    # Filter data for selected countries
    comparison_data = df[df['Country'].isin(countries)]
    
    if comparison_data.empty:
        return None
    
    # Create figure
    fig = px.line(
        comparison_data,
        x='year',
        y=metric,
        color='Country',
        title=f'<b>Comparison: {get_metric_display_name(metric)}</b>',
        line_shape='spline',
        markers=True
    )
    
    # Update layout for dark theme
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=get_metric_display_name(metric),
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title='Country',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color="white"
            )
        ),
        font=dict(
            family="Segoe UI, sans-serif",
            color="white"
        ),
        xaxis=dict(
            gridcolor='#333',
            linecolor='#444',
            tickfont_color='white'
        ),
        yaxis=dict(
            gridcolor='#333',
            linecolor='#444',
            tickfont_color='white'
        )
    )
    
    return fig


# --- Helper function for GIF generation ---
@st.cache_data
def generate_metric_gif(df, metric, years=range(2000, 2025)):
    """Generate an animated GIF with year displayed in bottom left"""
    images = []
    
    for year in years:
        year_data = df[df['year'] == year]
        if year_data.empty or metric not in year_data.columns:
            continue
            
        try:
            fig = go.Figure()
            
            # Create hover text
            hover_text = [
                f"<b>{row['Country']}</b><br>{get_metric_display_name(metric)}: {row[metric]:.2f}<br>Year: {year}"
                if not pd.isna(row[metric]) else f"<b>{row['Country']}</b><br>Data unavailable for {year}"
                for _, row in year_data.iterrows()
            ]
            
            # Add choropleth trace
            fig.add_trace(go.Choropleth(
                locations=year_data['Country'],
                locationmode='country names',
                z=year_data[metric],
                colorscale='RdYlGn',
                hovertext=hover_text,
                marker_line_color='white',
                marker_line_width=0.5
            ))
            
            # Update layout with year annotation
            fig.update_layout(
                title_text=f'<b>{get_metric_display_name(metric)}</b>',  # Removed year from title
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular',
                    bgcolor='rgba(0,0,0,0)',
                    landcolor='#2a2a2a'
                ),
                height=600,
                margin=dict(t=80, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                annotations=[
                    dict(
                        x=0.01,  # Left position
                        y=0.05,  # Bottom position
                        xref='paper',
                        yref='paper',
                        text=f'<b>{year}</b>',
                        showarrow=False,
                        font=dict(
                            size=24,
                            color='white'
                        ),
                        bgcolor='rgba(0,0,0,0.5)',
                        bordercolor='white',
                        borderwidth=1,
                        borderpad=4
                    )
                ]
            )
            
            # Export to PNG
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=1)
            images.append(imageio.v3.imread(img_bytes))
            
        except Exception as e:
            print(f"Error generating frame for {year}: {str(e)}")
            continue
    
    if images:
        buf = io.BytesIO()
        imageio.mimsave(
            buf,
            images,
            format="GIF",
            duration=500,  # 0.5 seconds per frame
            loop=0
        )
        buf.seek(0)
        return buf
    return None


def main():
    # Professional header with SDG colors
    st.markdown("""
    <div class="header">
        <h1>UN Sustainable Development Goals Progress Tracker</h1>
        <p>Monitoring global progress towards the 2030 Agenda for Sustainable Development</p>
        <div class="sdg-colors">
            <span class="sdg-1"></span>
            <span class="sdg-2"></span>
            <span class="sdg-3"></span>
            <span class="sdg-4"></span>
            <span class="sdg-5"></span>
            <span class="sdg-6"></span>
            <span class="sdg-7"></span>
            <span class="sdg-8"></span>
            <span class="sdg-9"></span>
            <span class="sdg-10"></span>
            <span class="sdg-11"></span>
            <span class="sdg-12"></span>
            <span class="sdg-13"></span>
            <span class="sdg-14"></span>
            <span class="sdg-15"></span>
            <span class="sdg-16"></span>
            <span class="sdg-17"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure the data files are available.")
        return
    
    # Create tabs with descriptive labels
    tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Performance", "Cross-Country Comparison"])
    
    with tab1:
        # Global Performance Overview Card
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h2>Global Performance Dashboard</h2>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        # Year and metric selection with clear labels
        st.markdown('<div class="control-group">', unsafe_allow_html=True)
        col_year, col_metric = st.columns(2)
        
        with col_year:
            # Year selection with descriptive tooltip
            years = sorted(df['year'].unique())
            selected_year = st.select_slider(
                "Select Year for Analysis",
                options=years,
                value=max(years) if years else 2023,
                help="Select a year to view global performance metrics for that specific year",
                key='year_slider'
            )
        
        with col_metric:
            # Metric selection with categorization
            metric_options = get_metric_options()
            selected_metric = st.selectbox(
                "Select Development Indicator",
                options=list(metric_options.keys()),
                format_func=lambda x: metric_options[x],
                index=0,
                help="Choose an indicator to analyze from the UN Sustainable Development Goals framework",
                key='metric_select'
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # World map with improved title
        try:
            map_fig = create_world_map(df, selected_metric, selected_year)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(map_fig, use_container_width=True)
            
            # GIF Download Button with better label
            st.markdown('<div style="display:flex;justify-content:flex-end;margin-bottom:1rem;">', unsafe_allow_html=True)
            if st.button("Generate Historical Trend Animation", key="download_gif_btn"):
                with st.spinner("Creating animation (this may take 20-30 seconds)..."):
                    gif_bytes = generate_metric_gif(df, selected_metric)
                    st.download_button(
                        label="Download Historical Trend GIF",
                        data=gif_bytes,
                        file_name=f"SDG_{selected_metric}_historical_trend.gif",
                        mime="image/gif"
                    )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying world map: {str(e)}")
        
        # Enhanced metadata display with expandable sections
        with st.expander("Indicator Technical Details", expanded=False):
            base_code = selected_metric[2:]  # Remove 'n_' prefix
            meta = get_indicator_metadata(selected_metric)
            
            if not meta.empty:
                st.markdown(f"""
                **Description:** {meta['Description']}  
                **Target Value:** 100 (normalized scale)  
                **Performance Levels:**  
                - Good: ≥80  
                - Moderate: 50-80  
                - Poor: <50  
                **Source:** [{meta['Source']}]({meta['Dwldlink']})  
                **Years Available:** {meta['Years used']}  
                **Data Methodology:** {meta['Imputation']}
                """)
            else:
                st.warning("Technical details not available for this indicator")

        # Performance metrics section with better organization
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            # Performance Metrics Card with clearer labels
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <h2>Key Performance Metrics</h2>
                </div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            # Filter data for selected year
            year_data = df[df['year'] == selected_year]

            if not year_data.empty and selected_metric in year_data.columns:
                # Calculate statistics
                valid_data = year_data[selected_metric].dropna()

                if not valid_data.empty:
                    avg_value = valid_data.mean()
                    max_value = valid_data.max()
                    min_value = valid_data.min()
                    countries_with_data = len(valid_data)
                    total_countries = len(year_data)
                    coverage = f"{100 * countries_with_data / total_countries:.1f}%"

                    # Calculate trend if possible
                    prev_year = selected_year - 1 if selected_year > min(years) else None
                    trend_value = None
                    if prev_year:
                        prev_data = df[df['year'] == prev_year][selected_metric].dropna().mean()
                        if not pd.isna(prev_data):
                            trend_value = avg_value - prev_data

                    # Display metrics in cards with better labels
                    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

                    # Metric cards
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-title">Global Average Score</div>
                            </div>
                            <div class="metric-value">{avg_value:.1f}</div>
                            <div class="metric-subtext">Across {countries_with_data} countries</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-title">Highest Score</div>
                            </div>
                            <div class="metric-value">{max_value:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-title">Lowest Score</div>
                            </div>
                            <div class="metric-value">{min_value:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cols[3]:
                        trend_class = "trend-up" if trend_value and trend_value >= 0 else "trend-down" if trend_value else ""
                        trend_display = f"{trend_value:+.1f}" if trend_value else "N/A"
                        trend_text = "Improving" if trend_value and trend_value >= 0 else "Declining" if trend_value else "No trend data"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-title">Yearly Change</div>
                            </div>
                            <div class="metric-value {trend_class}">{trend_display}</div>
                            <div class="metric-subtext">{trend_text} from previous year</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)  # Close metric-grid

                else:
                    st.warning("No data available for the selected indicator and year")
            else:
                st.warning("Selected indicator not found in the dataset")

            st.markdown("</div></div>", unsafe_allow_html=True)  # Close card

        with col2:
            # Data Coverage Card with progress bar
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <h2>Data Coverage</h2>
                </div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            if not year_data.empty and selected_metric in year_data.columns:
                countries_with_data = year_data[selected_metric].dropna().count()
                total_countries = year_data['Country'].nunique()
                coverage_pct = 100 * countries_with_data / total_countries
                
                st.markdown(f"""
                <div class="progress-container">
                    <div class="progress-label">
                        <span>Countries with data:</span>
                        <span>{countries_with_data}/{total_countries} ({coverage_pct:.1f}%)</span>
                    </div>
                    <progress value="{countries_with_data}" max="{total_countries}"></progress>
                </div>
                """, unsafe_allow_html=True)
                
                # Coverage quality indicator
                if coverage_pct >= 80:
                    coverage_status = "Excellent"
                    status_color = "#34a853"
                elif coverage_pct >= 50:
                    coverage_status = "Moderate"
                    status_color = "#fbbc05"
                else:
                    coverage_status = "Limited"
                    status_color = "#ea4335"
                
                st.markdown(f"""
                <div style="margin-top: 1rem; text-align: center;">
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">Data Coverage Quality:</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{coverage_status}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("No data coverage information available for this indicator.")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Top & Bottom Performers with improved presentation
        if not year_data.empty and selected_metric in year_data.columns:
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <h2>Performance Leaders and Laggards</h2>
                </div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            top_5 = year_data.nlargest(5, selected_metric)[['Country', selected_metric]].reset_index(drop=True)
            bottom_5 = year_data.nsmallest(5, selected_metric)[['Country', selected_metric]].reset_index(drop=True)

            st.markdown("**Top Performing Countries**", help="Countries with the highest scores for this indicator")
            cols_top = st.columns(5)
            for i, row in top_5.iterrows():
                if not pd.isna(row[selected_metric]):
                    with cols_top[i]:
                        st.markdown(f"""
                        <div class="performer-card">
                            <div class="performer-header">
                                <span>{i+1}.</span>
                                <span>{row['Country']}</span>
                            </div>
                            <div class="performer-value" style="color: #34a853;">{row[selected_metric]:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("**Countries Needing Improvement**", help="Countries with the lowest scores for this indicator")
            cols_bottom = st.columns(5)
            for i, row in bottom_5.iterrows():
                if not pd.isna(row[selected_metric]):
                    with cols_bottom[i]:
                        st.markdown(f"""
                        <div class="performer-card">
                            <div class="performer-header">
                                <span>{i+1}.</span>
                                <span>{row['Country']}</span>
                            </div>
                            <div class="performer-value" style="color: #ea4335;">{row[selected_metric]:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)  # Close card
        else:
            st.warning("No performance comparison data available for this selection")

        # Regional analysis section with improved organization
        st.markdown("""
        <div class="section-title">Regional Performance Analysis</div>
        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;">
            Explore how different world regions are performing on this indicator
        </p>
        """, unsafe_allow_html=True)

        # Map countries to regions
        @st.cache_data
        def map_countries_to_regions(df):
            def get_region(country):
                try:
                    code = pc.country_name_to_country_alpha2(country)
                    continent = pc.country_alpha2_to_continent_code(code)
                    return pc.convert_continent_code_to_continent_name(continent)
                except:
                    return "Other"
            df['Region'] = df['Country'].apply(get_region)
            return df

        df = map_countries_to_regions(df)
        year_data = df[df['year'] == selected_year].copy()

        # Performance categorization with clear thresholds
        def categorize(value):
            if pd.isna(value): return "No Data"
            elif value >= 80: return "Good Performance"
            elif value >= 50: return "Moderate Performance"
            else: return "Poor Performance"

        year_data['Performance'] = year_data[selected_metric].apply(categorize)

        # Interactive filters with better labels
        regions = sorted(year_data['Region'].unique())
        selected_regions = st.multiselect(
            "Filter by Geographic Region:",
            options=regions,
            default=regions,
            help="Select one or more regions to focus your analysis"
        )
        
        perf_cats = ['Good Performance','Moderate Performance','Poor Performance','No Data']
        selected_perf = st.multiselect(
            "Filter by Performance Level:",
            options=perf_cats,
            default=perf_cats,
            help="Filter countries based on their performance level"
        )
        
                
        # Apply filters
        filtered_data = year_data[
            year_data['Region'].isin(selected_regions) & 
            year_data['Performance'].isin(selected_perf)
        ].copy()

        # Visualization row 1: Regional averages and distribution
        col1, col2 = st.columns(2)
        with col1:
            region_avg = filtered_data.groupby('Region')[selected_metric].mean().reset_index()
            fig_bar = px.bar(
                region_avg, 
                x='Region', 
                y=selected_metric, 
                color='Region',
                title="Average Scores by Region",
                labels={selected_metric: "Average Score", "Region": "Geographic Region"}
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Region",
                yaxis_title="Average Score"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            perf_counts = filtered_data['Performance'].value_counts().reindex(perf_cats, fill_value=0).reset_index()
            perf_counts.columns = ['Performance Level','Count']
            fig_donut = px.pie(
                perf_counts, 
                names='Performance Level', 
                values='Count', 
                hole=0.4,
                title="Performance Level Distribution",
                color='Performance Level',
                color_discrete_map={
                    'Good Performance': '#34a853',
                    'Moderate Performance': '#fbbc05',
                    'Poor Performance': '#ea4335',
                    'No Data': '#9e9e9e'
                }
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                legend_title="Performance Level"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Visualization row 2: Detailed breakdowns
        st.markdown("""
        <div class="section-title">Detailed Performance Breakdown</div>
        """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            # Stacked bar showing performance by region
            stacked_data = filtered_data.groupby(['Region','Performance'])[selected_metric].count().reset_index(name='Count')
            fig_stacked = px.bar(
                stacked_data, 
                x='Region', 
                y='Count', 
                color='Performance',
                title="Country Counts by Region and Performance",
                labels={'Count': "Number of Countries", "Region": "Geographic Region"},
                color_discrete_map={
                    'Good Performance': '#34a853',
                    'Moderate Performance': '#fbbc05',
                    'Poor Performance': '#ea4335',
                    'No Data': '#9e9e9e'
                }
            )
            fig_stacked.update_layout(
                barmode='stack',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Region",
                yaxis_title="Number of Countries"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
            
        with col4:
            # Horizontal bar showing countries per region
            region_counts = filtered_data['Region'].value_counts().reset_index()
            region_counts.columns=['Region','Count']
            fig_hbar = px.bar(
                region_counts, 
                x='Count', 
                y='Region', 
                orientation='h',
                title="Countries Represented in Each Region",
                labels={'Count': "Number of Countries", "Region": "Geographic Region"}
            )
            fig_hbar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                yaxis=dict(autorange='reversed'),
                xaxis_title="Number of Countries",
                yaxis_title="Region"
            )
            st.plotly_chart(fig_hbar, use_container_width=True)

        # Top performers visualization
        st.markdown("""
        <div class="section-title">Top Performing Countries</div>
        """, unsafe_allow_html=True)
        
        # Horizontal bar for top performers
        top10 = filtered_data.nlargest(10, selected_metric)[['Country', selected_metric, 'Region']]
        fig_top = px.bar(
            top10, 
            x=selected_metric, 
            y='Country',
            orientation='h',
            color='Region',
            title="Top 10 Performing Countries",
            labels={selected_metric: "Score", "Country": "Country"}
        )
        fig_top.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(autorange='reversed'),
            xaxis_title="Score",
            yaxis_title="Country"
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # Performance category breakdown by region
        st.markdown("""
        <div class="section-title">Regional Performance Composition</div>
        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;">
            How each region's countries are distributed across performance levels
        </p>
        """, unsafe_allow_html=True)
        
        # Create tabs for each performance category
        tab_good, tab_moderate, tab_poor = st.tabs(["Good Performance", "Moderate Performance", "Poor Performance"])
        
        with tab_good:
            subset = filtered_data[filtered_data['Performance'] == 'Good Performance']
            if not subset.empty:
                rc = subset['Region'].value_counts().reset_index()
                rc.columns = ['Region','Count']
                fig_rcpie = px.pie(
                    rc, 
                    names='Region', 
                    values='Count', 
                    title="Regional Distribution of Countries with Good Performance",
                    color='Region'
                )
                fig_rcpie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    legend_title="Region"
                )
                st.plotly_chart(fig_rcpie, use_container_width=True)
            else:
                st.info("No countries with good performance in the selected filters")
                
        with tab_moderate:
            subset = filtered_data[filtered_data['Performance'] == 'Moderate Performance']
            if not subset.empty:
                rc = subset['Region'].value_counts().reset_index()
                rc.columns = ['Region','Count']
                fig_rcpie = px.pie(
                    rc, 
                    names='Region', 
                    values='Count', 
                    title="Regional Distribution of Countries with Moderate Performance",
                    color='Region'
                )
                fig_rcpie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    legend_title="Region"
                )
                st.plotly_chart(fig_rcpie, use_container_width=True)
            else:
                st.info("No countries with moderate performance in the selected filters")
                
        with tab_poor:
            subset = filtered_data[filtered_data['Performance'] == 'Poor Performance']
            if not subset.empty:
                rc = subset['Region'].value_counts().reset_index()
                rc.columns = ['Region','Count']
                fig_rcpie = px.pie(
                    rc, 
                    names='Region', 
                    values='Count', 
                    title="Regional Distribution of Countries with Poor Performance",
                    color='Region'
                )
                fig_rcpie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    legend_title="Region"
                )
                st.plotly_chart(fig_rcpie, use_container_width=True)
            else:
                st.info("No countries with poor performance in the selected filters")
    
    with tab2:
        # Country and year selection
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox(
            "Select Country:",
            countries,
            index=0,
            key='ctry2'
        )
        country_years = sorted(df['year'].unique())
        country_year = st.select_slider(
            "Select Year:",
            options=country_years,
            value=country_years[-1],
            key='year2'
        )
        metric_options = get_metric_options()
        selected_metric = st.selectbox(
            "Select Indicator:",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            key='metric2'
        )

        # Filter data
        country_data = df[df['Country'] == selected_country]
        filtered = country_data[country_data['year'] == country_year]
        if filtered.empty:
            st.error(f"No data available for {selected_country} in {country_year}")
            return
        latest = filtered.iloc[0]

        meta = get_indicator_metadata(selected_metric)

        if meta is not None and 'SDG' in meta and pd.notna(meta['SDG']):
            try:
                sdg_num = int(meta['SDG'])
                parent_col = f'goal{sdg_num}'
                sub_cols = [col for col in df.columns if col.startswith(f'goal{sdg_num}_')]
            except:
                parent_col = None
                sub_cols = []
        else:
            # Fallback: no subgoals, no parent
            parent_col = None
            sub_cols = []

        # Header
        st.markdown(f"""
        <div class="country-title">{selected_country} — {country_year}</div>
        """, unsafe_allow_html=True)

        # Two columns: Radar + Gauge
        col1, col2 = st.columns(2)

        # Radar chart
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            radar_fig = create_sdg_radar_chart(country_data, country_year)
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Gauge chart: selected metric vs goal
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            value = latest[selected_metric] if not pd.isna(latest[selected_metric]) else 0
            ref_value = latest.get(parent_col, 100)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                delta={'reference': ref_value},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#1e88e5'},
                    'steps': [
                        {'range': [0, 50], 'color': '#ff5252'},
                        {'range': [50, 80], 'color': '#ffd600'},
                        {'range': [80, 100], 'color': '#00c853'}
                    ]
                },
                title={'text': wrap_title(metric_options[selected_metric])}
            ))
            gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=450)
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Get SDG number from selected metric
        meta = get_indicator_metadata(selected_metric)
        if selected_metric.startswith('goal'):
            try:
                sdg_num = int(selected_metric.replace('goal', ''))
                metadata_df = get_metadata()
                # collect all the sub‑indicator codes for this SDG
                sub_inds = metadata_df[metadata_df['SDG'] == sdg_num]['IndCode'].tolist()
                sub_cols = [f"n_{ind}" for ind in sub_inds if f"n_{ind}" in df.columns]
                # **make sure to include the main goal column itself first**
                related_indicators = [selected_metric] + sub_cols
            except Exception:
                related_indicators = []

        elif meta is not None and 'SDG' in meta and pd.notna(meta['SDG']):
            # Handle normalized indicators
            try:
                sdg_num = int(meta['SDG'])
                # Find all indicators under this SDG
                metadata = get_metadata()
                related_indicators = metadata[metadata['SDG'] == sdg_num]['IndCode'].tolist()
                related_indicators = [f"n_{ind}" for ind in related_indicators if f"n_{ind}" in df.columns]
                
                # Add the main SDG goal score if available
                goal_col = f'goal{sdg_num}'
                if goal_col in df.columns:
                    related_indicators.insert(0, goal_col)
            except:
                related_indicators = []
        else:
            related_indicators = []

       # Related Indicators Time Series Comparison
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Header with dynamic SDG number
            try:
                if selected_metric.startswith("goal") and selected_metric[4:].isdigit():
                    sdg_display_num = int(selected_metric.replace("goal", ""))
                else:
                    meta = get_indicator_metadata(selected_metric)
                    sdg_display_num = int(meta["SDG"]) if pd.notna(meta["SDG"]) else "?"
            except:
                sdg_display_num = "?"

            # Render title safely
            st.markdown(
                f"<h3 style='color:white; margin-bottom:10px;'>SDG {sdg_display_num} Component Trends</h3>",
                unsafe_allow_html=True
            )

            
            if related_indicators:
                # Prepare and clean data
                ts_data = country_data[['year'] + related_indicators].sort_values('year')
                melted_data = ts_data.melt(id_vars='year', var_name='Indicator', value_name='Score')
                melted_data = melted_data.dropna(subset=['Score'])


                
                # Create cleaner display names
                def clean_indicator_name(indicator):
                    name = metric_options.get(indicator, indicator)
                    name = re.sub(r'^SDG \d+:\s*', '', name)
                    name = re.sub(r'\(progress towards target\)', '', name)
                    return name

                
                melted_data['DisplayName'] = melted_data['Indicator'].map(clean_indicator_name)
                main_sdg_col = selected_metric if selected_metric.startswith("goal") else None
                melted_data["IsMainSDG"] = melted_data["Indicator"] == main_sdg_col

                # Create interactive line chart
                fig = px.line(
                    melted_data,
                    x='year',
                    y='Score',
                    color='DisplayName',
                    line_shape='spline',
                    markers=True,
                    hover_name='DisplayName',
                    hover_data={'DisplayName': False, 'Indicator': False, 'year': True, 'Score': ':.1f'}
                )
                
                # Enhanced layout
                fig.update_layout(
                    title_text="",  # We're using our own header
                    xaxis_title="Year",
                    yaxis_title="Performance Score",
                    height=450,
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    legend=dict(
                        title=None,
                        orientation="h",
                        yanchor="bottom",
                        y=-0.4,  # Position below chart
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10),
                        itemwidth=30
                    ),
                    margin=dict(t=30, b=100, l=50, r=50),  # Add space for legend
                    xaxis=dict(
                        gridcolor='#333', 
                        linecolor='#444', 
                        tickfont_color='white',
                        fixedrange=True  # Disable zoom
                    ),
                    yaxis=dict(
                        gridcolor='#333', 
                        linecolor='#444', 
                        tickfont_color='white',
                        range=[0, 100]  # Fixed scale for comparison
                    )
                )
                
                # Customize hover appearance
                fig.update_traces(
                    hovertemplate="<b>%{hovertext}</b><br>Year: %{x}<br>Score: %{y:.1f}<extra></extra>",
                    line=dict(width=2.5)
                )
                
                # Display with tight layout
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            else:
                st.info(
                    "No component indicators available for this SDG goal score. "
                    "Try selecting a specific indicator from the dropdown above.",
                    icon="ℹ️"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Waterfall chart: Year-over-year changes for selected metric
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        country_series = country_data[['year', selected_metric]].dropna().sort_values('year')
        if len(country_series) > 1:
            years = country_series['year'].astype(str).tolist()
            values = country_series[selected_metric].tolist()
            diffs = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]
            measures = ['absolute'] + ['relative'] * (len(diffs)-1)
            wf = go.Figure(go.Waterfall(
                x=years,
                y=diffs,
                measure=measures,
                text=[f"{v:.2f}" for v in diffs],
                textposition="outside"
            ))
            wf.update_layout(
                title=f"Yearly Change: {metric_options[selected_metric]}",
                xaxis_title="Year",
                yaxis_title=metric_options[selected_metric],
                paper_bgcolor='rgba(0,0,0,0)', font_color='white', waterfallgap=0.3
            )
            st.plotly_chart(wf, use_container_width=True)
        else:
            st.warning("Not enough data for waterfall chart")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Cross-Country Comparison with improved interface
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h2>Cross-Country Performance Comparison</h2>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        # Comparison controls with better organization
        st.markdown('<div class="control-group">', unsafe_allow_html=True)
        col_metric, col_countries = st.columns([1, 2])
        
        with col_metric:
            # Metric selection with search
            metric_options = get_metric_options()
            compare_metric = st.selectbox(
                "Select Indicator to Compare:",
                options=list(metric_options.keys()),
                format_func=lambda x: metric_options[x],
                index=0,
                help="Choose an indicator to compare across countries",
                key='compare_metric'
            )
        
        with col_countries:
            # Country selection with grouping and search
            countries = sorted(df['Country'].unique())
            selected_countries = st.multiselect(
                "Select Countries to Compare (2-5 recommended):",
                options=countries,
                default=[countries[0], countries[min(1, len(countries)-1)]] if len(countries) > 1 else [countries[0]],
                help="Select multiple countries to compare their performance",
                key='compare_countries'
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison visualization
        if selected_countries:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Create comparison figure
            fig = go.Figure()
            
            # Color palette for countries
            colors = px.colors.qualitative.Plotly
            
            for i, country in enumerate(selected_countries):
                country_data = df[df['Country'] == country][['year', compare_metric]].dropna().sort_values('year')
                series = country_data.set_index('year')[compare_metric]
                
                if series.empty:
                    continue
                
                # Actual line with unique color
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines+markers',
                    name=country,
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    marker=dict(size=8, color=colors[i % len(colors)])
                ))
            
            if fig.data:
                # Add target line if available
                meta = get_indicator_metadata(compare_metric)
                if meta is not None and pd.notna(meta['Green threshold']):
                    # In comparison charts, replace any dynamic target lines with:
                    fig.add_hline(
                        y=100,
                        line_dash="dot",
                        line_color="#34a853",
                        annotation_text="Target Level (100)",
                        annotation_position="bottom right",
                        annotation_font_color="#34a853"
                    )
                
                fig.update_layout(
                    title=f"<b>Comparison: {get_metric_display_name(compare_metric)}</b>",
                    xaxis_title="Year",
                    yaxis_title=get_metric_display_name(compare_metric),
                    height=500,
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    legend=dict(
                        title='Country',
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(color="white")
                    ),
                    xaxis=dict(gridcolor='#333', linecolor='#444', tickfont_color='white'),
                    yaxis=dict(gridcolor='#333', linecolor='#444', tickfont_color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation notes
                st.markdown("""
                <div style="margin-top: -15px; font-size: 0.9rem; color: var(--text-secondary);">
                    <strong>Interpretation Tips:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.2rem;">
                        <li>Lines moving upward indicate improving performance over time</li>
                        <li>The further apart the lines, the greater the performance gap between countries</li>
                        <li>Dotted green line shows the target level for this indicator (where available)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No comparable data available for the selected countries")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add comparison metrics table
            if len(selected_countries) > 1:
                st.markdown("""
                <div class="section-title">Comparison Metrics</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
                    Key statistics comparing the selected countries
                </p>
                """, unsafe_allow_html=True)
                
                # Get latest data for each country
                comparison_data = []
                for country in selected_countries:
                    country_latest = df[(df['Country'] == country) & 
                                       (df['year'] == df['year'].max())][[compare_metric]]
                    if not country_latest.empty:
                        comparison_data.append({
                            'Country': country,
                            'Latest Score': country_latest[compare_metric].iloc[0],
                            'Year': df['year'].max()
                        })
                
                if comparison_data:
                    # Create comparison table
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('Latest Score', ascending=False)
                    comp_df['Rank'] = range(1, len(comp_df)+1)
                    comp_df['Difference from Leader'] = comp_df['Latest Score'].max() - comp_df['Latest Score']
                    
                    # Format table
                    comp_df['Latest Score'] = comp_df['Latest Score'].apply(lambda x: f"{x:.1f}")
                    comp_df['Difference from Leader'] = comp_df['Difference from Leader'].apply(
                        lambda x: f"{x:.1f}" if x != 0 else "-"
                    )
                    
                    # Display table
                    st.dataframe(
                        comp_df[['Rank', 'Country', 'Latest Score', 'Difference from Leader', 'Year']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No comparable latest data available for the selected countries")
        else:
            st.warning("Please select at least one country to compare")
        
        st.markdown("</div></div>", unsafe_allow_html=True)  # Close comparison card
    
    # Professional footer with links and attribution
    st.markdown("""
    <div class="footer">
        <p>Data Source: United Nations Sustainable Development Goals Database | Last Updated: 2024</p>
        <p>© 2023 UN SDG Progress Tracker | 
           <a href="https://unstats.un.org/sdgs" target="_blank">Official UN SDG Website</a> | 
           <a href="https://sdg-tracker.org" target="_blank">SDG Methodology</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()