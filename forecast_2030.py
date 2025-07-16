import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import io
import json
from fpdf import FPDF
import plotly.io as pio
from PIL import Image

@st.cache_data
def load_sdg_json_data():
    try:
        with open("sdg_dashboard_complete.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("SDG JSON data file not found. Please ensure the file is available.")
        return None
    
# Page configuration
st.set_page_config(
    page_title="SDG Progress Forecast Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def styled_section(title, content):
    st.markdown(f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px">
        <h4 style="color:#2c3e50;margin-top:0">{title}</h4>
        {content}
    </div>
    """, unsafe_allow_html=True)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .status-excellent { border-left-color: #28a745; }
    .status-good { border-left-color: #ffc107; }
    .status-warning { border-left-color: #fd7e14; }
    .status-poor { border-left-color: #dc3545; }
    .insight-box {
        background: rgba(248, 249, 250, 0.95);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">SDG Progress Forecast Dashboard</h1>
    <p style="color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Advanced Analytics for Sustainable Development Goals using ARIMA Time Series Forecasting</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("SDG_dataset.xlsx")
        return df
    except FileNotFoundError:
        st.error("SDG_dataset.xlsx file not found. Please ensure the data file is available.")
        return None

# SDG Information
SDG_MAPPING = {f"goal{i}": f"SDG {i}" for i in range(1, 18)}
SDG_DESCRIPTIONS = {
    "goal1": "No Poverty - End poverty in all its forms everywhere",
    "goal2": "Zero Hunger - End hunger, achieve food security and improved nutrition",
    "goal3": "Good Health and Well-being - Ensure healthy lives and promote well-being",
    "goal4": "Quality Education - Ensure inclusive and equitable quality education",
    "goal5": "Gender Equality - Achieve gender equality and empower all women and girls",
    "goal6": "Clean Water and Sanitation - Ensure availability and sustainable management of water",
    "goal7": "Affordable and Clean Energy - Ensure access to affordable, reliable, sustainable energy",
    "goal8": "Decent Work and Economic Growth - Promote sustained, inclusive economic growth",
    "goal9": "Industry, Innovation and Infrastructure - Build resilient infrastructure",
    "goal10": "Reduced Inequalities - Reduce inequality within and among countries",
    "goal11": "Sustainable Cities and Communities - Make cities and human settlements inclusive",
    "goal12": "Responsible Consumption and Production - Ensure sustainable consumption patterns",
    "goal13": "Climate Action - Take urgent action to combat climate change",
    "goal14": "Life Below Water - Conserve and sustainably use the oceans, seas and marine resources",
    "goal15": "Life on Land - Protect, restore and promote sustainable use of terrestrial ecosystems",
    "goal16": "Peace, Justice and Strong Institutions - Promote peaceful and inclusive societies",
    "goal17": "Partnerships for the Goals - Strengthen the means of implementation"
}

def get_status_info(color):
    """Return detailed status information based on color coding"""
    status_info = {
        'green': {
            'label': 'On Track',
            'description': 'Country is progressing well and likely to achieve the target by 2030',
            'recommendation': 'Maintain current momentum and policies',
            'css_class': 'status-excellent'
        },
        'yellow': {
            'label': 'Moderate Progress',
            'description': 'Some progress is being made, but acceleration is needed',
            'recommendation': 'Implement targeted interventions to accelerate progress',
            'css_class': 'status-good'
        },
        'orange': {
            'label': 'Slow Progress',
            'description': 'Progress is insufficient to meet 2030 targets',
            'recommendation': 'Significant policy changes and increased investment required',
            'css_class': 'status-warning'
        },
        'red': {
            'label': 'Off Track',
            'description': 'Country is moving away from the target or showing decline',
            'recommendation': 'Immediate intervention required with comprehensive policy overhaul',
            'css_class': 'status-poor'
        }
    }
    return status_info.get(color, status_info['red'])

def determine_trend_color(last_value, predicted_2030):
    """Determine trend color based on forecast from last known value"""
    if predicted_2030 >= 99:
        return 'green'
    elif predicted_2030 >= last_value + 3:
        return 'yellow'
    elif predicted_2030 < last_value:
        return 'red'
    else:
        return 'orange'


def fit_arima_model(data, max_order=3):
    """Fit ARIMA model with automatic parameter selection (based on BIC)"""
    if len(data) < 5:
        return None

    data = data.dropna()
    if len(data) < 5:
        return None

    best_bic = np.inf
    best_model = None
    best_order = None

    for p in range(max_order + 1):
        for d in range(3):  # Try 0, 1, 2 differencing
            for q in range(max_order + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.bic < best_bic:
                        best_bic = fitted_model.bic
                        best_model = fitted_model
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_model:
        print(f"✔️ Best ARIMA order selected: {best_order}, BIC: {best_bic:.2f}")
    return best_model

def predict_sdg_progress(df, country, sdg_col):
    """Generate ARIMA forecast for SDG progress"""
    country_data = df[df['Country'] == country].sort_values('year')
    ts_data = country_data[['year', sdg_col]].dropna()

    if len(ts_data) < 5:
        return None

    years = ts_data['year'].values
    values = ts_data[sdg_col].values
    ts_series = pd.Series(values, index=years)

    model = fit_arima_model(ts_series)
    if model is None:
        return None

    current_year = int(years[-1])
    years_to_predict = 2030 - current_year
    if years_to_predict <= 0:
        return None

    try:
        forecast_result = model.get_forecast(steps=years_to_predict)
        forecast_values = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()

        lower_ci = confidence_intervals.iloc[:, 0].values
        upper_ci = confidence_intervals.iloc[:, 1].values

        forecast_values = forecast_values.values if hasattr(forecast_values, 'values') else np.array(forecast_values)
        forecast_years = np.arange(current_year + 1, 2031)

        start_value = float(values[0])
        current_value = float(values[-1])
        end_value = float(forecast_values[-1])  # Predicted 2030

        progress_rate = (end_value - start_value) / len(values)

        # Determine color based on prediction vs last real value
        color = determine_trend_color(current_value, end_value)

        return {
            'country': country,
            'sdg': sdg_col,
            'start_value': start_value,
            'current_value': current_value,
            'predicted_2030': end_value,
            'predicted_2030_lower': float(lower_ci[-1]),
            'predicted_2030_upper': float(upper_ci[-1]),
            'progress_rate': progress_rate,
            'color': color,
            'historical_years': years,
            'historical_values': values,
            'forecast_years': forecast_years,
            'forecast_values': forecast_values,
            'forecast_lower': lower_ci,
            'forecast_upper': upper_ci,
            'model_aic': model.aic if model else None
        }

    except Exception as e:
        return None



def create_enhanced_chart(pred):
    """Create enhanced visualization with better styling"""
    fig = go.Figure()
    
    color_map = {
        'green': '#28a745',
        'yellow': '#ffc107', 
        'orange': '#fd7e14',
        'red': '#dc3545'
    }
    
    forecast_color = color_map.get(pred['color'], pred['color'])
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([pred['forecast_years'], pred['forecast_years'][::-1]]),
        y=np.concatenate([pred['forecast_upper'], pred['forecast_lower'][::-1]]),
        fill='toself',
        fillcolor=f"rgba{tuple(list(bytes.fromhex(forecast_color[1:])) + [0.2])}",
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=pred['historical_years'],
        y=pred['historical_values'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#2a5298', width=3),
        marker=dict(size=8, color='#2a5298'),
        hovertemplate='<b>Year:</b> %{x}<br><b>Score:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=pred['forecast_years'],
        y=pred['forecast_values'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=forecast_color, width=3, dash='dash'),
        marker=dict(size=8, color=forecast_color),
        hovertemplate='<b>Year:</b> %{x}<br><b>Predicted Score:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Connection line
    fig.add_trace(go.Scatter(
        x=[pred['historical_years'][-1], pred['forecast_years'][0]],
        y=[pred['historical_values'][-1], pred['forecast_values'][0]],
        mode='lines',
        line=dict(color='#666666', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Target line
    fig.add_hline(y=100, line_dash="dot", line_color="#e74c3c", line_width=3,
                  annotation_text="2030 Target (100)", annotation_position="top right")
    
    # Add annotations for key points
    fig.add_annotation(
        x=pred['forecast_years'][-1],
        y=pred['predicted_2030'],
        text=f"2030: {pred['predicted_2030']:.1f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=forecast_color,
        bgcolor="white",
        bordercolor=forecast_color,
        borderwidth=2
    )

    fig.update_layout(
        title=dict(
            text=f"{pred['country']} - {SDG_MAPPING.get(pred['sdg'], pred['sdg'])}",
            font=dict(size=24, color="#2c3e50"),
            x=0.5
        ),
        xaxis_title="Year", 
        yaxis_title="SDG Score (0-100)",
        xaxis=dict(
            tickmode='linear',
            dtick=2,
            range=[pred['historical_years'][0] - 1, 2031],
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            range=[0, 110],
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_progress_bar(current, target=100):
    """Create a visual progress bar"""
    percentage = min(current / target * 100, 100)
    color = '#28a745' if percentage >= 80 else '#ffc107' if percentage >= 60 else '#fd7e14' if percentage >= 40 else '#dc3545'
    
    return f"""
    <div style="background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden;">
        <div style="height: 100%; border-radius: 10px; width: {percentage}%; background-color: {color}; transition: width 0.3s ease;"></div>
    </div>
    <div style="text-align: center; margin-top: 5px; font-size: 12px; color: #666;">
        {current:.1f} / {target} ({percentage:.1f}%)
    </div>
    """

def create_scorecard(prediction):
    """Create a comprehensive scorecard"""
    status = get_status_info(prediction['color'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### Current Score")
        st.metric("Score", f"{prediction['current_value']:.1f}", help="Current SDG performance score")
        # Progress bar without HTML
        progress = min(prediction['current_value'] / 100, 1.0)
        st.progress(progress)
        st.caption(f"{prediction['current_value']:.1f} / 100 ({prediction['current_value']:.1f}%)")
    
    with col2:
        change = prediction['predicted_2030'] - prediction['current_value']
        st.markdown("#### 2030 Forecast")
        st.metric("Predicted Score", f"{prediction['predicted_2030']:.1f}", 
                 delta=f"{change:.1f}", help="Forecasted score for 2030")
        st.caption(f"Range: {prediction['predicted_2030_lower']:.1f} - {prediction['predicted_2030_upper']:.1f}")
    
    with col3:
        st.markdown("#### Progress Rate")
        st.metric("Annual Change", f"{prediction['progress_rate']:.2f}", 
                 help="Average points change per year")
        st.caption("points per year")
    
    with col4:
        st.markdown("#### Status")
        st.metric("Assessment", status['label'], help=status['description'])
        
        # Color indicator
        color_indicators = {
            'green': '🟢',
            'yellow': '🟡', 
            'orange': '🟠',
            'red': '🔴'
        }
        st.caption(f"{color_indicators.get(prediction['color'], '⚪')} {status['label']}")

def create_insights_panel(prediction):
    """Create detailed insights panel"""
    status = get_status_info(prediction['color'])
    
    # Calculate additional metrics
    years_remaining = 2030 - prediction['historical_years'][-1]
    required_growth = (100 - prediction['current_value']) / years_remaining if years_remaining > 0 else 0
    confidence_range = (prediction['predicted_2030_upper'] - prediction['predicted_2030_lower']) / 2
    
    # Determine if target is achievable
    target_achievable = "Yes" if prediction['predicted_2030'] >= 90 else "Unlikely" if prediction['predicted_2030'] >= 70 else "No"
    
    st.markdown("### Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Situation")
        st.write(status['description'])
        
        st.markdown("#### Target Analysis")
        st.write(f"**Target Achievable by 2030:** {target_achievable}")
        st.write(f"**Gap to Target:** {max(0, 100 - prediction['current_value']):.1f} points")
        
    with col2:
        st.markdown("#### Recommended Action")
        st.write(status['recommendation'])
        
        st.markdown("#### Key Metrics")
        st.write(f"**Years Remaining:** {years_remaining}")
        st.write(f"**Required Annual Growth:** {required_growth:.2f} points/year")
        st.write(f"**Forecast Uncertainty:** ±{confidence_range:.1f} points")
    
    # Additional insights based on trend
    if prediction['progress_rate'] > 0:
        trend_insight = f"The country is making positive progress at {prediction['progress_rate']:.2f} points per year."
    elif prediction['progress_rate'] < -0.5:
        trend_insight = f"The country is experiencing decline at {abs(prediction['progress_rate']):.2f} points per year."
    else:
        trend_insight = "Progress has been largely stagnant with minimal change over time."
    
    st.info(f"**Trend Analysis:** {trend_insight}")
    
    # Performance assessment
    if prediction['predicted_2030'] >= 100:
        performance_msg = "Excellent! The country is on track to exceed the 2030 target."
    elif prediction['predicted_2030'] >= 90:
        performance_msg = "Good progress! The country is close to achieving the 2030 target."
    elif prediction['predicted_2030'] >= 70:
        performance_msg = "Moderate progress. Acceleration needed to meet the 2030 target."
    else:
        performance_msg = "Significant challenges ahead. Major policy interventions required."
    
    if prediction['predicted_2030'] >= 90:
        st.success(performance_msg)
    elif prediction['predicted_2030'] >= 70:
        st.warning(performance_msg)
    else:
        st.error(performance_msg)

def main():
    df = load_data()
    if df is None:
        return

    sdg_cols = [col for col in df.columns if col.startswith('goal')]
    if not sdg_cols:
        st.error("No SDG columns found in the dataset.")
        return

    if 'Country' not in df.columns or 'year' not in df.columns:
        st.error("Required columns 'Country' and 'year' not found.")
        return

    countries = sorted(df['Country'].unique())

    # Sidebar for navigation and information
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("Select Analysis Type", ["Single Country Analysis", "Multi-Country Comparison", "About SDGs"])
        
        if page != "About SDGs":
            st.markdown("### Quick Stats")
            st.info(f"**Countries Available:** {len(countries)}")
            st.info(f"**SDGs Tracked:** {len(sdg_cols)}")
            st.info(f"**Data Period:** {df['year'].min()} - {df['year'].max()}")

# Fixed About SDGs section - replace the existing one with this

    if page == "About SDGs":
        col1, col2 = st.columns([1, 4])
        with col1:
            try:
                un_logo = Image.open("icons/un_sdg.png")
                st.image(un_logo, width=100)
            except FileNotFoundError:
                pass
        with col2:
            st.markdown("""
            <h1 style="margin-bottom:0">Sustainable Development Goals</h1>
            <p style="color:#666;margin-top:0">The 17 Global Goals for 2030</p>
            """, unsafe_allow_html=True)
        
        # Load JSON data
        sdg_data = load_sdg_json_data()
        if sdg_data is None:
            st.stop()
        
        # Create tabs for each SDG
        tabs = st.tabs([f"SDG {i}" for i in range(1, 18)])
        
        for i, tab in enumerate(tabs, 1):
            with tab:
                # Find the SDG in the JSON data
                sdg_info = next((item for item in sdg_data["sdgs"] if item["sdg_number"] == i), None)
                if not sdg_info:
                    st.warning(f"No data found for SDG {i}")
                    continue
                
                # Two-column layout for icon and basic info
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    try:
                        icon_path = f"icons/E_PRINT_{i}.jpg"
                        icon = Image.open(icon_path)
                        st.image(icon, use_container_width=True)
                    except FileNotFoundError:
                        st.warning(f"Icon for SDG {i} not found")
                
                with col2:
                    # Status with colored indicator
                    progress = sdg_info['data']['progress']
                    status_color = {
                        "Significant challenges": "#dc3545",
                        "Challenges remain": "#fd7e14",
                        "Moderate progress": "#ffc107",
                        "On track": "#28a745"
                    }.get(progress['overall_status'], "#6c757d")
                    
                    st.markdown(f"""
                    <h2 style="margin-top:0">{sdg_info['sdg_name']}</h2>
                    <div style="display:flex;align-items:center;margin-bottom:15px">
                        <div style="width:12px;height:12px;background-color:{status_color};border-radius:50%;margin-right:8px"></div>
                        <span><b>Status:</b> {progress['overall_status']} | <b>Global Progress:</b> {progress['global_progress_percentage']}</span>
                    </div>
                    <p style="font-size:16px">{SDG_DESCRIPTIONS[f'goal{i}'].split(' - ')[1]}</p>
                    """, unsafe_allow_html=True)
                
                # Horizontal divider
                st.markdown("---")
                
                # Targets section
                st.markdown("### Targets")
                for target in sdg_info['data'].get('targets', []):
                    # Fixed: Use st.expander instead of custom HTML
                    with st.expander(f"Target {target['target_number']}"):
                        st.write(target['description'])
                        st.write("**Key Indicators:**")
                        for indicator in target.get('key_indicators', []):
                            st.write(f"• {indicator}")
                
                # Progress and statistics
                st.markdown("### Current Progress")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Key Statistics")
                    for stat in progress.get('key_statistics', []):
                        st.write(f"• {stat}")
                
                with col2:
                    st.markdown("#### Regional Performance")
                    regions_on_track = ", ".join(progress.get('regions_on_track', []))
                    regions_lagging = ", ".join(progress.get('regions_lagging', []))
                    
                    if regions_on_track:
                        st.write(f"**Regions On Track:** {regions_on_track}")
                    else:
                        st.write("**Regions On Track:** None")
                    
                    if regions_lagging:
                        st.write(f"**Regions Lagging:** {regions_lagging}")
                    else:
                        st.write("**Regions Lagging:** Global")
                
                # Challenges section
                if sdg_info['data'].get('challenges'):
                    st.markdown("### Key Challenges")
                    for challenge in sdg_info['data']['challenges']:
                        with st.expander(challenge['challenge']):
                            st.write(f"**Impact:** {challenge['impact']}")
                            st.write(f"**Affected Regions:** {', '.join(challenge['affected_regions'])}")
                
                # Success stories
                if sdg_info['data'].get('success_stories'):
                    st.markdown("### Success Stories")
                    for story in sdg_info['data']['success_stories']:
                        with st.expander(f"{story['country']} ({story['year']})"):
                            st.write(f"**Initiative:** {story['initiative']}")
                            st.write(f"**Impact:** {story['impact']}")
                
                # Priority actions
                if sdg_info['data'].get('priority_actions'):
                    st.markdown("### Priority Actions")
                    st.write("**Recommended Actions:**")
                    for action in sdg_info['data']['priority_actions']:
                        st.write(f"• {action}")
                
                # Interconnections with other SDGs
                if sdg_info['data'].get('interconnections'):
                    st.markdown("### Connections to Other SDGs")
                    for link in sdg_info['data']['interconnections']:
                        with st.expander(f"Connection with {link['related_sdg']}"):
                            st.write(link['relationship'])

    elif page == "Single Country Analysis":
        st.markdown("## Single Country SDG Analysis")
        st.markdown("Select a country and SDG to view detailed forecasting analysis with actionable insights.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_country = st.selectbox("Select Country", countries)
        with col2:
            selected_sdg = st.selectbox(
                "Select SDG",
                sdg_cols,
                format_func=lambda x: f"SDG {x.replace('goal', '')}: {SDG_DESCRIPTIONS[x].split(' - ')[0]}"
            )

        if st.button("Generate Analysis", type="primary"):
            with st.spinner("Generating forecast analysis..."):
                prediction = predict_sdg_progress(df, selected_country, selected_sdg)

                if prediction:
                    # SDG Description
                    st.markdown(f"### {SDG_DESCRIPTIONS[selected_sdg]}")
                    
                    # Scorecard
                    create_scorecard(prediction)
                    
                    # Chart
                    st.markdown("### Forecast Visualization")
                    fig = create_enhanced_chart(prediction)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    create_insights_panel(prediction)
                    
                    # Technical Details
                    with st.expander("Technical Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Data Points Used", len(prediction['historical_years']))
                        with col2:
                            st.metric("Model AIC Score", f"{prediction['model_aic']:.2f}" if prediction['model_aic'] else "N/A")
                        with col3:
                            st.metric("Forecast Horizon", f"{len(prediction['forecast_years'])} years")
                
                else:
                    st.error("Unable to generate forecast. Insufficient data or modeling failed.")

    elif page == "Multi-Country Comparison":
        st.markdown("## Multi-Country SDG Comparison")
        st.markdown("Compare multiple countries on a single SDG to identify best practices and regional patterns.")
        
        selected_sdg = st.selectbox(
            "Select SDG for Comparison",
            sdg_cols,
            format_func=lambda x: f"SDG {x.replace('goal', '')}: {SDG_DESCRIPTIONS[x].split(' - ')[0]}"
        )
        
        selected_countries = st.multiselect(
            "Select Countries to Compare", 
            countries, 
            default=countries[:3] if len(countries) >= 3 else countries
        )

        if len(selected_countries) >= 2:
            st.markdown(f"### {SDG_DESCRIPTIONS[selected_sdg]}")
            
            # Create comparison chart
            fig = go.Figure()
            
            color_palette = ['#2a5298', '#e74c3c', '#28a745', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', "#c8cc4b", "#584a80"]
            
            comparison_data = []
            
            for i, country in enumerate(selected_countries):
                color = color_palette[i % len(color_palette)]
                pred = predict_sdg_progress(df, country, selected_sdg)
                
                if pred:
                    comparison_data.append({
                        'Country': country,
                        'Current Score': pred['current_value'],
                        '2030 Forecast': pred['predicted_2030'],
                        'Progress Rate': pred['progress_rate'],
                        'Status': get_status_info(pred['color'])['label']
                    })
                    
                    # Confidence interval
                    r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                    fillcolor = f"rgba({r}, {g}, {b}, 0.2)"

                    fig.add_trace(go.Scatter(
                        x=np.concatenate([pred['forecast_years'], pred['forecast_years'][::-1]]),
                        y=np.concatenate([pred['forecast_upper'], pred['forecast_lower'][::-1]]),
                        fill='toself',
                        fillcolor=fillcolor,
                        line=dict(color='rgba(255, 255, 255,0)'),
                        name=f"{country} Confidence Interval",
                        showlegend=False
                    ))

                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=pred['historical_years'],
                        y=pred['historical_values'],
                        mode='lines+markers',
                        name=f"{country} (Historical)",
                        line=dict(color=color, width=3),
                        marker=dict(size=8)
                    ))

                    # Forecast data
                    fig.add_trace(go.Scatter(
                        x=pred['forecast_years'],
                        y=pred['forecast_values'],
                        mode='lines+markers',
                        name=f"{country} (Forecast)",
                        line=dict(color=color, dash='dash', width=3),
                        marker=dict(size=8)
                    ))

            fig.add_hline(y=100, line_dash="dot", line_color="#e74c3c", line_width=3,
                          annotation_text="2030 Target (100)", annotation_position="top right")

            fig.update_layout(
                title=dict(
                    text=f"SDG {selected_sdg.replace('goal', '')} Comparison",
                    font=dict(size=24, color="#2c3e50"),
                    x=0.5
                ),
                height=650,
                xaxis_title="Year",
                yaxis_title="SDG Score (0-100)",
                xaxis=dict(tickmode='linear', dtick=2, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.1)'),
                hovermode='x unified',
                legend=dict(
                    orientation="h", 
                    y=-0.2, 
                    x=0.5, 
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0.9)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            if comparison_data:
                st.markdown("### Comparison Summary")
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
        else:
            st.info("Please select at least 2 countries for comparison.")

if __name__ == "__main__":
    main()
    st.stop()  # This ensures nothing after it runs