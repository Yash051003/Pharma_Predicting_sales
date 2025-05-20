on github 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Pharmaceutical Sales Predictor",
    page_icon="logo.png",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💊 Pharmaceutical Sales Predictor")
st.markdown("""
    This dashboard provides insights and predictions for pharmaceutical sales.
    Select a product and date range to view detailed forecasts and analysis.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/fake_sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Load model
@st.cache_resource
def load_model(product_name):
    return joblib.load(f'models/prophet_{product_name.lower()}.pkl')

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Settings")

# Product selection
products = df['Product'].unique()
selected_product = st.sidebar.selectbox(
    "Select Product",
    products
)

# Date range selection
min_date = df['Date'].min()
max_date = df['Date'].max()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data based on selection
mask = (df['Product'] == selected_product) & \
       (df['Date'].dt.date >= selected_dates[0]) & \
       (df['Date'].dt.date <= selected_dates[1])
filtered_df = df[mask]

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Sales")
    fig = px.line(filtered_df, x='Date', y='Quantity',
                  title=f'Sales History for {selected_product}')
    st.plotly_chart(fig, use_container_width=True)

# Load model and make predictions
model = load_model(selected_product)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

with col2:
    st.subheader("Sales Forecast")
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=model.history['ds'],
        y=model.history['y'],
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Predictions',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(255,0,0,0.2)',
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,0,0,0.2)',
        name='Lower Bound'
    ))
    
    fig.update_layout(
        title=f'30-Day Forecast for {selected_product}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    avg_sales = filtered_df['Quantity'].mean()
    st.metric("Average Daily Sales", f"{avg_sales:.1f} units")

with col2:
    max_sales = filtered_df['Quantity'].max()
    st.metric("Highest Daily Sales", f"{max_sales:.1f} units")

with col3:
    total_sales = filtered_df['Quantity'].sum()
    st.metric("Total Sales in Period", f"{total_sales:.1f} units")

# Seasonality Analysis
st.subheader("Seasonality Analysis")
fig = model.plot_components(forecast)
st.pyplot(fig)

# Insights
st.subheader("Key Insights")
st.markdown("""
    - **Trend Analysis**: The model shows [trend description]
    - **Seasonality**: [Seasonality patterns]
    - **Forecast Confidence**: The prediction intervals show [confidence level]
    - **Recommendations**: [Action items based on the analysis]
""")

# Add download button for the data
st.download_button(
    label="Download Forecast Data",
    data=forecast.to_csv(index=False).encode('utf-8'),
    file_name=f'{selected_product.lower()}_forecast.csv',
    mime='text/csv'
) 