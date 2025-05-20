README.md file of the sales predictor application

# Pharmaceutical Sales Predictor

A powerful web application built with Streamlit that provides sales forecasting and analysis for pharmaceutical products. This dashboard helps pharmaceutical companies make data-driven decisions by visualizing historical sales data and predicting future trends.

## Features

- 📊 Interactive sales visualization and forecasting
- 📈 30-day sales predictions using Facebook Prophet
- 🔍 Detailed seasonality analysis
- 📱 Responsive and user-friendly interface
- 📥 Export functionality for forecasts
- 📈 Key metrics and insights
- 🎯 Product-specific analysis

## Installation

1. Clone the repository:
``` Go to your bash/terminal
git clone [your-repository-url]
cd Pharmaceutical-Sales-Predictor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Predicting_Sales/app.py
```

## Dependencies

- Streamlit
- Pandas
- Prophet
- Plotly
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```
Pharmaceutical-Sales-Predictor/
├── Predicting_Sales/
│   ├── app.py
│   ├── data/
│   │   └── fake_sales_data.csv
│   └── models/
│       └── prophet_*.pkl
├── requirements.txt
└── README.md
```

## Usage

1. Launch the application using `streamlit run Predicting_Sales/app.py`
2. Select a product from the sidebar
3. Choose your desired date range
4. View historical sales data and forecasts
5. Analyze seasonality patterns
6. Download forecast data if needed

## Features in Detail

### Sales Visualization
- Interactive line charts showing historical sales data
- Customizable date range selection
- Real-time data filtering

### Forecasting
- 30-day sales predictions
- Confidence intervals for forecasts
- Trend analysis
- Seasonality decomposition

### Analytics
- Key performance metrics
- Average daily sales
- Highest daily sales
- Total sales in selected period

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or suggestions, please open an issue in the repository. 
