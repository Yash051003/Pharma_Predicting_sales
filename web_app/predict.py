import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('/models/prophet_paracetamol.pkl')

# Create future dataframe
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('/output/forecast_results.csv', index=False)

# Plot
fig = model.plot(forecast)
fig.savefig('/output/forecast_plot.png')
