import joblib
import pandas as pd
from datetime import datetime, timedelta

# Load the saved model
print("Loading saved model...")
model = joblib.load('models/prophet_paracetamol.pkl')

# Create future dates for prediction
future_dates = pd.DataFrame({
    'ds': pd.date_range(
        start=datetime.now(),
        periods=7,  # Predict next 7 days
        freq='D'
    )
})

# Make predictions
print("\nMaking predictions for next 7 days...")
forecast = model.predict(future_dates)

# Display predictions
print("\nPredicted Sales for Paracetamol:")
print("--------------------------------")
for date, prediction in zip(forecast['ds'], forecast['yhat']):
    print(f"{date.strftime('%Y-%m-%d')}: {prediction:.2f} units")

# Show prediction intervals
print("\nPrediction Intervals (80% confidence):")
print("--------------------------------------")
for date, lower, upper in zip(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper']):
    print(f"{date.strftime('%Y-%m-%d')}: {lower:.2f} - {upper:.2f} units") 