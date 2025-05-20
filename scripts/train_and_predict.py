import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
from pathlib import Path

# Create necessary directories if they don't exist
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv('data/fake_sales_data.csv')

# Function to train model for a product
def train_product_model(product_name):
    print(f"\nTraining model for {product_name}...")
    
    # Filter data for the product
    df_product = df[df['Product'] == product_name].copy()
    
    # Prepare data for Prophet
    df_product.rename(columns={"Date": "ds", "Quantity": "y"}, inplace=True)
    df_product['ds'] = pd.to_datetime(df_product['ds'])
    
    # Train Prophet model
    model = Prophet(yearly_seasonality=True, 
                   weekly_seasonality=True,
                   daily_seasonality=True)
    model.fit(df_product)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Predict next 30 days
    forecast = model.predict(future)
    
    # Save model
    model_path = f'models/prophet_{product_name.lower()}.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df_product['ds'], df_product['y'], label='Actual Sales')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sales', color='red')
    plt.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'], 
                     color='red', alpha=0.1)
    plt.title(f'Sales Prediction for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    
    # Save plot
    plot_path = f'output/{product_name.lower()}_prediction.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    return model, forecast

# Train models for each product
products = df['Product'].unique()
for product in products:
    model, forecast = train_product_model(product)
    
print("\nTraining and prediction completed!") 