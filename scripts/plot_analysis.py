import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import joblib
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv('data/fake_sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_historical_sales(df):
    plt.figure(figsize=(15, 6))
    
    # Plot for each product
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product]
        plt.plot(product_data['Date'], product_data['Quantity'], 
                label=product, marker='o', markersize=4)
    
    plt.title('Historical Sales by Product', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Quantity Sold', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('output/historical_sales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Historical sales plot saved to output/historical_sales.png")

def plot_predictions(product_name):
    # Load the model
    model = joblib.load(f'models/prophet_{product_name.lower()}.pkl')
    
    # Create future dates for prediction
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(model.history['ds'], model.history['y'], 
            label='Historical Data', color='blue', alpha=0.6)
    
    # Plot predictions
    plt.plot(forecast['ds'], forecast['yhat'], 
            label='Predictions', color='red', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='red', alpha=0.2, 
                    label='Confidence Interval')
    
    plt.title(f'Sales Predictions for {product_name}', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Quantity Sold', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'output/{product_name.lower()}_detailed_prediction.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed prediction plot saved for {product_name}")

def plot_seasonality(product_name):
    # Load the model
    model = joblib.load(f'models/prophet_{product_name.lower()}.pkl')
    
    # Create future dates for prediction
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # Plot components
    fig = model.plot_components(forecast)
    plt.suptitle(f'Seasonality Analysis for {product_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'output/{product_name.lower()}_seasonality.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Seasonality plot saved for {product_name}")

def main():
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot historical sales
    plot_historical_sales(df)
    
    # Plot predictions and seasonality for each product
    for product in df['Product'].unique():
        print(f"\nAnalyzing {product}...")
        plot_predictions(product)
        plot_seasonality(product)
    
    print("\nAll plots have been generated in the 'output' directory!")

if __name__ == "__main__":
    main() 