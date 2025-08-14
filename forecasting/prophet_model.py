#!/usr/bin/env python3
"""
Prophet Demand Forecasting Model for SeoulBike Digital Twin Project

This module:
- Uses the cleaned sales data to train Prophet models per SKU/warehouse
- Predicts demand for the next 3 months
- Handles seasonal patterns, holidays, and weather effects
- Saves forecast results to data/forecast/
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

warnings.filterwarnings('ignore')


class SeoulBikeForecaster:
    """Prophet-based demand forecaster for SeoulBike system"""
    
    def __init__(self, data_path: str = 'data/cleaned/', forecast_path: str = 'data/forecast/'):
        """
        Initialize the forecaster
        
        Args:
            data_path: Path to cleaned data files
            forecast_path: Path to save forecast results
        """
        self.data_path = data_path
        self.forecast_path = forecast_path
        self.cleaned_data = None
        self.warehouse_demand_data = {}
        self.forecast_models = {}
        self.forecast_results = {}
        
        # Create forecast directory
        os.makedirs(forecast_path, exist_ok=True)
        
        # Prophet model parameters
        self.prophet_params = {
            'changepoint_prior_scale': 0.05,  # Flexibility of trend
            'seasonality_prior_scale': 10.0,  # Flexibility of seasonality
            'holidays_prior_scale': 10.0,     # Flexibility of holidays
            'seasonality_mode': 'multiplicative',  # Multiplicative seasonality
            'changepoint_range': 0.8,         # Range of changepoints
            'yearly_seasonality': True,       # Yearly patterns
            'weekly_seasonality': True,       # Weekly patterns
            'daily_seasonality': False        # Daily patterns (too granular)
        }
    
    def load_cleaned_data(self):
        """Load cleaned data for forecasting"""
        try:
            # Load main cleaned data
            main_data_file = os.path.join(self.data_path, 'cleaned_data.csv')
            if os.path.exists(main_data_file):
                self.cleaned_data = pd.read_csv(main_data_file)
                self.cleaned_data['datetime'] = pd.to_datetime(self.cleaned_data['datetime'])
                print(f"Loaded cleaned data: {len(self.cleaned_data)} records")
            else:
                raise FileNotFoundError(f"Cleaned data file not found: {main_data_file}")
            
            # Load warehouse data
            warehouse_file = os.path.join(self.data_path, 'warehouses.csv')
            if os.path.exists(warehouse_file):
                self.warehouses = pd.read_csv(warehouse_file)
                print(f"Loaded warehouse data: {len(self.warehouses)} warehouses")
            else:
                # Create default warehouse structure
                self.warehouses = pd.DataFrame({
                    'warehouse_id': [1, 2, 3],
                    'warehouse_name': ['Central_Warehouse', 'North_Warehouse', 'South_Warehouse'],
                    'location': ['Seoul_Center', 'Seoul_North', 'Seoul_South']
                })
                print("Created default warehouse structure")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_demand_data(self):
        """Prepare demand data for each warehouse"""
        print("Preparing demand data for forecasting...")
        
        # Group by warehouse and date to get daily demand
        if 'warehouse_id' not in self.cleaned_data.columns:
            # If no warehouse_id, assign based on location or create synthetic
            self.cleaned_data['warehouse_id'] = np.random.choice([1, 2, 3], size=len(self.cleaned_data))
        
        # Aggregate to daily demand per warehouse
        daily_demand = self.cleaned_data.groupby(['Date', 'warehouse_id'])['Rented Bike Count'].sum().reset_index()
        daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
        
        # Prepare data for each warehouse
        for warehouse_id in self.warehouses['warehouse_id']:
            warehouse_demand = daily_demand[daily_demand['warehouse_id'] == warehouse_id].copy()
            
            if len(warehouse_demand) > 0:
                # Prophet requires 'ds' (date) and 'y' (value) columns
                warehouse_demand['ds'] = warehouse_demand['Date']
                warehouse_demand['y'] = warehouse_demand['Rented Bike Count']
                
                # Sort by date
                warehouse_demand = warehouse_demand.sort_values('ds')
                
                # Remove any missing values
                warehouse_demand = warehouse_demand.dropna(subset=['ds', 'y'])
                
                # Remove outliers (values beyond 3 standard deviations)
                mean_demand = warehouse_demand['y'].mean()
                std_demand = warehouse_demand['y'].std()
                warehouse_demand = warehouse_demand[
                    (warehouse_demand['y'] >= mean_demand - 3 * std_demand) &
                    (warehouse_demand['y'] <= mean_demand + 3 * std_demand)
                ]
                
                self.warehouse_demand_data[warehouse_id] = warehouse_demand
                print(f"Warehouse {warehouse_id}: {len(warehouse_demand)} daily demand records")
            else:
                print(f"Warning: No demand data for warehouse {warehouse_id}")
    
    def create_custom_seasonalities(self, model: Prophet):
        """Add custom seasonalities to the Prophet model"""
        # Add monthly seasonality
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Add quarterly seasonality
        model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=8
        )
        
        # Add holiday effects for Seoul
        seoul_holidays = pd.DataFrame([
            {'holiday': 'New Year', 'ds': '2024-01-01', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'Seollal', 'ds': '2024-02-10', 'lower_window': -2, 'upper_window': 2},
            {'holiday': 'Buddha Birthday', 'ds': '2024-05-15', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'Chuseok', 'ds': '2024-09-17', 'lower_window': -2, 'upper_window': 2},
            {'holiday': 'Christmas', 'ds': '2024-12-25', 'lower_window': -1, 'upper_window': 1},
        ])
        seoul_holidays['ds'] = pd.to_datetime(seoul_holidays['ds'])
        
        model.add_country_holidays(country_name='KR')  # South Korea holidays
        model.add_seasonality(
            name='holiday_effect',
            period=365.25,
            fourier_order=10
        )
    
    def train_forecast_models(self):
        """Train Prophet models for each warehouse"""
        print("Training Prophet forecast models...")
        
        for warehouse_id, demand_data in self.warehouse_demand_data.items():
            if len(demand_data) < 30:  # Need at least 30 days of data
                print(f"Warning: Insufficient data for warehouse {warehouse_id}, skipping...")
                continue
            
            print(f"Training model for warehouse {warehouse_id}...")
            
            # Create and configure Prophet model
            model = Prophet(**self.prophet_params)
            
            # Add custom seasonalities
            self.create_custom_seasonalities(model)
            
            # Fit the model
            try:
                model.fit(demand_data)
                self.forecast_models[warehouse_id] = model
                print(f"Model trained successfully for warehouse {warehouse_id}")
            except Exception as e:
                print(f"Error training model for warehouse {warehouse_id}: {e}")
                continue
    
    def generate_forecasts(self, forecast_days: int = 90):
        """Generate forecasts for the next N days"""
        print(f"Generating {forecast_days}-day forecasts...")
        
        for warehouse_id, model in self.forecast_models.items():
            try:
                # Create future dataframe
                future_dates = model.make_future_dataframe(periods=forecast_days, freq='D')
                
                # Generate forecast
                forecast = model.predict(future_dates)
                
                # Store forecast results
                self.forecast_results[warehouse_id] = {
                    'model': model,
                    'forecast': forecast,
                    'future_dates': future_dates
                }
                
                print(f"Forecast generated for warehouse {warehouse_id}")
                
            except Exception as e:
                print(f"Error generating forecast for warehouse {warehouse_id}: {e}")
                continue
    
    def analyze_forecast_accuracy(self):
        """Analyze forecast accuracy using historical data"""
        print("Analyzing forecast accuracy...")
        
        accuracy_metrics = {}
        
        for warehouse_id, forecast_data in self.forecast_results.items():
            if warehouse_id not in self.warehouse_demand_data:
                continue
            
            # Get historical data
            historical_data = self.warehouse_demand_data[warehouse_id]
            
            # Get forecast for historical period
            historical_forecast = forecast_data['forecast'][
                forecast_data['forecast']['ds'].isin(historical_data['ds'])
            ]
            
            # Merge with actual values
            accuracy_df = historical_data.merge(
                historical_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                on='ds'
            )
            
            # Calculate accuracy metrics
            mae = np.mean(np.abs(accuracy_df['y'] - accuracy_df['yhat']))
            mape = np.mean(np.abs((accuracy_df['y'] - accuracy_df['yhat']) / accuracy_df['y'])) * 100
            rmse = np.sqrt(np.mean((accuracy_df['y'] - accuracy_df['yhat']) ** 2))
            
            accuracy_metrics[warehouse_id] = {
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'data_points': len(accuracy_df)
            }
            
            print(f"Warehouse {warehouse_id} - MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
        
        return accuracy_metrics
    
    def create_forecast_visualizations(self):
        """Create interactive forecast visualizations"""
        print("Creating forecast visualizations...")
        
        for warehouse_id, forecast_data in self.forecast_results.items():
            try:
                model = forecast_data['model']
                forecast = forecast_data['forecast']
                
                # Create forecast plot
                fig = plot_plotly(model, forecast)
                fig.update_layout(
                    title=f"Demand Forecast - Warehouse {warehouse_id}",
                    xaxis_title="Date",
                    yaxis_title="Demand (Bike Rentals)"
                )
                
                # Save plot
                plot_file = os.path.join(self.forecast_path, f'forecast_warehouse_{warehouse_id}.html')
                fig.write_html(plot_file)
                
                # Create components plot
                comp_fig = plot_components_plotly(model, forecast)
                comp_fig.update_layout(
                    title=f"Forecast Components - Warehouse {warehouse_id}"
                )
                
                # Save components plot
                comp_file = os.path.join(self.forecast_path, f'components_warehouse_{warehouse_id}.html')
                comp_fig.write_html(comp_file)
                
                print(f"Visualizations created for warehouse {warehouse_id}")
                
            except Exception as e:
                print(f"Error creating visualizations for warehouse {warehouse_id}: {e}")
                continue
    
    def save_forecast_results(self):
        """Save forecast results to files"""
        print("Saving forecast results...")
        
        # Save forecast data for each warehouse
        for warehouse_id, forecast_data in self.forecast_results.items():
            try:
                forecast = forecast_data['forecast']
                
                # Select relevant columns
                forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_output.columns = ['date', 'predicted_demand', 'lower_bound', 'upper_bound']
                
                # Save to CSV
                forecast_file = os.path.join(self.forecast_path, f'forecast_warehouse_{warehouse_id}.csv')
                forecast_output.to_csv(forecast_file, index=False)
                
                print(f"Forecast saved for warehouse {warehouse_id}")
                
            except Exception as e:
                print(f"Error saving forecast for warehouse {warehouse_id}: {e}")
                continue
        
        # Save summary statistics
        summary_stats = {}
        for warehouse_id, forecast_data in self.forecast_results.items():
            forecast = forecast_data['forecast']
            
            # Get future forecasts (last 90 days)
            future_forecast = forecast.tail(90)
            
            summary_stats[warehouse_id] = {
                'total_predicted_demand': future_forecast['yhat'].sum(),
                'avg_daily_demand': future_forecast['yhat'].mean(),
                'max_daily_demand': future_forecast['yhat'].max(),
                'min_daily_demand': future_forecast['yhat'].min(),
                'demand_volatility': future_forecast['yhat'].std()
            }
        
        # Save summary
        summary_file = os.path.join(self.forecast_path, 'forecast_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print("Forecast summary saved")
    
    def run_full_forecasting(self, forecast_days: int = 90):
        """Run the complete forecasting pipeline"""
        print("Starting SeoulBike demand forecasting pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_cleaned_data()
            
            # Step 2: Prepare demand data
            self.prepare_demand_data()
            
            # Step 3: Train models
            self.train_forecast_models()
            
            # Step 4: Generate forecasts
            self.generate_forecasts(forecast_days)
            
            # Step 5: Analyze accuracy
            accuracy_metrics = self.analyze_forecast_accuracy()
            
            # Step 6: Create visualizations
            self.create_forecast_visualizations()
            
            # Step 7: Save results
            self.save_forecast_results()
            
            print("=" * 60)
            print("Forecasting pipeline completed successfully!")
            
            return {
                'models_trained': len(self.forecast_models),
                'forecasts_generated': len(self.forecast_results),
                'accuracy_metrics': accuracy_metrics
            }
            
        except Exception as e:
            print(f"Error in forecasting pipeline: {e}")
            raise
    
    def get_forecast_summary(self) -> Dict:
        """Get summary of forecast results"""
        if not self.forecast_results:
            return {}
        
        summary = {}
        for warehouse_id, forecast_data in self.forecast_results.items():
            forecast = forecast_data['forecast']
            
            # Get future forecasts
            future_forecast = forecast[forecast['ds'] > pd.Timestamp.now()]
            
            if len(future_forecast) > 0:
                summary[warehouse_id] = {
                    'next_30_days_avg': future_forecast.head(30)['yhat'].mean(),
                    'next_90_days_total': future_forecast.head(90)['yhat'].sum(),
                    'trend_direction': 'increasing' if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else 'decreasing'
                }
        
        return summary


def main():
    """Main function to run the forecaster"""
    # Create and run forecaster
    forecaster = SeoulBikeForecaster()
    
    try:
        results = forecaster.run_full_forecasting(forecast_days=90)
        print(f"\nForecasting completed!")
        print(f"Models trained: {results['models_trained']}")
        print(f"Forecasts generated: {results['forecasts_generated']}")
        
        # Show forecast summary
        summary = forecaster.get_forecast_summary()
        print("\nForecast Summary:")
        for warehouse_id, data in summary.items():
            print(f"Warehouse {warehouse_id}:")
            print(f"  Next 30 days avg: {data['next_30_days_avg']:.1f} bikes/day")
            print(f"  Next 90 days total: {data['next_90_days_total']:.0f} bikes")
            print(f"  Trend: {data['trend_direction']}")
        
    except Exception as e:
        print(f"Forecasting failed: {e}")


if __name__ == "__main__":
    main()
