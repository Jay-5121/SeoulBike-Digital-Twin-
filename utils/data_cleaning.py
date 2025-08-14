#!/usr/bin/env python3
"""
Data Cleaning Script for SeoulBike Digital Twin Project

This script:
1. Reads the SeoulBike dataset with proper encoding
2. Cleans missing values and standardizes data formats
3. Creates relational structures for suppliers → warehouses → customers
4. Saves cleaned data for further processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class SeoulBikeDataCleaner:
    """Data cleaner for SeoulBike dataset"""
    
    def __init__(self, raw_data_path, cleaned_data_path):
        self.raw_data_path = raw_data_path
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        
    def read_data(self):
        """Read the raw CSV data with proper encoding"""
        try:
            # Try different encodings to handle the Unicode issue
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.raw_data_path, encoding=encoding)
                    print(f"Successfully read data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")
                
            print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return False
    
    def clean_dates(self):
        """Standardize date formats and create datetime features"""
        print("Cleaning date formats...")
        
        # Convert Date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Create additional datetime features
        self.df['datetime'] = self.df['Date'] + pd.to_timedelta(self.df['Hour'], unit='h')
        self.df['year'] = self.df['Date'].dt.year
        self.df['month'] = self.df['Date'].dt.month
        self.df['day'] = self.df['Date'].dt.day
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_holiday'] = (self.df['Holiday'] == 'Holiday').astype(int)
        
        # Remove rows with invalid dates
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['Date'])
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with invalid dates")
    
    def clean_numerical_columns(self):
        """Clean and validate numerical columns"""
        print("Cleaning numerical columns...")
        
        # Clean temperature columns (remove extreme outliers)
        temp_cols = ['Temperature(°C)', 'Dew point temperature(°C)']
        for col in temp_cols:
            # Remove extreme temperature values (below -50°C or above 50°C)
            self.df = self.df[(self.df[col] >= -50) & (self.df[col] <= 50)]
        
        # Clean humidity (should be 0-100%)
        self.df = self.df[(self.df['Humidity(%)'] >= 0) & (self.df['Humidity(%)'] <= 100)]
        
        # Clean wind speed (remove negative values)
        self.df = self.df[self.df['Wind speed (m/s)'] >= 0]
        
        # Clean visibility (remove negative values)
        self.df = self.df[self.df['Visibility (10m)'] >= 0]
        
        # Clean rainfall and snowfall (remove negative values)
        self.df = self.df[self.df['Rainfall(mm)'] >= 0]
        self.df = self.df[self.df['Snowfall (cm)'] >= 0]
        
        # Clean solar radiation (remove negative values)
        self.df = self.df[self.df['Solar Radiation (MJ/m2)'] >= 0]
        
        print(f"After cleaning numerical columns: {len(self.df)} rows")
    
    def create_relational_structure(self):
        """Create relational structure for digital twin simulation"""
        print("Creating relational structure...")
        
        # Create supplier entities (representing bike suppliers/manufacturers)
        suppliers = pd.DataFrame({
            'supplier_id': range(1, 6),
            'supplier_name': ['BikeSupplier_A', 'BikeSupplier_B', 'BikeSupplier_C', 'BikeSupplier_D', 'BikeSupplier_E'],
            'location': ['Seoul_Center', 'Seoul_North', 'Seoul_South', 'Seoul_East', 'Seoul_West'],
            'capacity_per_day': [100, 80, 120, 90, 110],
            'reliability_score': [0.95, 0.88, 0.92, 0.85, 0.90]
        })
        
        # Create warehouse entities (representing bike storage locations)
        warehouses = pd.DataFrame({
            'warehouse_id': range(1, 4),
            'warehouse_name': ['Central_Warehouse', 'North_Warehouse', 'South_Warehouse'],
            'location': ['Seoul_Center', 'Seoul_North', 'Seoul_South'],
            'storage_capacity': [500, 400, 450],
            'current_inventory': [300, 250, 280],
            'reorder_point': [100, 80, 90],
            'reorder_quantity': [200, 150, 180],
            'supplier_id': [1, 2, 3]  # Link to suppliers
        })
        
        # Create customer demand patterns based on the bike rental data
        customer_demand = self.df.groupby(['Date', 'Hour', 'Seasons']).agg({
            'Rented Bike Count': 'sum',
            'Temperature(°C)': 'mean',
            'Humidity(%)': 'mean',
            'Wind speed (m/s)': 'mean',
            'Rainfall(mm)': 'sum',
            'Snowfall (cm)': 'sum',
            'is_weekend': 'first',
            'is_holiday': 'first'
        }).reset_index()
        
        # Add warehouse assignment based on demand patterns
        customer_demand['warehouse_id'] = np.random.choice(
            warehouses['warehouse_id'], 
            size=len(customer_demand),
            p=[0.4, 0.3, 0.3]  # Probability distribution for warehouse assignment
        )
        
        # Create transport routes
        transport_routes = pd.DataFrame({
            'route_id': range(1, 8),
            'from_location': ['Seoul_Center', 'Seoul_North', 'Seoul_South', 'Seoul_East', 'Seoul_West', 'Seoul_Center', 'Seoul_North'],
            'to_location': ['Seoul_North', 'Seoul_Center', 'Seoul_Center', 'Seoul_Center', 'Seoul_Center', 'Seoul_South', 'Seoul_South'],
            'distance_km': [15, 15, 20, 18, 22, 20, 25],
            'avg_transport_time_hours': [1.5, 1.5, 2.0, 1.8, 2.2, 2.0, 2.5],
            'cost_per_km': [2.5, 2.5, 2.8, 2.6, 3.0, 2.8, 3.2]
        })
        
        # Save relational data
        self.relational_data = {
            'suppliers': suppliers,
            'warehouses': warehouses,
            'customer_demand': customer_demand,
            'transport_routes': transport_routes
        }
        
        print("Relational structure created:")
        print(f"- {len(suppliers)} suppliers")
        print(f"- {len(warehouses)} warehouses")
        print(f"- {len(customer_demand)} customer demand records")
        print(f"- {len(transport_routes)} transport routes")
    
    def create_inventory_simulation_data(self):
        """Create data for inventory simulation"""
        print("Creating inventory simulation data...")
        
        # Generate daily inventory levels for each warehouse
        dates = pd.date_range(start=self.df['Date'].min(), end=self.df['Date'].max(), freq='D')
        
        inventory_data = []
        for warehouse_id in self.relational_data['warehouses']['warehouse_id']:
            for date in dates:
                # Base inventory level
                base_inventory = 300
                
                # Add seasonal variation
                month = date.month
                if month in [12, 1, 2]:  # Winter
                    seasonal_factor = 0.8
                elif month in [3, 4, 5]:  # Spring
                    seasonal_factor = 1.1
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.3
                else:  # Fall
                    seasonal_factor = 0.9
                
                # Add random variation
                random_factor = np.random.normal(1, 0.1)
                
                # Calculate final inventory
                inventory = int(base_inventory * seasonal_factor * random_factor)
                
                inventory_data.append({
                    'warehouse_id': warehouse_id,
                    'date': date,
                    'inventory_level': max(0, inventory),
                    'reorder_quantity': max(0, 300 - inventory) if inventory < 100 else 0
                })
        
        self.relational_data['inventory_levels'] = pd.DataFrame(inventory_data)
        print(f"Created {len(self.relational_data['inventory_levels'])} inventory records")
    
    def save_cleaned_data(self):
        """Save all cleaned data to files"""
        print("Saving cleaned data...")
        
        # Ensure output directory exists
        os.makedirs(self.cleaned_data_path, exist_ok=True)
        
        # Save main cleaned dataset
        main_columns = ['datetime', 'Date', 'Hour', 'Rented Bike Count', 'Temperature(°C)', 
                       'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 
                       'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 
                       'Rainfall(mm)', 'Snowfall (cm)', 'Seasons', 'Holiday', 
                       'Functioning Day', 'year', 'month', 'day', 'day_of_week', 
                       'is_weekend', 'is_holiday']
        
        main_data = self.df[main_columns].copy()
        main_data.to_csv(os.path.join(self.cleaned_data_path, 'cleaned_data.csv'), index=False)
        
        # Save relational data
        for name, data in self.relational_data.items():
            if isinstance(data, pd.DataFrame):
                filepath = os.path.join(self.cleaned_data_path, f'{name}.csv')
                data.to_csv(filepath, index=False)
                print(f"Saved {name}.csv with {len(data)} records")
        
        # Save a summary report
        self._save_summary_report()
        
        print(f"All cleaned data saved to {self.cleaned_data_path}")
    
    def _save_summary_report(self):
        """Save a summary report of the cleaning process"""
        summary = {
            'cleaning_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_rows': 8760,  # From the dataset info
            'final_rows': len(self.df),
            'removed_rows': 8760 - len(self.df),
            'columns_processed': len(self.df.columns),
            'missing_values_final': self.df.isnull().sum().sum(),
            'data_quality_score': round((len(self.df) / 8760) * 100, 2)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(self.cleaned_data_path, 'cleaning_summary.csv'), index=False)
        
        print(f"Data quality score: {summary['data_quality_score']}%")
    
    def run_full_cleaning(self):
        """Run the complete data cleaning pipeline"""
        print("Starting SeoulBike data cleaning pipeline...")
        print("=" * 50)
        
        # Step 1: Read data
        if not self.read_data():
            return False
        
        # Step 2: Clean dates
        self.clean_dates()
        
        # Step 3: Clean numerical columns
        self.clean_numerical_columns()
        
        # Step 4: Create relational structure
        self.create_relational_structure()
        
        # Step 5: Create inventory simulation data
        self.create_inventory_simulation_data()
        
        # Step 6: Save cleaned data
        self.save_cleaned_data()
        
        print("=" * 50)
        print("Data cleaning pipeline completed successfully!")
        return True

def main():
    """Main function to run the data cleaner"""
    # Define paths
    raw_data_path = 'data/raw/SeoulBikeData.csv'
    cleaned_data_path = 'data/cleaned'
    
    # Create and run cleaner
    cleaner = SeoulBikeDataCleaner(raw_data_path, cleaned_data_path)
    success = cleaner.run_full_cleaning()
    
    if success:
        print("\nData cleaning completed successfully!")
        print(f"Cleaned data saved to: {cleaned_data_path}")
    else:
        print("\nData cleaning failed!")

if __name__ == "__main__":
    main()
