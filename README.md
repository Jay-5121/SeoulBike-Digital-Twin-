# SeoulBike Digital Twin Project

A comprehensive digital twin simulation system for Seoul's bike sharing infrastructure, incorporating machine learning forecasting, optimization algorithms, and LLM-powered insights.

## Project Structure

souelbike_digital_twin/
│
├── data/ # Raw and cleaned dataset
│ ├── raw/ # Original SeoulBike dataset
│ └── cleaned/ # Processed and cleaned data
│
├── simulation/ # Digital twin simulation code
│ ├── entities.py # Classes for Supplier, Warehouse, Transport
│ ├── simulation.py # SimPy simulation loop
│
├── forecasting/ # ML demand forecasting models
│ ├── prophet_model.py
│
├── optimization/ # Inventory optimization code
│ ├── optimizer.py
│
├── llm_interface/ # LangChain + LLM integration
│ ├── chatbot.py
│
├── dashboard/ # Streamlit or Dash app
│ ├── app.py
│
├── utils/ # Helper functions
│ ├── data_cleaning.py # Data preprocessing and cleaning
│
├── main.py # Entry point to run everything
├── requirements.txt # Python dependencies
└── README.md # This file


## Features

- **Data Processing**: Automated cleaning and preprocessing of SeoulBike dataset
- **Digital Twin Simulation**: SimPy-based simulation of supply chain operations
- **Demand Forecasting**: Prophet-based time series forecasting
- **Inventory Optimization**: Linear programming optimization for inventory management
- **LLM Integration**: AI-powered insights and natural language queries
- **Interactive Dashboard**: Real-time visualization and monitoring

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd souelbike_digital_twin
Create virtual environment:

Bash

python -m venv ven
source venv/bin/activate # On Windows: venv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
Data Preparation
The project includes a comprehensive data cleaning pipeline:

Bash

python utils/data_cleaning.py
This script:

Reads the SeoulBike dataset with proper encoding handling

Cleans missing values and standardizes data formats

Creates relational structures for suppliers → warehouses → customers

Generates inventory simulation data

Saves cleaned data to data/cleaned/

Generated Data Files
cleaned_data.csv - Main cleaned dataset with enhanced features

suppliers.csv - Supplier information and capabilities

warehouses.csv - Warehouse locations and capacities

customer_demand.csv - Aggregated demand patterns

transport_routes.csv - Transportation network

inventory_levels.csv - Daily inventory tracking

cleaning_summary.csv - Data quality report

Usage
Data Cleaning
Python

from utils.data_cleaning import SeoulBikeDataCleaner

cleaner = SeoulBikeDataCleaner('data/raw/SeoulBikeData.csv', 'data/cleaned')
cleaner.run_full_cleaning()
Running the Complete System
Bash

python main.py
Dataset Information
The SeoulBike dataset contains hourly bike rental data from Seoul's bike sharing system, including:

Temporal Features: Date, Hour, Season, Holiday status

Weather Data: Temperature, Humidity, Wind speed, Visibility, Rainfall, Snowfall

Business Metrics: Rented bike count, Functioning day status

Key Components
1. Data Cleaning (utils/data_cleaning.py)
Handles Unicode encoding issues

Standardizes date formats

Creates relational supply chain structure

Generates simulation-ready datasets

2. Digital Twin Simulation (simulation/)
SimPy-based discrete event simulation

Models suppliers, warehouses, and transportation

Real-time inventory tracking

Performance metrics and analytics

3. Demand Forecasting (forecasting/)
Prophet time series forecasting

Seasonal and trend analysis

Weather impact modeling

Demand prediction accuracy metrics

4. Optimization Engine (optimization/)
Linear programming for inventory optimization

Supply chain cost minimization

Reorder point calculations

Capacity planning

5. LLM Interface (llm_interface/)
LangChain integration

Natural language query processing

AI-powered insights generation

Conversational analytics

6. Dashboard (dashboard/)
Real-time monitoring interface

Interactive visualizations

Performance metrics display

User-friendly controls

Data Quality
The cleaning pipeline achieves:

100% data quality score for the SeoulBike dataset

0 missing values in final cleaned data

Enhanced features including datetime, seasonal, and business indicators

Relational structure ready for simulation and analysis

Contributing
Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
UCI Machine Learning Repository for the SeoulBike dataset

SimPy community for simulation framework

Prophet team for time series forecasting

LangChain for LLM integration tools

Contact
For questions or support, please open an issue in the repository.