#!/usr/bin/env python3
"""
Main Entry Point for the SeoulBike Digital Twin System

This script orchestrates the entire pipeline, from data cleaning to
running the interactive dashboard. Use command-line arguments to
control which parts of the system to run.
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

# Add the project root to the Python path to ensure modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_cleaning import SeoulBikeDataCleaner
from forecasting.prophet_model import SeoulBikeForecaster
from optimization.optimizer import InventoryOptimizer
from simulation.simulation import run_simulation
# Note: The dashboard is run as a separate process via subprocess

class MainOrchestrator:
    """Orchestrates the execution of the digital twin pipeline."""

    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        print("🚲 SeoulBike Digital Twin System Initialized")
        print("="*60)

    def check_status(self):
        """Checks the status of data files at each stage of the pipeline."""
        print("📊 SeoulBike Digital Twin System Status")
        print("="*50)
        
        paths = {
            "Cleaned Data": "data/cleaned",
            "Forecast data": "data/forecast",
            "Optimization data": "data/optimization_results",
            "Simulation data": "data/simulation_results"
        }
        
        statuses = {}
        
        print("📊 Checking Data Availability...")
        for name, path in paths.items():
            full_path = os.path.join(self.base_path, path)
            if os.path.exists(full_path) and len(os.listdir(full_path)) > 0:
                statuses[name] = True
                print(f"✅ {name} available: {len(os.listdir(full_path))} files")
            else:
                statuses[name] = False
                print(f"❌ {name} not found")
        
        print("-" * 50)
        print("\n🔄 Pipeline Status:")
        print(f"  Data Cleaning: {'✅' if statuses['Cleaned Data'] else '❌'}")
        print(f"  Forecasting: {'✅' if statuses['Forecast data'] else '❌'}")
        print(f"  Optimization: {'✅' if statuses['Optimization data'] else '❌'}")
        print(f"  Simulation: {'✅' if statuses['Simulation data'] else '❌'}")
        
        print("\n📁 Data Directories:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
            
        print("="*50)

    def run_cleaning(self):
        """Runs the data cleaning pipeline."""
        print("🧹 Running Data Cleaning Pipeline...")
        try:
            cleaner = SeoulBikeDataCleaner(
                raw_data_path=os.path.join(self.base_path, 'data/raw/SeoulBikeData.csv'),
                cleaned_data_path=os.path.join(self.base_path, 'data/cleaned/')
            )
            # --- THIS IS THE FIX ---
            cleaner.run_full_cleaning() 
            # ---------------------
            print("✅ Data cleaning completed successfully!")
        except Exception as e:
            print(f"❌ Error in data cleaning: {e}")
            sys.exit(1)

    def run_forecasting(self):
        """Runs the demand forecasting pipeline."""
        if not os.path.exists(os.path.join(self.base_path, 'data/cleaned')):
            print("❌ Cleaned data not available. Run data cleaning first.")
            return
            
        print("🔮 Running Demand Forecasting Pipeline...")
        try:
            forecaster = SeoulBikeForecaster(
                data_path=os.path.join(self.base_path, 'data/cleaned/'),
                forecast_path=os.path.join(self.base_path, 'data/forecast/')
            )
            results = forecaster.run_full_forecasting(forecast_days=90)
            print("✅ Forecasting completed successfully!")
            print(f"  Models trained: {results.get('models_trained', 0)}")
            print(f"  Forecasts generated: {results.get('forecasts_generated', 0)}")
        except Exception as e:
            print(f"❌ Error in forecasting: {e}")
            sys.exit(1)

    def run_optimization(self):
        """Runs the inventory optimization pipeline."""
        if not os.path.exists(os.path.join(self.base_path, 'data/forecast')):
            print("❌ Forecast data not available. Run forecasting first.")
            return

        print("⚡ Running Inventory Optimization Pipeline...")
        try:
            optimizer = InventoryOptimizer(
                data_path=os.path.join(self.base_path, 'data/cleaned/'),
                forecast_path=os.path.join(self.base_path, 'data/forecast/')
            )
            results = optimizer.run_full_optimization()
            print("✅ Optimization completed successfully!")
            print(f"  Total potential cost savings: ${results['report']['total_cost_savings']:,.2f}")
        except Exception as e:
            print(f"❌ Error in optimization: {e}")
            sys.exit(1)

    def run_simulation(self, days=30):
        """Runs the digital twin simulation."""
        if not os.path.exists(os.path.join(self.base_path, 'data/cleaned')):
            print("❌ Cleaned data not available. Run cleaning first.")
            return

        print(f"🎮 Running Digital Twin Simulation ({days} days)...")
        try:
            results = run_simulation(simulation_days=days)
            summary = results.get_simulation_summary()
            print("✅ Simulation completed successfully!")
            print(f"  Final total cost: ${summary.get('final_total_cost', 0):,.2f}")
            print(f"  Final service level: {summary.get('final_service_level', 0):.2%}")
            print(f"  Total stockouts: {summary.get('total_stockouts', 0)}")
        except Exception as e:
            print(f"❌ Error in simulation: {e}")
            sys.exit(1)

    def run_pipeline(self, days=30):
        """Runs the complete end-to-end pipeline."""
        start_time = datetime.now()
        print("🚀 Running Complete SeoulBike Digital Twin Pipeline...")
        print("="*60)
        self.run_cleaning()
        self.run_forecasting()
        self.run_optimization()
        self.run_simulation(days)
        end_time = datetime.now()
        print("="*60)
        print("🎉 Complete Pipeline Finished Successfully!")
        print(f"⏱️  Total duration: {end_time - start_time}")
        print("="*60)

    def start_dashboard(self):
        """Starts the Streamlit dashboard."""
        print("📈 Launching Streamlit Dashboard...")
        print("Please open your web browser to the URL shown below.")
        
        dashboard_path = os.path.join(self.base_path, 'dashboard', 'app.py')
        try:
            subprocess.run(["streamlit", "run", dashboard_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch dashboard: {e}")
        except FileNotFoundError:
             print("❌ 'streamlit' command not found. Is Streamlit installed and in your PATH?")


def main():
    parser = argparse.ArgumentParser(
        description="SeoulBike Digital Twin System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py --status           # Show system status
  python main.py --clean            # Run data cleaning only
  python main.py --forecast         # Run forecasting only
  python main.py --optimize         # Run optimization only
  python main.py --simulate         # Run simulation only
  python main.py --pipeline         # Run complete pipeline
  python main.py --dashboard        # Start dashboard only
  python main.py --all              # Run pipeline + dashboard
"""
    )
    
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--clean", action="store_true", help="Run data cleaning")
    parser.add_argument("--forecast", action="store_true", help="Run forecasting")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--pipeline", action="store_true", help="Run complete pipeline")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard")
    parser.add_argument("--all", action="store_true", help="Run pipeline + dashboard")
    parser.add_argument("--days", type=int, default=30, help="Simulation days (default: 30)")
    
    args = parser.parse_args()
    orchestrator = MainOrchestrator()

    if args.status:
        orchestrator.check_status()
    elif args.clean:
        orchestrator.run_cleaning()
    elif args.forecast:
        orchestrator.run_forecasting()
    elif args.optimize:
        orchestrator.run_optimization()
    elif args.simulate:
        orchestrator.run_simulation(args.days)
    elif args.pipeline:
        orchestrator.run_pipeline(args.days)
    elif args.dashboard:
        orchestrator.start_dashboard()
    elif args.all:
        orchestrator.run_pipeline(args.days)
        orchestrator.start_dashboard()
    else:
        # Default action if no arguments are given
        print("============================================================")
        print("🚲 SeoulBike Digital Twin System")
        print("============================================================")
        print("Use --help for command-line options")
        print("Use --status to check system status")
        print("Use --all to run complete pipeline and start dashboard")


if __name__ == "__main__":
    main()