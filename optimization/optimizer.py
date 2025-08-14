#!/usr/bin/env python3
"""
Inventory Optimization Engine for SeoulBike Digital Twin Project

This module implements:
- Economic Order Quantity (EOQ) calculations
- Safety Stock calculations
- PuLP-based optimization to minimize total costs
- Multi-warehouse optimization with constraints
"""

import pandas as pd
import numpy as np
from pulp import *
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """Inventory optimization engine using EOQ and linear programming"""
    
    def __init__(self, data_path: str = 'data/cleaned/', forecast_path: str = 'data/forecast/'):
        """
        Initialize the optimizer
        
        Args:
            data_path: Path to cleaned data files
            forecast_path: Path to forecast data files
        """
        self.data_path = data_path
        self.forecast_path = forecast_path
        self.cleaned_data = None
        self.forecast_data = {}
        self.warehouse_data = None
        self.supplier_data = None
        
        # Optimization parameters
        self.optimization_params = {
            'service_level': 0.95,  # 95% service level
            'lead_time_variability': 0.2,  # 20% lead time variability
            'holding_cost_rate': 0.25,  # 25% annual holding cost
            'order_cost': 50.0,  # Fixed cost per order
            'stockout_cost': 100.0,  # Cost per unit of stockout
            'transport_cost_per_unit': 5.0  # Transport cost per unit
        }
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load required data for optimization"""
        try:
            # Load cleaned data
            cleaned_file = os.path.join(self.data_path, 'cleaned_data.csv')
            if os.path.exists(cleaned_file):
                self.cleaned_data = pd.read_csv(cleaned_file)
                self.cleaned_data['datetime'] = pd.to_datetime(self.cleaned_data['datetime'])
                print(f"Loaded cleaned data: {len(self.cleaned_data)} records")
            
            # Load warehouse data
            warehouse_file = os.path.join(self.data_path, 'warehouses.csv')
            if os.path.exists(warehouse_file):
                self.warehouse_data = pd.read_csv(warehouse_file)
                print(f"Loaded warehouse data: {len(self.warehouse_data)} warehouses")
            
            # Load supplier data
            supplier_file = os.path.join(self.data_path, 'suppliers.csv')
            if os.path.exists(supplier_file):
                self.supplier_data = pd.read_csv(supplier_file)
                print(f"Loaded supplier data: {len(self.supplier_data)} suppliers")
            
            # Load forecast data
            self.load_forecast_data()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def load_forecast_data(self):
        """Load forecast data for optimization"""
        try:
            forecast_files = [f for f in os.listdir(self.forecast_path) if f.startswith('forecast_warehouse_') and f.endswith('.csv')]
            
            for forecast_file in forecast_files:
                warehouse_id = int(forecast_file.split('_')[2].split('.')[0])
                forecast_path = os.path.join(self.forecast_path, forecast_file)
                
                if os.path.exists(forecast_path):
                    forecast_df = pd.read_csv(forecast_path)
                    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                    self.forecast_data[warehouse_id] = forecast_df
                    print(f"Loaded forecast for warehouse {warehouse_id}: {len(forecast_df)} records")
            
        except Exception as e:
            print(f"Error loading forecast data: {e}")
    
    def calculate_eoq(self, annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
        """
        Calculate Economic Order Quantity
        
        Args:
            annual_demand: Annual demand in units
            order_cost: Fixed cost per order
            holding_cost_per_unit: Annual holding cost per unit
            
        Returns:
            float: Optimal order quantity
        """
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
        return eoq
    
    def calculate_safety_stock(self, lead_time: float, demand_std: float, service_level: float = 0.95) -> float:
        """
        Calculate Safety Stock using normal distribution
        
        Args:
            lead_time: Lead time in days
            demand_std: Standard deviation of daily demand
            service_level: Desired service level (default: 0.95)
            
        Returns:
            float: Safety stock level
        """
        # Z-score for service level (95% = 1.645)
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.645)
        
        # Safety stock = Z * sqrt(lead_time) * demand_std
        safety_stock = z_score * np.sqrt(lead_time) * demand_std
        return safety_stock
    
    def calculate_reorder_point(self, avg_daily_demand: float, lead_time: float, safety_stock: float) -> float:
        """
        Calculate reorder point
        
        Args:
            avg_daily_demand: Average daily demand
            lead_time: Lead time in days
            safety_stock: Safety stock level
            
        Returns:
            float: Reorder point
        """
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        return reorder_point
    
    def calculate_total_inventory_cost(self, order_quantity: float, annual_demand: float, 
                                     holding_cost_per_unit: float, order_cost: float) -> float:
        """
        Calculate total annual inventory cost
        
        Args:
            order_quantity: Order quantity
            annual_demand: Annual demand
            holding_cost_per_unit: Annual holding cost per unit
            order_cost: Fixed cost per order
            
        Returns:
            float: Total annual inventory cost
        """
        # Annual ordering cost
        annual_orders = annual_demand / order_quantity
        annual_ordering_cost = annual_orders * order_cost
        
        # Annual holding cost
        avg_inventory = order_quantity / 2
        annual_holding_cost = avg_inventory * holding_cost_per_unit
        
        # Total cost
        total_cost = annual_ordering_cost + annual_holding_cost
        return total_cost
    
    def optimize_single_warehouse(self, warehouse_id: int) -> Dict:
        """
        Optimize inventory for a single warehouse
        
        Args:
            warehouse_id: Warehouse ID to optimize
            
        Returns:
            Dict: Optimization results
        """
        if warehouse_id not in self.forecast_data:
            return {}
        
        forecast_df = self.forecast_data[warehouse_id]
        
        # Get warehouse parameters
        warehouse_info = self.warehouse_data[self.warehouse_data['warehouse_id'] == warehouse_id].iloc[0]
        storage_capacity = warehouse_info['storage_capacity']
        current_reorder_point = warehouse_info['reorder_point']
        current_reorder_quantity = warehouse_info['reorder_quantity']
        
        # Calculate demand statistics
        future_forecast = forecast_df[forecast_df['date'] > pd.Timestamp.now()]
        if len(future_forecast) == 0:
            return {}
        
        annual_demand = future_forecast['predicted_demand'].sum() * (365 / len(future_forecast))
        avg_daily_demand = future_forecast['predicted_demand'].mean()
        demand_std = future_forecast['predicted_demand'].std()
        
        # Get supplier lead time (use average if multiple suppliers)
        avg_lead_time = 3.0  # Default 3 days
        
        # Calculate optimal parameters
        optimal_eoq = self.calculate_eoq(
            annual_demand=annual_demand,
            order_cost=self.optimization_params['order_cost'],
            holding_cost_per_unit=self.optimization_params['holding_cost_rate'] * 150  # Assume $150 per bike
        )
        
        safety_stock = self.calculate_safety_stock(
            lead_time=avg_lead_time,
            demand_std=demand_std,
            service_level=self.optimization_params['service_level']
        )
        
        reorder_point = self.calculate_reorder_point(
            avg_daily_demand=avg_daily_demand,
            lead_time=avg_lead_time,
            safety_stock=safety_stock
        )
        
        # Calculate costs
        current_total_cost = self.calculate_total_inventory_cost(
            order_quantity=current_reorder_quantity,
            annual_demand=annual_demand,
            holding_cost_per_unit=self.optimization_params['holding_cost_rate'] * 150,
            order_cost=self.optimization_params['order_cost']
        )
        
        optimal_total_cost = self.calculate_total_inventory_cost(
            order_quantity=optimal_eoq,
            annual_demand=annual_demand,
            holding_cost_per_unit=self.optimization_params['holding_cost_rate'] * 150,
            order_cost=self.optimization_params['order_cost']
        )
        
        cost_savings = current_total_cost - optimal_total_cost
        
        return {
            'warehouse_id': warehouse_id,
            'annual_demand': annual_demand,
            'avg_daily_demand': avg_daily_demand,
            'demand_std': demand_std,
            'current_reorder_point': current_reorder_point,
            'current_reorder_quantity': current_reorder_quantity,
            'optimal_eoq': optimal_eoq,
            'safety_stock': safety_stock,
            'optimal_reorder_point': reorder_point,
            'current_total_cost': current_total_cost,
            'optimal_total_cost': optimal_total_cost,
            'cost_savings': cost_savings,
            'cost_savings_percentage': (cost_savings / current_total_cost) * 100 if current_total_cost > 0 else 0
        }
    
    def optimize_multi_warehouse(self) -> Dict:
        """
        Optimize inventory across all warehouses using linear programming
        
        Returns:
            Dict: Multi-warehouse optimization results
        """
        print("Running multi-warehouse optimization...")
        
        # Create optimization problem
        prob = LpProblem("SeoulBike_Inventory_Optimization", LpMinimize)
        
        # Decision variables
        warehouses = list(self.forecast_data.keys())
        suppliers = list(self.supplier_data['supplier_id']) if self.supplier_data is not None else [1, 2, 3]
        
        # Order quantities for each warehouse
        order_quantities = LpVariable.dicts("order_quantity",
                                          [(w, s) for w in warehouses for s in suppliers],
                                          lowBound=0,
                                          cat='Integer')
        
        # Safety stock levels for each warehouse
        safety_stocks = LpVariable.dicts("safety_stock",
                                       warehouses,
                                       lowBound=0,
                                       cat='Integer')
        
        # Objective function: minimize total cost
        prob += lpSum([
            # Ordering costs
            self.optimization_params['order_cost'] * order_quantities[w, s]
            for w in warehouses for s in suppliers
        ]) + lpSum([
            # Holding costs
            self.optimization_params['holding_cost_rate'] * 150 * safety_stocks[w]
            for w in warehouses
        ]) + lpSum([
            # Transport costs
            self.optimization_params['transport_cost_per_unit'] * order_quantities[w, s]
            for w in warehouses for s in suppliers
        ])
        
        # Constraints
        for w in warehouses:
            if w in self.forecast_data:
                forecast_df = self.forecast_data[w]
                future_forecast = forecast_df[forecast_df['date'] > pd.Timestamp.now()]
                
                if len(future_forecast) > 0:
                    annual_demand = future_forecast['predicted_demand'].sum() * (365 / len(future_forecast))
                    avg_daily_demand = future_forecast['predicted_demand'].mean()
                    demand_std = future_forecast['predicted_demand'].std()
                    
                    # Demand satisfaction constraint
                    prob += lpSum([order_quantities[w, s] for s in suppliers]) >= annual_demand
                    
                    # Safety stock constraint (minimum safety stock based on demand variability)
                    min_safety_stock = demand_std * 2  # 2 standard deviations
                    prob += safety_stocks[w] >= min_safety_stock
                    
                    # Storage capacity constraint
                    warehouse_info = self.warehouse_data[self.warehouse_data['warehouse_id'] == w].iloc[0]
                    storage_capacity = warehouse_info['storage_capacity']
                    prob += safety_stocks[w] + lpSum([order_quantities[w, s] for s in suppliers]) <= storage_capacity
        
        # Supplier capacity constraints
        for s in suppliers:
            if self.supplier_data is not None:
                supplier_info = self.supplier_data[self.supplier_data['supplier_id'] == s]
                if len(supplier_info) > 0:
                    daily_capacity = supplier_info.iloc[0]['capacity_per_day']
                    prob += lpSum([order_quantities[w, s] for w in warehouses]) <= daily_capacity * 365
        
        # Solve the problem
        try:
            prob.solve()
            
            if prob.status == LpStatusOptimal:
                print("Multi-warehouse optimization completed successfully!")
                
                # Extract results
                results = {
                    'status': 'optimal',
                    'total_cost': value(prob.objective),
                    'warehouse_results': {},
                    'supplier_results': {}
                }
                
                # Warehouse results
                for w in warehouses:
                    results['warehouse_results'][w] = {
                        'safety_stock': value(safety_stocks[w]),
                        'total_orders': sum(value(order_quantities[w, s]) for s in suppliers)
                    }
                
                # Supplier results
                for s in suppliers:
                    results['supplier_results'][s] = {
                        'total_orders': sum(value(order_quantities[w, s]) for w in warehouses)
                    }
                
                return results
            else:
                print(f"Optimization failed with status: {LpStatus[prob.status]}")
                return {'status': 'failed', 'reason': LpStatus[prob.status]}
                
        except Exception as e:
            print(f"Error in optimization: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def run_full_optimization(self) -> Dict:
        """
        Run the complete optimization pipeline
        
        Returns:
            Dict: Complete optimization results
        """
        print("Starting SeoulBike inventory optimization pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Single warehouse optimization
            single_warehouse_results = {}
            for warehouse_id in self.forecast_data.keys():
                result = self.optimize_single_warehouse(warehouse_id)
                if result:
                    single_warehouse_results[warehouse_id] = result
                    print(f"Warehouse {warehouse_id} optimization completed")
            
            # Step 2: Multi-warehouse optimization
            multi_warehouse_results = self.optimize_multi_warehouse()
            
            # Step 3: Generate optimization report
            optimization_report = self.generate_optimization_report(
                single_warehouse_results, multi_warehouse_results
            )
            
            # Step 4: Save results
            self.save_optimization_results(
                single_warehouse_results, multi_warehouse_results, optimization_report
            )
            
            print("=" * 60)
            print("Optimization pipeline completed successfully!")
            
            return {
                'single_warehouse': single_warehouse_results,
                'multi_warehouse': multi_warehouse_results,
                'report': optimization_report
            }
            
        except Exception as e:
            print(f"Error in optimization pipeline: {e}")
            raise
    
    def generate_optimization_report(self, single_results: Dict, multi_results: Dict) -> Dict:
        """Generate comprehensive optimization report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'single_warehouse_summary': {},
            'multi_warehouse_summary': {},
            'total_cost_savings': 0,
            'recommendations': []
        }
        
        # Single warehouse summary
        total_single_cost_savings = 0
        for warehouse_id, result in single_results.items():
            report['single_warehouse_summary'][warehouse_id] = {
                'cost_savings': result['cost_savings'],
                'cost_savings_percentage': result['cost_savings_percentage'],
                'optimal_eoq': result['optimal_eoq'],
                'safety_stock': result['safety_stock']
            }
            total_single_cost_savings += result['cost_savings']
        
        # Multi-warehouse summary
        if multi_results.get('status') == 'optimal':
            report['multi_warehouse_summary'] = {
                'total_cost': multi_results['total_cost'],
                'status': 'optimal'
            }
        
        # Total cost savings
        report['total_cost_savings'] = total_single_cost_savings
        
        # Generate recommendations
        recommendations = []
        
        # EOQ recommendations
        for warehouse_id, result in single_results.items():
            if result['cost_savings_percentage'] > 10:
                recommendations.append({
                    'warehouse_id': warehouse_id,
                    'type': 'EOQ_optimization',
                    'priority': 'high',
                    'description': f"High cost savings potential: {result['cost_savings_percentage']:.1f}%",
                    'action': f"Change reorder quantity from {result['current_reorder_quantity']} to {result['optimal_eoq']:.0f}"
                })
        
        # Safety stock recommendations
        for warehouse_id, result in single_results.items():
            if result['safety_stock'] > result['current_reorder_point'] * 0.5:
                recommendations.append({
                    'warehouse_id': warehouse_id,
                    'type': 'safety_stock',
                    'priority': 'medium',
                    'description': f"Safety stock should be {result['safety_stock']:.0f} units",
                    'action': f"Increase reorder point to {result['optimal_reorder_point']:.0f}"
                })
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_optimization_results(self, single_results: Dict, multi_results: Dict, report: Dict):
        """Save optimization results to files"""
        # Create results directory
        os.makedirs('data/optimization_results', exist_ok=True)
        
        # Save single warehouse results
        single_df = pd.DataFrame.from_dict(single_results, orient='index')
        single_df.to_csv('data/optimization_results/single_warehouse_optimization.csv')
        
        # Save multi-warehouse results
        if multi_results.get('status') == 'optimal':
            with open('data/optimization_results/multi_warehouse_optimization.json', 'w') as f:
                json.dump(multi_results, f, indent=2, default=str)
        
        # Save optimization report
        with open('data/optimization_results/optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Optimization results saved to data/optimization_results/")
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results"""
        try:
            report_file = 'data/optimization_results/optimization_report.json'
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report = json.load(f)
                return report
            else:
                return {}
        except Exception as e:
            print(f"Error loading optimization summary: {e}")
            return {}


def main():
    """Main function to run the optimizer"""
    # Create and run optimizer
    optimizer = InventoryOptimizer()
    
    try:
        results = optimizer.run_full_optimization()
        
        print(f"\nOptimization completed!")
        print(f"Single warehouse optimizations: {len(results['single_warehouse'])}")
        print(f"Multi-warehouse status: {results['multi_warehouse'].get('status', 'unknown')}")
        
        # Show cost savings
        total_savings = results['report']['total_cost_savings']
        print(f"Total potential cost savings: ${total_savings:,.2f}")
        
        # Show recommendations
        recommendations = results['report']['recommendations']
        if recommendations:
            print(f"\nTop recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec['description']}")
                print(f"   Action: {rec['action']}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")


if __name__ == "__main__":
    main()
