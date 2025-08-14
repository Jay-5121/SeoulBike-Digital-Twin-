#!/usr/bin/env python3
"""
SimPy Simulation Environment for SeoulBike Digital Twin Project

This module creates a discrete event simulation environment that:
- Processes customer orders based on demand data
- Manages warehouse inventory levels
- Places supplier orders when below reorder point
- Updates stock after lead time
- Tracks KPIs: stockouts, holding cost, order cost
"""

import simpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import json
import os

from .entities import Supplier, Warehouse, Transport, SupplyChainNetwork


class SeoulBikeSimulation:
    """Main simulation class for SeoulBike digital twin"""
    
    def __init__(self, simulation_days: int = 30, time_step_hours: int = 1):
        """
        Initialize the simulation environment
        
        Args:
            simulation_days: Number of days to simulate
            time_step_hours: Time step for simulation (default: 1 hour)
        """
        self.simulation_days = simulation_days
        self.time_step_hours = time_step_hours
        self.total_simulation_steps = simulation_days * 24 // time_step_hours
        
        # Create SimPy environment
        self.env = simpy.Environment()
        
        # Initialize supply chain network
        self.network = SupplyChainNetwork(self.env)
        
        # Simulation state
        self.current_step = 0
        self.simulation_log = []
        self.kpi_history = []
        
        # Load demand data
        self.demand_data = None
        self.load_demand_data()
        
        # Initialize entities
        self.setup_supply_chain()
        
        # Simulation processes
        self.demand_process = None
        self.inventory_process = None
        self.delivery_process = None
        self.monitoring_process = None
    
    def load_demand_data(self):
        """Load cleaned demand data for simulation"""
        try:
            demand_file = 'data/cleaned/customer_demand.csv'
            if os.path.exists(demand_file):
                self.demand_data = pd.read_csv(demand_file)
                # Convert Date column to datetime
                self.demand_data['Date'] = pd.to_datetime(self.demand_data['Date'])
                print(f"Loaded demand data: {len(self.demand_data)} records")
            else:
                print("Warning: Demand data not found, using synthetic data")
                self.create_synthetic_demand()
        except Exception as e:
            print(f"Error loading demand data: {e}")
            self.create_synthetic_demand()
    
    def create_synthetic_demand(self):
        """Create synthetic demand data for simulation"""
        dates = pd.date_range(start='2024-01-01', periods=self.simulation_days, freq='D')
        synthetic_demand = []
        
        for date in dates:
            for hour in range(24):
                # Base demand with seasonal and hourly patterns
                base_demand = 100
                seasonal_factor = 1.0
                hourly_factor = 1.0
                
                # Seasonal variation
                month = date.month
                if month in [12, 1, 2]:  # Winter
                    seasonal_factor = 0.7
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.3
                
                # Hourly variation (peak hours)
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                    hourly_factor = 1.5
                elif 0 <= hour <= 5:  # Night hours
                    hourly_factor = 0.3
                
                demand = int(base_demand * seasonal_factor * hourly_factor * random.uniform(0.8, 1.2))
                
                synthetic_demand.append({
                    'Date': date,
                    'Hour': hour,
                    'Rented Bike Count': demand,
                    'warehouse_id': random.choice([1, 2, 3])
                })
        
        self.demand_data = pd.DataFrame(synthetic_demand)
        print(f"Created synthetic demand data: {len(self.demand_data)} records")
    
    def setup_supply_chain(self):
        """Initialize supply chain entities"""
        print("Setting up supply chain entities...")
        
        # Create suppliers
        suppliers_data = [
            {'id': 1, 'name': 'BikeSupplier_A', 'location': 'Seoul_Center', 'capacity': 100, 'reliability': 0.95, 'lead_time': 3, 'cost': 150.0},
            {'id': 2, 'name': 'BikeSupplier_B', 'location': 'Seoul_North', 'capacity': 80, 'reliability': 0.88, 'lead_time': 4, 'cost': 145.0},
            {'id': 3, 'name': 'BikeSupplier_C', 'location': 'Seoul_South', 'capacity': 120, 'reliability': 0.92, 'lead_time': 2, 'cost': 160.0},
            {'id': 4, 'name': 'BikeSupplier_D', 'location': 'Seoul_East', 'capacity': 90, 'reliability': 0.85, 'lead_time': 5, 'cost': 140.0},
            {'id': 5, 'name': 'BikeSupplier_E', 'location': 'Seoul_West', 'capacity': 110, 'reliability': 0.90, 'lead_time': 3, 'cost': 155.0}
        ]
        
        for sup_data in suppliers_data:
            supplier = Supplier(
                supplier_id=sup_data['id'],
                name=sup_data['name'],
                location=sup_data['location'],
                capacity_per_day=sup_data['capacity'],
                reliability_score=sup_data['reliability'],
                lead_time_days=sup_data['lead_time'],
                cost_per_unit=sup_data['cost'],
                env=self.env
            )
            self.network.add_supplier(supplier)
        
        # Create warehouses
        warehouses_data = [
            {'id': 1, 'name': 'Central_Warehouse', 'location': 'Seoul_Center', 'capacity': 500, 'initial': 300, 'reorder_point': 100, 'reorder_qty': 200, 'holding_cost': 2.0},
            {'id': 2, 'name': 'North_Warehouse', 'location': 'Seoul_North', 'capacity': 400, 'initial': 250, 'reorder_point': 80, 'reorder_qty': 150, 'holding_cost': 2.2},
            {'id': 3, 'name': 'South_Warehouse', 'location': 'Seoul_South', 'capacity': 450, 'initial': 280, 'reorder_point': 90, 'reorder_qty': 180, 'holding_cost': 2.1}
        ]
        
        for wh_data in warehouses_data:
            warehouse = Warehouse(
                warehouse_id=wh_data['id'],
                name=wh_data['name'],
                location=wh_data['location'],
                storage_capacity=wh_data['capacity'],
                initial_inventory=wh_data['initial'],
                reorder_point=wh_data['reorder_point'],
                reorder_quantity=wh_data['reorder_qty'],
                holding_cost_per_unit=wh_data['holding_cost'],
                env=self.env,
                network=self.network  # This line is crucial
            )
            self.network.add_warehouse(warehouse)
        
        # Create transport routes
        transport_data = [
            {'id': 1, 'from': 'Seoul_Center', 'to': 'Seoul_North', 'distance': 15, 'time': 1.5, 'cost': 2.5, 'capacity': 300},
            {'id': 2, 'from': 'Seoul_North', 'to': 'Seoul_Center', 'distance': 15, 'time': 1.5, 'cost': 2.5, 'capacity': 300},
            {'id': 3, 'from': 'Seoul_South', 'to': 'Seoul_Center', 'distance': 20, 'time': 2.0, 'cost': 2.8, 'capacity': 250},
            {'id': 4, 'from': 'Seoul_East', 'to': 'Seoul_Center', 'distance': 18, 'time': 1.8, 'cost': 2.6, 'capacity': 280},
            {'id': 5, 'from': 'Seoul_West', 'to': 'Seoul_Center', 'distance': 22, 'time': 2.2, 'cost': 3.0, 'capacity': 270}
        ]
        
        for trans_data in transport_data:
            transport = Transport(
                route_id=trans_data['id'],
                from_location=trans_data['from'],
                to_location=trans_data['to'],
                distance_km=trans_data['distance'],
                avg_travel_time_hours=trans_data['time'],
                cost_per_km=trans_data['cost'],
                capacity=trans_data['capacity'],
                env=self.env
            )
            self.network.add_transport_route(transport)
        
        print(f"Created {len(self.network.suppliers)} suppliers, {len(self.network.warehouses)} warehouses, {len(self.network.transport_routes)} transport routes")
    
    def start_simulation(self):
        """Start the simulation processes"""
        print("Starting SeoulBike simulation...")
        
        # Start simulation processes
        self.demand_process = self.env.process(self.process_customer_demand())
        self.inventory_process = self.env.process(self.process_inventory_management())
        self.delivery_process = self.env.process(self.process_deliveries())
        self.monitoring_process = self.env.process(self.monitor_kpis())
        
        # Run simulation
        self.env.run(until=self.total_simulation_steps)
        
        print("Simulation completed!")
        self.generate_simulation_report()
    
    def process_customer_demand(self):
        """Process customer demand and update warehouse inventory"""
        while self.current_step < self.total_simulation_steps:
            # Get current time
            current_time = self.current_step * self.time_step_hours
            current_day = (current_time // 24) % self.simulation_days
            current_hour = current_time % 24
            
            # Get demand for current time
            if self.demand_data is not None:
                # Find matching demand data (handles wrapping around the year)
                matching_demand = self.demand_data[
                    (self.demand_data['Date'].dt.dayofyear == (current_day + 1)) & 
                    (self.demand_data['Hour'] == current_hour)
                ]
                
                for _, demand_row in matching_demand.iterrows():
                    warehouse_id = demand_row['warehouse_id']
                    demand_quantity = demand_row['Rented Bike Count']
                    
                    if warehouse_id in self.network.warehouses:
                        warehouse = self.network.warehouses[warehouse_id]
                        warehouse.daily_demand += demand_quantity
                        
                        # Update inventory based on demand
                        initial_inventory = warehouse.current_inventory
                        warehouse.update_inventory(demand_quantity, 'subtract')
                        fulfilled = initial_inventory - warehouse.current_inventory

                        if fulfilled < demand_quantity:
                             self.log_event('stockout', {
                                'warehouse_id': warehouse_id,
                                'requested': demand_quantity,
                                'fulfilled': fulfilled,
                                'time': self.env.now
                            })
                        else:
                            self.log_event('demand_fulfilled', {
                                'warehouse_id': warehouse_id,
                                'quantity': demand_quantity,
                                'time': self.env.now
                            })
            
            # Wait for next time step
            yield self.env.timeout(1)
            self.current_step += 1
    
    def process_inventory_management(self):
        """Process inventory management and reorder logic"""
        while self.current_step < self.total_simulation_steps:
            # Check each warehouse for reorder needs
            for warehouse in self.network.warehouses.values():
                if warehouse.check_reorder_needed():
                    # Find appropriate supplier
                    supplier = self.find_best_supplier(warehouse)
                    if supplier:
                        order = warehouse.place_reorder(supplier)
                        if order:
                            self.log_event('reorder_placed', {
                                'warehouse_id': warehouse.warehouse_id,
                                'supplier_id': supplier.supplier_id,
                                'quantity': order['quantity'],
                                'expected_delivery': order['expected_delivery'],
                                'time': self.env.now
                            })
            
            # Wait for next check (every 4 hours)
            yield self.env.timeout(4)
    
    def process_deliveries(self):
        """Process supplier deliveries to warehouses"""
        while self.current_step < self.total_simulation_steps:
            # Check for completed deliveries
            for warehouse in self.network.warehouses.values():
                for order in warehouse.orders_in_transit[:]:  # Copy list to avoid modification during iteration
                    if order['status'] == 'processing' and self.env.now >= order['expected_delivery']:
                        
                        warehouse.receive_delivery(order['order_id'])
                        
                        # The order status is updated inside receive_delivery, so we recheck it
                        if order['status'] == 'delivered':
                            self.log_event('delivery_received', {
                                'warehouse_id': warehouse.warehouse_id,
                                'order_id': order['order_id'],
                                'quantity': order['quantity'],
                                'time': self.env.now
                            })
                        else: # Failed
                            self.log_event('delivery_failed', {
                                'warehouse_id': warehouse.warehouse_id,
                                'order_id': order['order_id'],
                                'reason': order.get('reason', 'Unknown'),
                                'time': self.env.now
                            })
            
            # Wait for next check (every hour)
            yield self.env.timeout(1)
    
    def monitor_kpis(self):
        """Monitor and record KPIs during simulation"""
        while self.current_step < self.total_simulation_steps:
            # Calculate current KPIs
            kpis = self.calculate_current_kpis()
            self.kpi_history.append(kpis)
            
            # Log KPI update
            self.log_event('kpi_update', kpis)
            
            # Wait for next KPI calculation (every 6 hours)
            yield self.env.timeout(6)
    
    def find_best_supplier(self, warehouse: Warehouse) -> Optional[Supplier]:
        """Find the best supplier for a warehouse based on cost and reliability"""
        available_suppliers = []
        
        for supplier in self.network.suppliers.values():
            if supplier.can_fulfill_order(warehouse.reorder_quantity):
                # Calculate supplier score (lower is better)
                cost_score = supplier.cost_per_unit
                reliability_score = (1 - supplier.reliability_score) * 100 # Penalize unreliability
                lead_time_score = supplier.lead_time_days * 10 # Penalize long lead times
                
                total_score = cost_score + reliability_score + lead_time_score
                available_suppliers.append((supplier, total_score))
        
        if available_suppliers:
            # Return supplier with lowest score
            best_supplier = min(available_suppliers, key=lambda x: x[1])
            return best_supplier[0]
        
        return None
    
    def calculate_current_kpis(self) -> Dict:
        """Calculate current Key Performance Indicators"""
        total_inventory = sum(w.current_inventory for w in self.network.warehouses.values())
        total_stockouts = sum(w.total_stockouts for w in self.network.warehouses.values())
        total_holding_cost = sum(w.total_holding_cost for w in self.network.warehouses.values())
        total_order_cost = sum(s.total_cost for s in self.network.suppliers.values())
        total_transport_cost = sum(t.total_cost for t in self.network.transport_routes.values())
        
        # Calculate service level
        total_demand = sum(w.daily_demand for w in self.network.warehouses.values())
        service_level = 1 - (total_stockouts / max(total_demand, 1))
        
        return {
            'time': self.env.now,
            'total_inventory': total_inventory,
            'total_stockouts': total_stockouts,
            'total_holding_cost': total_holding_cost,
            'total_order_cost': total_order_cost,
            'total_transport_cost': total_transport_cost,
            'total_cost': total_holding_cost + total_order_cost + total_transport_cost,
            'service_level': service_level,
            'network_efficiency': self.network.get_network_stats()['network_efficiency']
        }
    
    def log_event(self, event_type: str, event_data: Dict):
        """Log simulation events"""
        event = {
            'timestamp': self.env.now,
            'event_type': event_type,
            'event_data': event_data
        }
        self.simulation_log.append(event)
    
    def generate_simulation_report(self):
        """Generate comprehensive simulation report"""
        print("\n" + "="*50)
        print("SIMULATION REPORT")
        print("="*50)
        
        # Final KPIs
        final_kpis = self.calculate_current_kpis()
        print(f"Final Total Cost: ${final_kpis['total_cost']:,.2f}")
        print(f"Final Service Level: {final_kpis['service_level']:.2%}")
        print(f"Total Stockouts: {final_kpis['total_stockouts']}")
        print(f"Network Efficiency: {final_kpis['network_efficiency']:.3f}")
        
        # Warehouse summary
        print("\nWAREHOUSE SUMMARY:")
        for warehouse in self.network.warehouses.values():
            stats = warehouse.get_warehouse_stats()
            print(f"  {stats['name']}: Inventory={stats['current_inventory']}, "
                  f"Stockouts={stats['total_stockouts']}, "
                  f"Holding Cost=${stats['total_holding_cost']:,.2f}")
        
        # Save simulation results
        self.save_simulation_results()
    
    def save_simulation_results(self):
        """Save simulation results to files"""
        # Create results directory
        os.makedirs('data/simulation_results', exist_ok=True)
        
        # Save KPI history
        kpi_df = pd.DataFrame(self.kpi_history)
        kpi_df.to_csv('data/simulation_results/kpi_history.csv', index=False)
        
        # Save simulation log
        log_df = pd.DataFrame(self.simulation_log)
        log_df.to_csv('data/simulation_results/simulation_log.csv', index=False)
        
        # Save final network stats
        network_stats = self.network.get_network_stats()
        with open('data/simulation_results/network_stats.json', 'w') as f:
            json.dump(network_stats, f, indent=2)
        
        print(f"\nSimulation results saved to data/simulation_results/")
    
    def get_simulation_summary(self) -> Dict:
        """Get summary of simulation results"""
        if not self.kpi_history:
            return {}
        
        final_kpis = self.kpi_history[-1]
        initial_kpis = self.kpi_history[0] if len(self.kpi_history) > 1 else final_kpis
        
        return {
            'simulation_duration_days': self.simulation_days,
            'total_steps': self.current_step,
            'final_total_cost': final_kpis['total_cost'],
            'final_service_level': final_kpis['service_level'],
            'total_stockouts': final_kpis['total_stockouts'],
            'network_efficiency': final_kpis['network_efficiency'],
            'cost_change': final_kpis['total_cost'] - initial_kpis.get('total_cost', 0)
        }


def run_simulation(simulation_days: int = 30) -> SeoulBikeSimulation:
    """
    Run the SeoulBike simulation
    
    Args:
        simulation_days: Number of days to simulate
        
    Returns:
        SeoulBikeSimulation: Completed simulation object
    """
    simulation = SeoulBikeSimulation(simulation_days=simulation_days)
    simulation.start_simulation()
    return simulation


if __name__ == "__main__":
    # Run a 30-day simulation
    sim = run_simulation(simulation_days=30)
    print("Simulation completed successfully!")