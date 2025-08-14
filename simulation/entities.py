#!/usr/bin/env python3
"""
Simulation Entities for SeoulBike Digital Twin Project

This module defines the core entities used in the digital twin simulation:
- Supplier: Manages bike supply with lead times and capacity constraints
- Warehouse: Handles inventory management and reorder logic
- Transport: Manages logistics and transportation between entities
"""

from __future__ import annotations  # <-- ADD THIS LINE AT THE TOP
import simpy
from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class Supplier:
    """Supplier entity representing bike manufacturers/suppliers"""
    
    supplier_id: int
    name: str
    location: str
    capacity_per_day: int
    reliability_score: float
    lead_time_days: int
    cost_per_unit: float
    env: simpy.Environment
    
    def __post_init__(self):
        """Initialize supplier state"""
        self.current_orders = []
        self.total_units_supplied = 0
        self.total_cost = 0.0
        self.lead_time_variability = 0.2  # 20% variability in lead time
        
    def calculate_actual_lead_time(self) -> int:
        """Calculate actual lead time with variability"""
        base_lead_time = self.lead_time_days
        variability = random.uniform(-self.lead_time_variability, self.lead_time_variability)
        actual_lead_time = max(1, int(base_lead_time * (1 + variability)))
        return actual_lead_time
    
    def can_fulfill_order(self, quantity: int) -> bool:
        """Check if supplier can fulfill an order"""
        return quantity <= self.capacity_per_day
    
    def process_order(self, order_id: str, quantity: int, warehouse_id: int) -> Dict:
        """Process a new order and return order details"""
        if not self.can_fulfill_order(quantity):
            return {
                'order_id': order_id,
                'status': 'rejected',
                'reason': 'Insufficient capacity',
                'quantity': quantity,
                'warehouse_id': warehouse_id
            }
        
        actual_lead_time = self.calculate_actual_lead_time()
        order = {
            'order_id': order_id,
            'quantity': quantity,
            'warehouse_id': warehouse_id,
            'order_time': self.env.now,
            'expected_delivery': self.env.now + actual_lead_time,
            'status': 'processing',
            'supplier_id': self.supplier_id
        }
        
        self.current_orders.append(order)
        self.total_cost += quantity * self.cost_per_unit
        
        return order
    
    def get_supplier_stats(self) -> Dict:
        """Get current supplier statistics"""
        return {
            'supplier_id': self.supplier_id,
            'name': self.name,
            'total_units_supplied': self.total_units_supplied,
            'total_cost': self.total_cost,
            'active_orders': len(self.current_orders),
            'reliability_score': self.reliability_score
        }


@dataclass
class Warehouse:
    """Warehouse entity managing bike inventory"""
    
    warehouse_id: int
    name: str
    location: str
    storage_capacity: int
    initial_inventory: int
    reorder_point: int
    reorder_quantity: int
    holding_cost_per_unit: float
    env: simpy.Environment
    network: SupplyChainNetwork # This was our previous fix

    def __post_init__(self):
        """Initialize warehouse state"""
        self.current_inventory = self.initial_inventory
        self.orders_in_transit = []
        self.total_orders_placed = 0
        self.total_stockouts = 0
        self.total_holding_cost = 0.0
        self.inventory_history = []
        self.daily_demand = 0
        
    def update_inventory(self, quantity: int, operation: str = 'add'):
        """Update warehouse inventory"""
        if operation == 'add':
            self.current_inventory = min(self.storage_capacity, self.current_inventory + quantity)
        elif operation == 'subtract':
            if self.current_inventory >= quantity:
                self.current_inventory -= quantity
            else:
                shortfall = quantity - self.current_inventory
                self.total_stockouts += shortfall
                self.current_inventory = 0
                
        self.inventory_history.append({
            'time': self.env.now,
            'inventory': self.current_inventory,
            'operation': operation,
            'quantity': quantity
        })
        
        self.total_holding_cost += self.current_inventory * self.holding_cost_per_unit / 365

    def check_reorder_needed(self) -> bool:
        """Check if reorder is needed based on reorder point"""
        effective_inventory = self.current_inventory + sum(o['quantity'] for o in self.orders_in_transit)
        return effective_inventory <= self.reorder_point
    
    def place_reorder(self, supplier: Supplier) -> Optional[Dict]:
        """Place a reorder with a given supplier"""
        order_id = f"ORDER_{self.warehouse_id}_{self.total_orders_placed + 1}"
        order = supplier.process_order(order_id, self.reorder_quantity, self.warehouse_id)
        
        if order and order['status'] == 'processing':
            self.orders_in_transit.append(order)
            self.total_orders_placed += 1
        
        return order
    
    def receive_delivery(self, order_id: str):
        """Receive delivery and update inventory"""
        order_to_remove = None
        for order in self.orders_in_transit:
            if order['order_id'] == order_id:
                supplier = self.network.suppliers.get(order['supplier_id'])
                if not supplier:
                    print(f"Error: Could not find supplier {order['supplier_id']}")
                    order['status'] = 'failed'
                    order['reason'] = 'Supplier not found in network'
                    order_to_remove = order
                    break

                if random.random() < supplier.reliability_score:
                    self.update_inventory(order['quantity'], 'add')
                    order['status'] = 'delivered'
                    supplier.total_units_supplied += order['quantity']
                else:
                    order['status'] = 'failed'
                    order['reason'] = 'Supplier reliability issue'
                
                order_to_remove = order
                break

        if order_to_remove:
            self.orders_in_transit.remove(order_to_remove)
    
    def get_warehouse_stats(self) -> Dict:
        """Get current warehouse statistics"""
        return {
            'warehouse_id': self.warehouse_id,
            'name': self.name,
            'current_inventory': self.current_inventory,
            'storage_capacity': self.storage_capacity,
            'utilization_rate': (self.current_inventory / self.storage_capacity) if self.storage_capacity > 0 else 0,
            'orders_in_transit': len(self.orders_in_transit),
            'total_orders_placed': self.total_orders_placed,
            'total_stockouts': self.total_stockouts,
            'total_holding_cost': self.total_holding_cost
        }


@dataclass
class Transport:
    """Transport entity managing logistics between locations"""
    
    route_id: int
    from_location: str
    to_location: str
    distance_km: float
    avg_travel_time_hours: float
    cost_per_km: float
    capacity: int
    env: simpy.Environment
    
    def __post_init__(self):
        """Initialize transport state"""
        self.current_shipments = []
        self.total_shipments = 0
        self.total_cost = 0.0
        self.travel_time_variability = 0.15  # 15% variability in travel time
        
    def calculate_actual_travel_time(self) -> float:
        """Calculate actual travel time with variability"""
        base_time = self.avg_travel_time_hours
        variability = random.uniform(-self.travel_time_variability, self.travel_time_variability)
        actual_time = max(0.5, base_time * (1 + variability))
        return actual_time
    
    def start_shipment(self, shipment_id: str, quantity: int, priority: str = 'normal') -> Dict:
        """Start a new shipment"""
        if quantity > self.capacity:
            return {
                'shipment_id': shipment_id,
                'status': 'rejected',
                'reason': 'Exceeds transport capacity',
                'quantity': quantity
            }
        
        actual_travel_time = self.calculate_actual_travel_time()
        shipment = {
            'shipment_id': shipment_id,
            'quantity': quantity,
            'priority': priority,
            'start_time': self.env.now,
            'expected_arrival': self.env.now + actual_travel_time,
            'status': 'in_transit',
            'route_id': self.route_id
        }
        
        self.current_shipments.append(shipment)
        self.total_shipments += 1
        self.total_cost += self.distance_km * self.cost_per_km
        
        return shipment
    
    def complete_shipment(self, shipment_id: str):
        """Complete a shipment when it reaches destination"""
        for i, shipment in enumerate(self.current_shipments):
            if shipment['shipment_id'] == shipment_id and shipment['status'] == 'in_transit':
                if self.env.now >= shipment['expected_arrival']:
                    shipment['status'] = 'completed'
                    shipment['actual_arrival'] = self.env.now
                    # Remove from active shipments
                    self.current_shipments.pop(i)
                    break
    
    def get_transport_stats(self) -> Dict:
        """Get current transport statistics"""
        return {
            'route_id': self.route_id,
            'from_location': self.from_location,
            'to_location': self.to_location,
            'active_shipments': len(self.current_shipments),
            'total_shipments': self.total_shipments,
            'total_cost': self.total_cost,
            'capacity_utilization': sum(s['quantity'] for s in self.current_shipments) / self.capacity if self.capacity > 0 else 0
        }


class SupplyChainNetwork:
    """Manages the overall supply chain network and relationships"""
    
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.suppliers: Dict[int, Supplier] = {}
        self.warehouses: Dict[int, Warehouse] = {}
        self.transport_routes: Dict[int, Transport] = {}
        self.network_stats = {
            'total_inventory': 0,
            'total_orders': 0,
            'total_cost': 0.0,
            'network_efficiency': 0.0
        }
    
    def add_supplier(self, supplier: Supplier):
        """Add a supplier to the network"""
        self.suppliers[supplier.supplier_id] = supplier
    
    def add_warehouse(self, warehouse: Warehouse):
        """Add a warehouse to the network"""
        self.warehouses[warehouse.warehouse_id] = warehouse
    
    def add_transport_route(self, transport: Transport):
        """Add a transport route to the network"""
        self.transport_routes[transport.route_id] = transport
    
    def get_network_stats(self) -> Dict:
        """Get overall network statistics"""
        total_inventory = sum(w.current_inventory for w in self.warehouses.values())
        total_orders = sum(w.total_orders_placed for w in self.warehouses.values())
        total_cost = sum(s.total_cost for s in self.suppliers.values()) + \
                     sum(w.total_holding_cost for w in self.warehouses.values()) + \
                     sum(t.total_cost for t in self.transport_routes.values())
        
        # Calculate network efficiency (inventory turnover)
        total_demand = sum(w.daily_demand for w in self.warehouses.values())
        network_efficiency = total_demand / max(total_inventory, 1)
        
        self.network_stats.update({
            'total_inventory': total_inventory,
            'total_orders': total_orders,
            'total_cost': total_cost,
            'network_efficiency': network_efficiency
        })
        
        return self.network_stats