#!/usr/bin/env python3
"""
Streamlit Dashboard for SeoulBike Digital Twin

This module creates the user interface for visualizing simulation results,
forecasts, optimizations, and interacting with the LLM chatbot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import sys
sys.path.append('.')
# Import your project modules
from llm_interface.chatbot import SeoulBikeChatbot
from simulation.simulation import run_simulation

class SeoulBikeDashboard:
    """Streamlit dashboard application class."""

    def __init__(self):
        self.app_title = "🚲 SeoulBike Digital Twin Dashboard"
        st.set_page_config(page_title=self.app_title, layout="wide")
        st.title(self.app_title)

        # Initialize chatbot state
        if 'chatbot' not in st.session_state:
            try:
                st.session_state.chatbot = SeoulBikeChatbot()
            except ValueError as e: # Handle missing API key
                st.session_state.chatbot = None
                st.error(f"Could not initialize chatbot: {e}. Please set your OPENAI_API_KEY.")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def load_data(self, file_path):
        """Safely load dataframes."""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None

    def display_simulation_view(self):
        """Tab for live simulation view."""
        st.header("Digital Twin Simulation")
        
        # Scenario Sliders
        st.sidebar.subheader("Simulation Scenario Parameters")
        sim_days = st.sidebar.slider("Simulation Days", 1, 90, 30)
        demand_surge = st.sidebar.slider("Demand Surge (%)", -50, 100, 0)
        
        if st.sidebar.button("Run Simulation"):
            with st.spinner(f"Running simulation for {sim_days} days..."):
                # Here you would modify simulation parameters before running
                # For this example, we'll just re-run the default simulation
                results = run_simulation(simulation_days=sim_days)
                st.session_state.simulation_results = results.get_simulation_summary()
                st.success("Simulation complete!")
        
        kpi_data = self.load_data('data/simulation_results/kpi_history.csv')
        if kpi_data is not None:
            st.subheader("Key Performance Indicators (KPIs) Over Time")

            col1, col2 = st.columns(2)
            
            # Total Cost Chart
            fig_cost = px.line(kpi_data, x='time', y='total_cost', title='Total Cost Over Time')
            col1.plotly_chart(fig_cost, use_container_width=True)
            
            # Inventory Levels
            fig_inv = px.line(kpi_data, x='time', y='total_inventory', title='Total Inventory Level Over Time')
            col2.plotly_chart(fig_inv, use_container_width=True)

            # Stockouts and Service Level
            col3, col4 = st.columns(2)
            fig_stockouts = px.line(kpi_data, x='time', y='total_stockouts', title='Total Stockouts Over Time')
            col3.plotly_chart(fig_stockouts, use_container_width=True)
            
            fig_service = px.line(kpi_data, x='time', y='service_level', title='Service Level Over Time')
            fig_service.update_layout(yaxis_range=[0,1]) # Keep service level between 0 and 1
            col4.plotly_chart(fig_service, use_container_width=True)

        else:
            st.warning("No simulation data found. Please run the simulation.")

    def display_forecast_view(self):
        """Tab for demand forecasting."""
        st.header("Demand Forecast")
        
        forecast_files = [f for f in os.listdir('data/forecast/') if f.endswith('.csv')]
        if not forecast_files:
            st.warning("No forecast data found. Please run the forecasting pipeline.")
            return

        warehouse_id = st.selectbox(
            "Select Warehouse",
            options=[int(f.split('_')[-1].split('.')[0]) for f in forecast_files],
            format_func=lambda x: f"Warehouse {x}"
        )
        
        forecast_data = self.load_data(f'data/forecast/forecast_warehouse_{warehouse_id}.csv')
        
        if forecast_data is not None:
            st.subheader(f"Predicted Demand for Warehouse {warehouse_id}")
            fig = px.line(forecast_data, x='date', y='predicted_demand', title='Demand Forecast (yhat)')
            fig.add_scatter(x=forecast_data['date'], y=forecast_data['lower_bound'], fill='tonexty', mode='lines', line_color='lightgrey', name='Confidence Interval')
            fig.add_scatter(x=forecast_data['date'], y=forecast_data['upper_bound'], fill='tonexty', mode='lines', line_color='lightgrey', name='Confidence Interval')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load forecast data for the selected warehouse.")

    def display_optimization_view(self):
        """Tab for optimization results."""
        st.header("Inventory Optimization")
        
        opt_data = self.load_data('data/optimization_results/single_warehouse_optimization.csv')
        
        if opt_data is not None:
            st.subheader("Optimization Results per Warehouse")
            st.dataframe(opt_data)

            # Display key recommendations
            report_path = 'data/optimization_results/optimization_report.json'
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                st.subheader("Key Recommendations")
                for rec in report.get('recommendations', []):
                    st.info(f"**{rec['type']} (Warehouse {rec['warehouse_id']}):** {rec['description']}. **Action:** {rec['action']}")
            
        else:
            st.warning("No optimization data found. Please run the optimization pipeline.")

    def display_chatbot_view(self):
        """Tab for LLM chatbot interaction."""
        st.header("Chat with the Digital Twin")

        if st.session_state.chatbot is None:
            st.error("Chatbot is not available. Please check your OpenAI API key.")
            return

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask about inventory, forecasts, or run scenarios..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """Main function to run the Streamlit app."""
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Forecasting", "Optimization", "Chatbot"])

        with tab1:
            self.display_simulation_view()
        
        with tab2:
            self.display_forecast_view()

        with tab3:
            self.display_optimization_view()
        
        with tab4:
            self.display_chatbot_view()

if __name__ == '__main__':
    app = SeoulBikeDashboard()
    app.run()