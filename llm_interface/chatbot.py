#!/usr/bin/env python3
"""
LLM Interface with Chatbot for SeoulBike Digital Twin Project

This module provides natural language query capabilities for the SeoulBike system.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings('ignore')


class SeoulBikeChatbot:
    """AI-powered chatbot for SeoulBike digital twin system"""
    
    def __init__(self, openai_api_key: Optional[str] = None, data_path: str = 'data/'):
        """Initialize the chatbot"""
        self.data_path = data_path
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize components
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        
        # Initialize the system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the LLM system and vector database"""
        print("Initializing SeoulBike chatbot system...")
        
        try:
            # Initialize OpenAI components
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=self.openai_api_key
            )
            
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize vector database
            self.initialize_vector_database(embeddings)
            
            print("Chatbot system initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing chatbot system: {e}")
            raise
    
    def initialize_vector_database(self, embeddings):
        """Initialize ChromaDB vector database with system knowledge"""
        try:
            # Create documents from system knowledge
            documents = self.create_system_documents()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory="data/vector_db"
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True
            )
            
            print(f"Vector database initialized with {len(texts)} document chunks")
            
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            raise
    
    def create_system_documents(self):
        """Create documents containing system knowledge"""
        from langchain.schema import Document
        
        documents = []
        
        # System overview document
        system_overview = """
        SeoulBike Digital Twin System Overview:
        
        This is a comprehensive digital twin simulation system for Seoul's bike sharing infrastructure.
        The system includes:
        
        1. Data Processing: Automated cleaning and preprocessing of SeoulBike dataset
        2. Digital Twin Simulation: SimPy-based simulation of supply chain operations
        3. Demand Forecasting: Prophet-based time series forecasting
        4. Inventory Optimization: Linear programming optimization for inventory management
        5. LLM Integration: AI-powered insights and natural language queries
        
        The system models:
        - 5 suppliers with different capacities, lead times, and reliability scores
        - 3 warehouses with storage capacities and reorder points
        - Transportation network between locations
        - Customer demand patterns based on weather and seasonal factors
        
        Key metrics tracked:
        - Inventory levels and stockouts
        - Order costs and holding costs
        - Service levels and network efficiency
        - Transport costs and delivery times
        """
        
        documents.append(Document(page_content=system_overview, metadata={"type": "system_overview"}))
        
        # Query capabilities document
        query_capabilities = """
        Available Query Types:
        
        1. Inventory Status Queries:
           - "Show current inventory levels"
           - "Which warehouses are low on stock?"
           - "What's the reorder status?"
        
        2. Performance Analysis:
           - "Show KPIs for the last week"
           - "What's the service level?"
           - "How many stockouts occurred?"
        
        3. Forecasting Queries:
           - "Predict demand for next month"
           - "Show seasonal trends"
           - "What's the forecast accuracy?"
        
        4. Optimization Queries:
           - "Show cost savings opportunities"
           - "What are the optimal reorder points?"
           - "How can we reduce holding costs?"
        
        5. Simulation Scenarios:
           - "Simulate a 10% demand increase"
           - "What if lead time increases by 2 days?"
           - "Run optimization with new parameters"
        """
        
        documents.append(Document(page_content=query_capabilities, metadata={"type": "query_capabilities"}))
        
        return documents
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query"""
        try:
            print(f"Processing query: {query}")
            
            # Use RAG for queries
            response = self.qa_chain({"query": query})
            
            return {
                'answer': response['result'],
                'source_documents': [doc.page_content for doc in response['source_documents']],
                'query_type': 'general'
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'error': True
            }
    
    def chat(self, user_input: str) -> str:
        """Main chat interface"""
        try:
            result = self.process_query(user_input)
            
            if result.get('error'):
                return result['answer']
            
            return result['answer']
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Please try rephrasing your question."


def main():
    """Main function to test the chatbot"""
    try:
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize chatbot
        chatbot = SeoulBikeChatbot(openai_api_key=api_key)
        
        print("SeoulBike Chatbot initialized!")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        # Simple chat loop
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input:
                response = chatbot.chat(user_input)
                print(f"Bot: {response}")
                print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
