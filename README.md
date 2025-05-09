# Sales Performance Dashboard

## Overview

This repository contains a Streamlit dashboard application designed to visualize sales performance data across multiple dimensions using an artificial dataset. The dashboard is built with cognitive design principles and visualization best practices to optimize decision-making across different organizational contexts.

## Features

### Multi-Modal Views

The dashboard offers three distinct views tailored to different decision-making contexts:

1. **Strategic View** - For executives and high-level decision makers
   - High-level KPIs and performance metrics
   - Strategic insights highlighting key trends and exceptions
   - Revenue and profit trends with comparative analysis
   - Product and regional performance breakdowns

2. **Operational View** - For day-to-day management and operations
   - Daily revenue trends
   - Product mix analysis 
   - Regional performance comparisons
   - Day-of-week patterns and cyclical insights

3. **Analytical View** - For data analysts and deep-dive exploration
   - Correlation analysis between key metrics
   - Customizable performance breakdowns
   - Interactive forecasting with adjustable parameters
   - Detailed data tables with comprehensive metrics

### Key Visualizations

- **Performance Metric Cards** - At-a-glance KPIs with comparison indicators
- **Time Series Analysis** - Trend visualization with monthly/quarterly toggles
- **Product & Regional Performance** - Multi-dimensional bar charts encoding revenue and margin
- **Forecasting Tools** - Interactive prediction models with uncertainty visualization
- **Correlation Heatmaps** - Relationship strength analysis between metrics
- **Automated Insights** - Algorithmically generated observations from the data

## Installation

### Prerequisites

- Python 3.7+
- Pip package manager

### Setup

1. Clone this repository:
   ```
   git clone [https://github.com/samad-ahuja/Sales_Analytics_Dashboard.git]
   cd sales-dashboard
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Launch the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided in the terminal (typically http://localhost:8501...)

3. Use the mode selector at the top to switch between Strategic, Operational, and Analytical views

## Data

The dashboard uses synthetic sales data generated within the application. The data model includes:

- Products (A, B, C, D)
- Regions (North, South, East, West)
- Daily sales for one year (2023)
- Realistic seasonal patterns and weekly cycles
- Monthly special events and promotions

To use your own data, modify the `load_data()` function to load and process external data sources.

## Design Rationale

This dashboard was designed with specific cognitive and visualization principles in mind:

- **Cognitive Load Management** - Segmenting information into appropriate contexts
- **Working Memory Optimization** - Limiting information to 4-7 key items per view
- **Visual Hierarchy** - Organizing elements by importance using size, color, and position
- **Preattentive Processing** - Using color and position for instant pattern recognition
- **Decision Hierarchy** - Structuring information from high-level awareness to detailed analysis
- **Comparative Analysis** - Facilitating meaningful comparisons across dimensions
- **Focus + Context** - Maintaining orientation during detailed exploration

## Dependencies

- [Streamlit](https://streamlit.io/) - Main framework for the interactive web application
- [Pandas](https://pandas.pydata.org/) - Data processing and analysis
- [NumPy](https://numpy.org/) - Numerical computations and data generation
- [Plotly](https://plotly.com/) - Interactive data visualizations
- [scikit-learn](https://scikit-learn.org/) - Linear regression for forecasting functionality
