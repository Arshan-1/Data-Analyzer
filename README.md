# Data-Analyzer

`Data-Analyzer` is a Streamlit-based data analysis application that allows users to upload datasets, clean data, analyze patterns, build simple data models, and generate interactive visualizations from a single interface.

The goal of this project is to provide an end-to-end workflow for tabular data without requiring users to write code.

## Features

- Upload tabular datasets in multiple formats: CSV, Excel (`.xls`, `.xlsx`), JSON, and Parquet.
- Automatically clean data:
  - Normalize common missing-value tokens (`na`, `null`, `-`, etc.)
  - Remove duplicate rows
  - Infer numeric and datetime columns
  - Fill missing values (median for numeric, mode for non-numeric)
- Analyze data:
  - Dataset overview (rows, columns, missing values, duplicates)
  - Column data types
  - Descriptive statistics
- Model transformed data for final charts:
  - Aggregate by category
  - Time-trend modeling (resampling by day/week/month/quarter/year)
- Visualize with chart selection + axis mapping:
  - Histogram
  - Box Plot
  - Bar Chart
  - Line Chart
  - Scatter Plot
  - Correlation Heatmap
- Use an auto-suggestion option to recommend chart type and columns based on dataset structure.

## Project Workflow

1. Upload a dataset through the Streamlit UI.
2. The app reads the file and selects valid data (including non-empty sheet detection for Excel files).
3. Data cleaning runs automatically.
4. Cleaned data is summarized in overview and analysis tabs.
5. Optional data modeling transforms the cleaned data into chart-ready format.
6. Final visualization is created using selected chart type and X/Y column choices, or via auto-suggestion.

## Libraries Used and Why

- `streamlit`  
  Builds the web UI for file upload, tabs, controls, metrics, and charts.

- `pandas`  
  Core data handling library used for reading files, cleaning, type conversion, grouping, aggregation, and statistics.

- `numpy`  
  Supports numeric operations and dtype-based processing logic.

- `plotly`  
  Generates interactive visualizations such as histogram, box plot, scatter plot, bar chart, and heatmap.

- `matplotlib` and `seaborn`  
  Included for extended plotting support and future custom visual analysis.

- `scikit-learn`  
  Included as a base dependency for future ML/modeling extensions.

- `openpyxl`  
  Required to read Excel `.xlsx` files.

- `xlrd`  
  Required to read Excel `.xls` files.

## Run the Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run app/ui.py
```

3. Open the local URL shown in terminal (usually `http://localhost:8501`).

## Current Structure

- `app/ui.py` - Streamlit interface and workflow controls.
- `app/tools.py` - Data loading, cleaning, profiling, modeling, and visualization helper functions.
- `app/agent.py` - Basic agent-style orchestration logic for CLI flow.
- `app/main.py` - Command-line smoke script.
- `data/` - Sample datasets for testing.
- `requirements.txt` - Python dependencies.

## Notes

- The app is designed for **tabular datasets**.
- "Any type of data" in this project means common tabular formats, not unstructured formats like images/audio/PDF text extraction.
