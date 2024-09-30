# Survey Data Analysis using Multiple Linear Regression

## Overview

This repository provides a complete framework for analyzing and visualizing survey data using **Multiple Linear Regression (MLR)**.

1. Performs MLR on survey data to explore relationships between different variables.
2. Creates clear and insightful visualizations of the MLR results, including distribution plots and correlation matrices.

The program is customizable and can be adapted for any survey data across industries, making it suitable for various research or organizational purposes.

## Features

- **Customizable MLR Analysis**: Easily map qualitative survey responses to numerical values for statistical analysis.
- **Data Preparation**: Includes automatic handling of missing or infinite values for a clean dataset ready for analysis.
- **Multiple Pathways**: Perform MLR on multiple dependent variables using a set of relevant independent variables to gain deeper insights.
- **Statistical Outputs**: Displays model summaries, F-tests, and significance levels for a comprehensive understanding of the results.

- **Distribution Plots**: Visualizes the frequency and distribution of responses across different survey questions.
- **Correlation Analysis**: Generates heatmaps to identify and visualize correlations between different survey variables.
- **Relationship Analysis**: Uses scatter plots and regression lines to illustrate the influence of independent variables on dependent variables.
- **Flexible Plotting Functions**: Easily adaptable plotting functions allow for a wide range of survey data to be visualized effectively.

## Getting Started

### Prerequisites
To run these scripts, ensure you have the following:
- Python 3.x
- Required libraries such as `pandas`, `numpy`, `statsmodels`, `matplotlib`, and `seaborn`.

### Installation

1. **Download or clone** the repository to your local machine.
2. **Install the necessary libraries** using:
   ```bash
   pip install -r requirements.txt
   ```

### Usage Instructions

1. **Running the Analysis Script**:
   - The script performs MLR based on the survey data. 
   - To customize, assign numerical values to your survey responses within the script.
   - Set the desired dependent and independent variables for each pathway.
   - Execute the script, and it will display model summaries and save the cleaned dataset.

2. **Running the Visualization Script**:
   - Use the cleaned dataset generated from the analysis script.
   - Customize the visualization functions to reflect your specific survey questions and options.
   - Execute the script to generate various plots and visualizations for deeper insights.

## Customization

### Adapting the Analysis Script
- Update the `text-to-numeric` mappings in the script to match the text-based survey responses.
- Modify the dependent and independent variables as needed for each regression pathway to fit your research questions.

### Modifying Visualization Functions
- Adjust the plot titles, labels, and color palettes to match your survey context.
- Use the plotting functions provided for creating visualizations, and modify parameters like figure size and layout for better clarity.

## Example Use Cases

- **Employee Engagement Surveys**: Analyze factors contributing to job satisfaction, retention, or performance.
- **Consumer Behavior Surveys**: Determine the impact of various marketing channels on purchasing decisions.
- **Market Research Surveys**: Understand preferences and perceptions of different product features.

## Outputs

The scripts will generate the following:

- **Cleaned Dataset**: The analysis script produces a cleaned version of the dataset, ready for MLR.
- **Model Summaries**: Detailed summaries and statistical significance of the regression models for each pathway.
- **Visualizations**: Distribution plots, correlation matrices, and relationship analysis charts, providing visual insights into the survey data.
