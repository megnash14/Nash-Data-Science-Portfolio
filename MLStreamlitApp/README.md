# Machine Learning Playground App

## Project Overview
This project is an application built using Streamlit. It allows users to upload their own dataset, select a supervised learning model, tune hyperparameters, and evaluate live model performance.

The purpose of this project is to demonstrate applied machine learning concepts while creating an  app that enables users to explore how model choices and parameters impact results.

## Live App
http://localhost:8501

## Key Features

### Data Interaction
- Upload custom CSV datasets
- Preview the dataset within the app
- View summary statistics for quick understanding of the data

### Model Selection
The app supports multiple supervised learning models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

### Hyperparameter Tuning
- Adjust model parameters using interactive sliders
- Retrain models dynamically based on user input
- Explore how parameter changes affect model performance

### Performance Metrics
- Accuracy score displayed clearly
- Training and test dataset sizes shown
- Immediate feedback after model training

### User Interface Design
- Sidebar layout for user inputs and controls
- Structured main display for results and data
- Organized sections for clarity and usability
- Responsive column layout for presenting outputs

## How It Works

1. Upload a dataset using the sidebar
2. Select a target variable (the column to predict)
3. Choose a machine learning model
4. Adjust hyperparameters as desired
5. Click "Train Model"
6. View model performance and summary outputs

## Installation and Running Locally

### 1. Clone the repository
git clone https://github.com/megnash14/MLStreamlitApp.git
cd MLStreamlitApp

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the application
streamlit run app.py

### 4. Open in browser
Navigate to:
http://localhost:8501