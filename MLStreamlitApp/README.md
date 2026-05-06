# Machine Learning Playground App

## Project Overview: 
This application was developed using the Streamlit framework to provide users with an environment for interactive graphs and modeling. By allowing users to upload their own datasets and change various  algorithms in real time, the project serves as a bridge between machine learning concepts and application. The primary objective of this project is to demonstrate the principles of applied machine learning through an interface that shows how specific model selections and hyperparameter adjustments influence the final outcomes.

## Live App
Access the application locally at https://megnash14-nash-data-science-portfolio-mlstreamlitappapp-cmzfme.streamlit.app/

## Key Features

### Data Interaction and Preprocessing
People can upload CSV files to begin their analysis without requiring manual code adjustments. The application has an interactive view of the dataset directly within the app to allow for inspection of data quality and structure. The backend features a preprocessing engine that automatically detects and converts categorical text data into numerical formats to ensure it will fit with the learn models.

### Model Selection
The app supports a diverse range of supervised learning algorithms which each offer different approaches to pattern recognition. Logistic Regression is included as a fundamental linear model used for binary and  classification tasks. The K Nearest Neighbors method provides a approach that predicts outcomes based on the proximity of data points in a multidimensional space. Finally, the Decision Tree model offers a non linear approach that maps observations about an item to conclusions through a tree like graph of decisions.

### Hyperparameter Tuning
Users can use sidebar sliders to specific model parameters such as the regularization strength in Logistic Regression or the maximum depth of a Decision Tree. The application is designed to retrain models dynamically based on user input which provides a hands on way to observe the different algorithms to parameter changes.

### Performance Analysis and Metrics
The app displays an accuracy score after every training cycle to provide a clear representation of the predictive power of the model. Users can monitor the specific sizes of their training and testing splits to understand how data affects model stability. For tree based and linear models, the app generates feature importance charts to explain which specific variables are most influential in driving the decisions of the model.

## How It Works

First, use the file uploader in the sidebar to import your own dataset for analysis. Second, select the specific target variable from your data that you want the machine learning model to predict. Third, choose from the available machine learning models to determine the approach to the problem. Fourth, adjust the metrics on the sidebar to optimize the learning behavior of the model. Fifth, click the Train Model button to initiate the splitting, cleaning, and fitting process. Sixth, review the metrics and visual charts to gain insights into the behavior and performance of the model.

## Installation and Running Locally

### 1. Clone the repository
git clone https://github.com/megnash14/MLStreamlitApp.git
MLStreamlitApp

### 2. Install dependencies
Ensure you have all necessary Python libraries installed by running the following command in your terminal:
pip install requirements.txt

### 3. Run the application
Start the Streamlit server from your terminal:
streamlit run MLStreamlitApp/app.py

### 4. Open in browser
The application should automatically open in your default browser or you can  go to http://localhost:8501