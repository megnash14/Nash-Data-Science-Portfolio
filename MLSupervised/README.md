# Happiness Predictor: Supervised Machine Learning App

This project is an interactive machine learning application built with Python and Streamlit. It allows users to explore how different social and economic factors—like GDP, social support, and life expectancy—influence a country’s overall happiness score. By shifting from unsupervised patterns to supervised predictions, this app demonstrates how regression models can be trained to forecast numerical outcomes based on historical data.

## Live Application
You can access the deployed version of the app here: **[Insert Your Streamlit Cloud Link Here]**

## How the App Works
The application is designed to be intuitive and responsive. Once a dataset is loaded, the model trains automatically as you adjust settings in the sidebar, providing instant feedback on how your changes affect the results.

1. **Data Input:** By default, the app uses the **2026 World Happiness Report** rankings. However, I have included a file uploader that allows you to test the model on any custom CSV file you might have.
2. **Data Configuration:** You can choose between two primary regression algorithms:
    * **Linear Regression:** This model is great for understanding direct, straight-line relationships between factors.
    * **Random Forest:** This is a more complex, tree-based model that is excellent at capturing non-linear patterns that a simple line might miss.
3. **Hyperparameter Tuning:** For the Random Forest model, I included sliders to adjust the "Number of Trees" and "Max Tree Depth." This gives you a hands-on look at how tuning a model's structure impacts its final performance.
4. **Performance Feedback:** The app immediately calculates the R-Squared score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to give you a quantitative look at the model's accuracy.

## Project Features and Visuals
To help make sense of the math, the app generates two key visualizations:
* **Actual vs. Predicted Plot:** This scatter plot shows how close the model's guesses were to the actual happiness scores. The closer the dots are to the red dashed line, the more accurate the model is.
* **Feature Importance Chart:** This bar chart ranks which factors (like Freedom or Corruption) had the biggest impact on the final happiness score, helping you see what truly drives happiness globally.

## Local Installation
If you would like to run this project on your own machine, follow these steps:

1. Clone this repository to your local drive.
2. Ensure you have the required libraries installed by running:
   ```bash
   pip install -r requirements.txt