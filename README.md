Customer Churn Prediction System


![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Last Commit](https://img.shields.io/github/last-commit/Dhananjay-Vaidya/ChurnGuard)
![Issues](https://img.shields.io/github/issues/Dhananjay-Vaidya/ChurnGuard)
![Forks](https://img.shields.io/github/forks/Dhananjay-Vaidya/ChurnGuard?style=social)
![Stars](https://img.shields.io/github/stars/Dhananjay-Vaidya/ChurnGuard?style=social)
![Contributions](https://img.shields.io/badge/contributions-welcome-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)


Table of Contents
Project Description
Motivation
Problem Statement
Technologies Used
Current Features
Installation
Usage
Dataset
Contributing
Author
License
Project Description
The Customer Churn Prediction System is a machine learning-based application designed to predict whether a customer is likely to leave a service or remain. The system is built using XGBoost with advanced hyperparameter tuning and class imbalance handling to improve accuracy. The project also includes a web-based interface built with Streamlit for user-friendly interactions.

Motivation
Customer churn is a significant concern for businesses, as retaining customers is often more cost-effective than acquiring new ones. This project aims to help businesses identify at-risk customers early, allowing them to take proactive measures to retain them.

Problem Statement
The goal of this project is to build a predictive model that can classify customers as likely to churn or stay. The model analyzes various factors, including customer demographics, account details, and transaction history, to make an informed prediction.

Technologies Used
This project is developed using the following technologies:

Python – Primary programming language for the project
XGBoost – Machine learning model for prediction
Scikit-learn – Data preprocessing and evaluation
Optuna – Hyperparameter tuning for optimized performance
SMOTE (Synthetic Minority Over-sampling Technique) – Handles class imbalance
Pandas & NumPy – Data handling and processing
Matplotlib & Seaborn – Data visualization
Streamlit – Web-based interface for making predictions
Joblib – Model persistence and storage
Current Features
Data preprocessing with categorical encoding and feature scaling
Hyperparameter tuning using Optuna for improved model performance
Class imbalance handling using SMOTE
Web-based prediction system using Streamlit
Feature importance visualization for better insights
Model performance evaluation with precision, recall, and accuracy scores

Installation
To install and run this project, follow these steps:

Clone the repository:

git clone : https://github.com/Dhananjay-Vaidya/ChurnGuard.git
cd customer-churn-prediction


Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate


Install the required dependencies:

pip install -r requirements.txt


Usage
Training the Model
To train the machine learning model and save it for predictions, run:

python src/churn.py


Running the Web Application
To launch the Streamlit web app for making predictions, execute:

streamlit run app/streamlit_app.py

This will open an interactive UI where users can enter customer details and get churn predictions.


Dataset
The project uses the Churn Modelling Dataset, which contains customer demographics, financial behavior, and account details. The dataset is located in:

data/Churn_Modelling.csv


Contributing
Contributions are welcome. To contribute:

Fork the repository
Create a new branch
Make your changes and commit them
Submit a pull request for review
Author
Developed by Dhananjay Vaidya


License
This project is licensed under the MIT License.