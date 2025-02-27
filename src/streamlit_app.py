import streamlit as st
import numpy as np
import pandas as pd
from model import load_model
from config import Config
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the trained model with caching
@st.cache_resource
def get_model():
    """Load the trained model and related data"""
    try:
        model_data = load_model()
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_churn(model_data, features_df):
    """
    Make prediction using the loaded model
    
    Parameters:
    model_data (dict): Dictionary containing model and related objects
    features_df (pd.DataFrame): DataFrame containing features
    
    Returns:
    prediction, probability
    """
    # Get model components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Ensure features are in correct order
    features_df = features_df[feature_names]
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
    
    # Load the model data
    model_data = get_model()
    
    if model_data is None:
        st.error("Failed to load the model. Please ensure the model file exists and is valid.")
        return

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict", "Visualization", "About"])

    if page == "Home":
        st.title("🏦 Customer Churn Prediction System")
        st.write("""
        ## Welcome to the Customer Churn Predictor
        
        This application helps predict whether a customer is likely to leave your bank based on various factors.
        Use the sidebar to navigate to the prediction page and input customer details.
        
        ### Features Used:
        - Credit Score
        - Geography
        - Gender
        - Age
        - Tenure
        - Balance
        - Number of Products
        - Credit Card Status
        - Active Membership Status
        - Estimated Salary
        """)

    elif page == "Predict":
        st.title("🔮 Churn Prediction")
        st.write("Enter customer details for prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            age = st.slider("Age", 18, 100, 35)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        
        with col2:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
            salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 50000.0)
        
        features = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': salary,
            'Gender_Male': 1 if gender == "Male" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }
        
        features_df = pd.DataFrame([features])
        
        if st.button("Predict Churn Probability"):
            try:
                prediction, probability = predict_churn(model_data, features_df)
                churn_prob = probability[1]
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("### Prediction Result")
                    if prediction == 1:
                        st.error("⚠️ Customer is likely to churn!")
                    else:
                        st.success("✅ Customer is likely to stay!")
                
                with result_col2:
                    st.markdown("### Churn Probability")
                    st.progress(churn_prob)
                    st.write(f"Probability: {churn_prob:.2%}")
                
                if hasattr(model_data['model'], 'feature_importances_'):
                    st.markdown("### Feature Importance")
                    importances = pd.DataFrame({
                        'Feature': model_data['feature_names'],
                        'Importance': model_data['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(importances.set_index('Feature'))
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    elif page == "Visualization":
        st.title("📊 Data Insights")
        st.write("Explore visual insights from the dataset.")
        df = pd.read_csv(Config.DATA_PATH)
        
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Exited', data=df, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    elif page == "About":
        st.title("ℹ️ About")
        st.write("""
        ### Customer Churn Prediction System
        
        This application uses machine learning to predict customer churn in banking. 
        The model is trained on historical customer data and uses various features 
        to predict the likelihood of a customer leaving the bank.
        
        #### Model Details:
        - Algorithm: Random Forest Classifier
        - Features: Customer demographics and banking behavior
        - Performance metrics available in the prediction page
        
        #### How to Use:
        1. Navigate to the Predict page
        2. Enter customer information
        3. Click "Predict Churn Probability"
        4. View the prediction results and probability
        """)

if __name__ == "__main__":
    main()