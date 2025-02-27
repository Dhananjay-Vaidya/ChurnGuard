import os

class Config:
    DATA_PATH = r"C:/Users/dhananjay vaidya/Downloads/ProjectChurn/data/Churn_Modelling.csv"  # Path to your dataset
    model_dir = r"C:/Users/dhananjay vaidya/Downloads/ProjectChurn/models" # Directory to save the model

    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    MODEL_PATH = os.path.join(model_dir, "saved_model.pkl")  # Full path to save your trained model

    FEATURES = [
        "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", 
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography_Germany", 
        "Geography_Spain"]