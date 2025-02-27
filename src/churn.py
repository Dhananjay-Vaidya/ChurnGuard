import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from preprocess import preprocess_data
from model import train_model, save_model, load_model
from utils import load_data  
from config import Config

"""
df = pd.read_csv(r"C:/Users/dhananjay vaidya/Downloads/ProjectChurn/Churn_Modelling.csv")
df.head()

df.info()
df.isnull().sum()

a = df[df.duplicated()]
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df = pd.get_dummies(df , columns=["Geography"] , drop_first=True)

features = ["CreditScore" , "Gender" , "Age" , "Tenure" , "Balance" , "NumOfProducts" , "HasCrCard" , "IsActiveMember" , "EstimatedSalary" , "Geography_Germany" , "Geography_Spain"]
x = df[features]
y = df["Exited"]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state= 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = RandomForestClassifier(n_estimators= 160 , random_state= 42 , criterion="gini")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

conf_matrix = confusion_matrix(y_test , y_pred)
class_report = classification_report(y_test , y_pred)
accuracy = accuracy_score(y_test , y_pred)
print("Confusion Matrix :" ,conf_matrix)
print("/n")
print("Class Report :" ,class_report)
print("/n")
print("Accuracy :" ,accuracy)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]

plt.figure(figsize = (10 , 6))
plt.title("Feature Importance")
plt.barh(range(x.shape[1]) , importances[indices])
plt.yticks(range(x.shape[1]) , names)
"""

def main():
    """Main function to execute churn analysis."""
    # Load the data
    df = load_data(Config.DATA_PATH)

    # Preprocess the data
    X, y = preprocess_data(df)
    
    try:
        model = load_model()  # Try to load the saved model
    except (FileNotFoundError, EOFError):
        print("The model file is corrupted or empty. Please retrain the model.")
        # If the model doesn't exist or is corrupted, train a new one
        model = train_model(X, y)  # Now train_model is called with X and y correctly
        save_model(model)  # Save the trained model for future use

if __name__ == "__main__":
    main()
