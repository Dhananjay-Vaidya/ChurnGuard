from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import Config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from config import Config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import optuna

def optimize_xgb(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'gamma': trial.suggest_float('gamma', 0, 5, step=0.1)
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model.score(X_train, y_train)

def train_model(X, y):
    if X is None or y is None:
        raise ValueError("Features (X) and target (y) cannot be None")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_xgb(trial, X_train_scaled, y_train_resampled), n_trials=30)
    
    best_params = study.best_params
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_scaled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    print("Best Parameters:", best_params)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
    
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(model_data, Config.MODEL_PATH)
    print(f"\nModel saved to {Config.MODEL_PATH}")
    
    return model

def save_model(model, filename=None):
    """
    Save the trained model to a file.
    
    Parameters:
    model: Trained model object
    filename (str, optional): Path to save the model
    """
    if filename is None:
        filename = Config.MODEL_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def load_model(filename=None):
    """
    Load a trained model from a file.
    
    Parameters:
    filename (str, optional): Path to load the model from
    
    Returns:
    dict: Dictionary containing model and related objects
    """
    if filename is None:
        filename = Config.MODEL_PATH
        
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found at {filename}")
        
    try:
        model_data = joblib.load(filename)
        print(f'Model loaded from {filename}')
        return model_data
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

# churn.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import train_model, save_model, load_model
from config import Config

def prepare_data(df):
    """
    Prepare the data for training.
    
    Parameters:
    df (pd.DataFrame): Raw dataframe
    
    Returns:
    tuple: (X, y) prepared features and target
    """
    if df is None:
        raise ValueError("DataFrame cannot be None")

    # Drop unnecessary columns
    df = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)
    
    # Create X (features) and y (target)
    y = df['Exited']
    X = df.drop('Exited', axis=1)
    
    # Handle categorical variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    
    return X, y

def main():
    """Main function to execute churn analysis."""
    try:
        # Load the data
        df = pd.read_csv(Config.DATA_PATH)
        
        # Prepare the data
        X, y = prepare_data(df)
        
        try:
            model = load_model()
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Training new model...")
            model = train_model(X, y)
            print("Model training completed!")
        except Exception as e:
            print(f"Error during model operations: {e}")
            return
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return

if __name__ == "__main__":
    main()