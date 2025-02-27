import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Preprocess the input DataFrame and return features and target."""
    # Drop columns that are not useful for the model
    columns_to_drop = ['CustomerId', 'RowNumber', 'Surname']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1, errors = "ignore")

    # Handle categorical features (example: Geography)
    # Assuming 'Geography' is the only necessary categorical column
    if 'Geography' in df.columns:
        le = LabelEncoder()
        df['Geography'] = le.fit_transform(df['Geography'])

    # Drop the 'Exited' column if it exists
    if 'Exited' in df.columns:
        df.drop(columns=['Exited'], inplace=True)

    # Drop the 'Churn' column if it exists and set it as the target variable
    if 'Churn' in df.columns:
        y = df['Churn']
        df.drop(columns=['Churn'], inplace=True)
    else:
        y = None  # No target variable found

    # Perform any additional preprocessing steps, like encoding or scaling
    # Example: Encoding categorical variables
    df = pd.get_dummies(df, columns = ["Gender" , "Geography"] , drop_first=True)

    # Define X (features)
    X = df

    return X, y

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def main():
    # Load your dataset
    file_path = r"C:/Users/dhananjay vaidya/Downloads/ProjectChurn/data/Churn_Modelling.csv"  # Change this to your CSV file path
    df = load_data(file_path)
    
    if df is not None:
        X, y = preprocess_data(df)
        
        # Display the shapes of X and y
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
        else:
            print("No target variable found.")

if __name__ == "__main__":
    main()
