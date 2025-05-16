import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Data Preprocessing ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    print("Columns found in CSV:", df.columns.tolist())

    required_columns = ['Accident_Severity', 'Weather_Conditions', 'Road_Type', 'Time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df.dropna(subset=required_columns, inplace=True)

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    df.dropna(subset=['Time'], inplace=True)
    df['Hour'] = df['Time'].dt.hour
    df.drop('Time', axis=1, inplace=True)

    if 'Accident_Type' in df.columns:
        df.dropna(subset=['Accident_Type'], inplace=True)
    else:
        df['Accident_Type'] = 'Unknown'

    return df

# --- Feature Engineering ---
def encode_features(df, encoder=None):
    categorical_columns = ['Weather_Conditions', 'Road_Type', 'Accident_Type']
    if encoder is None:
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
        return df, encoder
    else:
        df[categorical_columns] = encoder.transform(df[categorical_columns])
        return df

# --- Model Training ---
def train_model(df):
    X = df.drop(['Accident_Severity'], axis=1)
    y = df['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/accident_model.pkl')
    return model

# --- Model Evaluation ---
def evaluate_model(test_data_path, encoder):
    model = joblib.load('models/accident_model.pkl')
    test_df = load_and_clean_data(test_data_path)
    test_df = encode_features(test_df, encoder)
    X_test = test_df.drop(['Accident_Severity'], axis=1)
    y_test = test_df['Accident_Severity']

    y_pred = model.predict(X_test)
    print(f"\nEvaluation Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- Main Pipeline ---
def main():
    file_path = "ai.csv"

    df = load_and_clean_data(file_path)
    df, encoder = encode_features(df)
    train_model(df)
    evaluate_model(file_path, encoder)

if __name__ == '__main__':
    main()
