import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Gender Predictor", layout="centered")

st.title("🔍 Gender Prediction from CSV")

# File uploader
file = st.file_uploader("Upload CSV file with columns: name, age, company, gender", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Basic preview
    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df['gender_encoded'] = label_encoder.fit_transform(df['gender'])

    # Features and labels
    X = df[['name', 'age', 'company']]
    y = df['gender_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    categorical_features = ['name', 'company']
    numeric_features = ['age']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])

    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Train model
    model.fit(X_train, y_train)

   # Predict on full dataset
    full_pred = model.predict(X)
    df['predicted_gender'] = label_encoder.inverse_transform(full_pred)
    df['actual_gender'] = label_encoder.inverse_transform(y)

    # Full dataset accuracy
    full_accuracy = accuracy_score(df['actual_gender'], df['predicted_gender'])
    st.subheader("📊 Full Dataset Accuracy")
    st.info(f"Full Dataset Accuracy: {full_accuracy * 100:.2f}%")

    # Show comparison table
    st.subheader("🔍 Actual vs Predicted Gender (Sample)")
    st.dataframe(df[['name', 'age', 'company', 'actual_gender', 'predicted_gender']].head(10))

    # Download prediction
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Predictions CSV", csv, "predicted_gender.csv", "text/csv")