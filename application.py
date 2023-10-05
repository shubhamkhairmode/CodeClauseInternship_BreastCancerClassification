import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.title('Breast Cancer Classifier')

# Sidebar with input fields
st.sidebar.header('Input Features')

# Define features_name based on your model's feature names
features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                 'mean smoothness', 'mean compactness', 'mean concavity',
                 'mean concave points', 'mean symmetry', 'mean fractal dimension',
                 'radius error', 'texture error', 'perimeter error', 'area error',
                 'smoothness error', 'compactness error', 'concavity error',
                 'concave points error', 'symmetry error', 'fractal dimension error',
                 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                 'worst smoothness', 'worst compactness', 'worst concavity',
                 'worst concave points', 'worst symmetry', 'worst fractal dimension']

input_features = []

for feature_name in features_name:
    value = st.sidebar.slider(f"Select {feature_name}", float(0.0), float(100.0))
    input_features.append(value)

# Function to predict
def predict_cancer(input_features):
    df = pd.DataFrame([input_features], columns=features_name)
    output = model.predict(df)

    if output == 0:
        return "Breast Cancer Detected"
    else:
        return "No Breast Cancer Detected"

# Predict and display the result
if st.sidebar.button('Predict'):
    prediction = predict_cancer(input_features)
    st.write(f'Prediction: Patient has {prediction}')
