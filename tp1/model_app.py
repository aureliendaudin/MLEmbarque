import streamlit as st
import joblib
import numpy as np

# Load the model from the file
model = joblib.load("regression.joblib")

# Create the app title
st.title("House Price Prediction")

# Create input fields
size = st.number_input("Size of the house (sq ft)", min_value=0.0, step=10.0)
bedrooms = st.number_input("Number of bedrooms", min_value=0, step=1)
has_garden = st.number_input("Has garden (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)

# Make prediction when the user submits the inputs
if st.button("Predict Price"):
    # Prepare the input data for prediction
    features = np.array([[size, bedrooms, has_garden]])
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Display the prediction
    st.write(f"The predicted house price is: ${prediction[0]:.2f}")