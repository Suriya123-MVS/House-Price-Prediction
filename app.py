import pandas as pd
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open('House_prediction_model.pkl', 'rb'))

# Load the cleaned dataset
data = pd.read_csv('Cleaned_data.csv')

# App title
st.title('🏡 Bangalore House Price Predictor')

# Form inputs
location = st.selectbox('📍 Select Location', sorted(data['location'].unique()))
sqft = st.number_input('📏 Total Area (in Sqft)', min_value=100)
bedrooms = st.number_input('🛏️ Number of Bedrooms', min_value=1, step=1)
bathrooms = st.number_input('🛁 Number of Bathrooms', min_value=1, step=1)
balcony = st.number_input('🌿 Number of Balconies', min_value=0, step=1)

# Create input DataFrame
input_df = pd.DataFrame([[location, sqft, bathrooms, balcony, bedrooms]],
                        columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

# Predict button
if st.button("🔍 Predict Price"):
    try:
        # Predict and display result
        prediction = model.predict(input_df)
        price = round(prediction[0] * 1_00_000, 2)  # converting lakhs to rupees
        st.success(f"💰 Predicted House Price: ₹ {price:,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
