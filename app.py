"""
House Price Prediction — Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ----------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    furnishing_encoder = joblib.load("furnishing_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, furnishing_encoder, feature_columns

model, scaler, furnishing_encoder, feature_columns = load_artifacts()

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "Estimate a house's market price from its size, layout, and amenities. "
    "Model trained on the Kaggle **Housing Price Prediction** dataset."
)
st.divider()

# ----------------------------------------------------------------
# Input form
# ----------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=3000, step=50)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    with col2:
        stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
        parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)
        furnishingstatus = st.selectbox(
            "Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=1
        )

    st.subheader("Amenities")
    col3, col4, col5 = st.columns(3)
    with col3:
        mainroad = st.checkbox("On Main Road", value=True)
        guestroom = st.checkbox("Guest Room")
    with col4:
        basement = st.checkbox("Basement")
        hotwaterheating = st.checkbox("Hot Water Heating")
    with col5:
        airconditioning = st.checkbox("Air Conditioning", value=True)
        prefarea = st.checkbox("Preferred Area")

    submitted = st.form_submit_button("Predict Price", use_container_width=True, type="primary")

# ----------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------
if submitted:
    binary_map = {True: 1, False: 0}

    total_rooms = bedrooms + bathrooms
    area_per_room = area / total_rooms
    amenity_score = sum([
        binary_map[mainroad], binary_map[guestroom], binary_map[basement],
        binary_map[hotwaterheating], binary_map[airconditioning], binary_map[prefarea],
    ])
    is_luxury = int(airconditioning and prefarea and furnishingstatus == "furnished")
    furnishing_encoded = furnishing_encoder.transform([furnishingstatus])[0]

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": binary_map[mainroad],
        "guestroom": binary_map[guestroom],
        "basement": binary_map[basement],
        "hotwaterheating": binary_map[hotwaterheating],
        "airconditioning": binary_map[airconditioning],
        "parking": parking,
        "prefarea": binary_map[prefarea],
        "furnishingstatus": furnishing_encoded,
        "total_rooms": total_rooms,
        "area_per_room": area_per_room,
        "amenity_score": amenity_score,
        "is_luxury": is_luxury,
    }

    input_df = pd.DataFrame([input_dict])[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.divider()
    st.subheader("💰 Estimated Price")
    st.metric(label="Predicted House Price", value=f"₹ {prediction:,.0f}")

    with st.expander("See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.divider()
st.caption(
    "Dataset: [Housing Price Prediction — Kaggle]"
    "(https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction) · "
    "Model: Tuned XGBoost Regressor"
)
