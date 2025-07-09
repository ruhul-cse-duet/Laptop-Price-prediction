import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')


# Load trained model and dataset
pipe = pickle.load(open('pipe_model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


# Title and description
st.title("Laptop Price Predictor")
st.markdown("Developed by [**Ruhul Amin**](https://www.linkedin.com/in/ruhul-duet-cse/)")
st.markdown("Predicts the price of a laptop based on its specifications.")

# Input fields
brand = st.selectbox('Brand', sorted(df['company'].unique()))
laptop_type = st.selectbox('Laptop Type', sorted(df['type'].unique()))
screen_size = st.number_input('Screen Size (Inches)', min_value=10.1, max_value=18.4, value=15.6, step=0.1)
screen_resolution = st.selectbox('Screen Resolution', sorted(df['screen_resolution'].unique()))
ips_panel = st.selectbox('IPS Panel', sorted(df['ips'].unique()))
cpu = st.selectbox('CPU', sorted(df['cpu'].unique()))
ram = st.selectbox('RAM (GB)', sorted(df['ram'].unique()))
ram = int(ram)
ssd = st.selectbox('SSD (GB)', sorted(df['ssd'].unique()))
ssd = int(ssd)
hdd = st.selectbox('HDD (GB)', sorted(df['hdd'].unique()))
hdd = int(hdd)
gpu = st.selectbox('GPU', sorted(df['gpu'].unique()))
touchscreen = st.selectbox('Touchscreen', sorted(df['touchscreen'].unique()))
operating_system = st.selectbox('Operating System', sorted(df['os'].unique()))
weight = st.number_input('Weight (Kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)

# Currency selection
currency = st.selectbox("Select Currency", ["INR", "USD", "EUR", "BDT"])

# Editable exchange rate section
st.markdown("### Set Exchange Rates (per 1 INR) if you want or just ignore. Model Trained with INR price")
usd_rate = st.number_input("USD Rate", min_value=0.0001, value=0.0120, step=0.0001, format="%.4f")
eur_rate = st.number_input("EUR Rate", min_value=0.0001, value=0.0110, step=0.0001, format="%.4f")
bdt_rate = st.number_input("BDT Rate", min_value=0.01, value=1.29, step=0.01, format="%.2f")


# Predict button
if st.button('Predict Laptop Price'):
    input_dict = {
        'company': brand,
        'type': laptop_type,
        'screen_size': screen_size,
        'screen_resolution': screen_resolution,
        'ips': ips_panel,
        'cpu': cpu,
        'ram': ram,
        'ssd': ssd,
        'hdd': hdd,
        'gpu': gpu,
        'touchscreen': touchscreen,
        'os': operating_system,
        'weight': weight
    }

    query_df = pd.DataFrame([input_dict])
    predicted_log_price = pipe.predict(query_df)[0]
    predicted_price_inr = np.exp(predicted_log_price)

    # Currency conversion
    if currency == "INR":
        st.success(f"The predicted price of this configuration is {predicted_price_inr:,.2f} INR")
    elif currency == "USD":
        price_usd = predicted_price_inr * usd_rate
        st.success(f"The predicted price of this configuration is ${price_usd:,.2f} USD")
    elif currency == "EUR":
        price_eur = predicted_price_inr * eur_rate
        st.success(f"The predicted price of this configuration is {price_eur:,.2f} EUR")
    elif currency == "BDT":
        price_bdt = predicted_price_inr * bdt_rate
        st.success(f"The predicted price of this configuration is {price_bdt:,.2f} BDT")

    # To display the rates used
    st.markdown("#### Exchange Rates Used:")
    st.info(f"- 1 INR = {usd_rate:.4f} USD\n"
            f"- 1 INR = {eur_rate:.4f} EUR\n"
            f"- 1 INR = {bdt_rate:.2f} BDT")
