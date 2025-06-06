import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read dataset
@st.cache_data
def load_data():
    return pd.read_csv("austin_final_final.csv")

data = load_data()
X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches'].values.reshape(-1, 1)
model = LinearRegression().fit(X, Y)

st.title("🌧️ Weather Prediction using Time Series Analysis")

st.header("Enter Weather Attributes")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    TempHighF = st.number_input("Temp High (°F)", value=80.0)
    TempAvgF = st.number_input("Temp Avg (°F)", value=70.0)
    TempLowF = st.number_input("Temp Low (°F)", value=60.0)
    DewPointHighF = st.number_input("Dew Point High (°F)", value=65.0)
    DewPointAvgF = st.number_input("Dew Point Avg (°F)", value=55.0)
    DewPointLowF = st.number_input("Dew Point Low (°F)", value=45.0)

with col2:
    HumidityHighPercent = st.number_input("Humidity High (%)", value=90.0)
    HumidityAvgPercent = st.number_input("Humidity Avg (%)", value=70.0)
    HumidityLowPercent = st.number_input("Humidity Low (%)", value=50.0)
    SeaLevelPressureAvgInches = st.number_input("Sea Level Pressure (in)", value=30.0)
    VisibilityHighMiles = st.number_input("Visibility High (miles)", value=10.0)
    VisibilityAvgMiles = st.number_input("Visibility Avg (miles)", value=8.0)

with col3:
    VisibilityLowMiles = st.number_input("Visibility Low (miles)", value=5.0)
    WindHighMPH = st.number_input("Wind High (MPH)", value=20.0)
    WindAvgMPH = st.number_input("Wind Avg (MPH)", value=10.0)
    WindGustMPH = st.number_input("Wind Gust (MPH)", value=25.0)

# Prediction
if st.button("Predict Precipitation"):
    # Create input array
    sample = np.array([
        TempHighF, TempAvgF, TempLowF, DewPointHighF, DewPointAvgF, DewPointLowF,
        HumidityHighPercent, HumidityAvgPercent, HumidityLowPercent, SeaLevelPressureAvgInches,
        VisibilityHighMiles, VisibilityAvgMiles, VisibilityLowMiles,
        0,  # placeholder column
        WindHighMPH, WindAvgMPH, WindGustMPH
    ]).reshape(1, -1).astype(np.float64)

    prediction = model.predict(sample)[0][0]
    st.subheader(f"Predicted Precipitation: **{prediction:.3f} inches**")

    if prediction > 0.5:
        st.success("🌧️ High Rainfall to be Expected")
    elif 0.3 < prediction <= 0.5:
        st.warning("🌦️ Moderate Rainfall to be Expected")
    else:
        st.info("🌤️ Low Rainfall to be Expected")

    # Plot 1: Precipitation over days
    st.subheader("📈 Precipitation Trend")
    days = np.arange(len(Y))
    fig1, ax1 = plt.subplots()
    ax1.scatter(days, Y, color='b', label='Precipitation')
    ax1.scatter(400, Y[400], color='r', label='Day 400')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Precipitation (inches)")
    ax1.set_title("Precipitation Over Time")
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2: Feature-wise trends
    st.subheader("📊 Selected Attribute Trends")
    x_vis = X[[
        'TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
        'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'
    ]]
    fig2, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, column in enumerate(x_vis.columns):
        axs[i].scatter(days[:100], x_vis[column][:100], color='r')
        axs[i].scatter(400, x_vis[column][400], color='b')
        axs[i].set_title(column)
    plt.tight_layout()
    st.pyplot(fig2)
