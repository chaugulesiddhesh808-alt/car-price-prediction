import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# =====================================================
# Custom class (REQUIRED for pickle loading)
# =====================================================

class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.18):
        self.threshold = threshold

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        corr = X_df.apply(lambda col: np.corrcoef(col, y)[0, 1])
        self.selected_features_ = corr[corr.abs() >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.selected_features_].values


# =====================================================
# Load model & metadata
# =====================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_pipeline.pkl")
    input_columns = joblib.load("input_columns.pkl")
    return model, input_columns


model, INPUT_COLUMNS = load_artifacts()


# =====================================================
# UI
# =====================================================

st.title("🚗 Car Price Prediction")
st.write("Provide car details to estimate the price")


# -----------------------------
# Dropdown values (RAW)
# -----------------------------

brand_list = [
    'honda','bmw','lexus','hyundai','toyota','kia','nissan','audi',
    'chevrolet','ford','mercedes','porsche','infiniti','jaguar',
    'cadillac','land rover','jeep','volkswagen','maserati','subaru',
    'dodge','mazda','chrysler','aston martin','ferrari','bentley',
    'rolls royce','mclaren','lincoln','alfa romeo','volvo','mini',
    'fiat','acura','genesis','buick','gmc','lotus','suzuki','mg',
    'skoda','jac','changan'
]

body_list = ['sedan','suv','hatchback','coupe','convertible','wagon','truck','van','unknown']
fuel_list = ['petrol','diesel','hybrid','unknown']
gearbox_list = ['automatic','manual','cvt']
drive_list = ['front','rear','all','unknown']
yes_no = ['yes', 'no']


# -----------------------------
# Input form
# -----------------------------

with st.form("car_form"):

    st.subheader("Basic Info")

    brand = st.selectbox("Brand", brand_list)
    body = st.selectbox("Body Type", body_list)
    fuel = st.selectbox("Fuel Type", fuel_list)
    gearbox = st.selectbox("Gearbox Type", gearbox_list)
    drivetrain = st.selectbox("Drivetrain", drive_list)

    st.subheader("Engine & Performance")

    power = st.number_input("Power (hp)", 103.0, 986.0, 150.0)
    displacement = st.number_input("Displacement (L)", 1.2, 9.8, 2.0)
    torque = st.number_input("Torque (lb-ft)", 99.0, 1034.0, 200.0)
    cylinders = st.number_input("Cylinders", 3, 12, 4)

    mpg_city = st.number_input("MPG City", 16.0, 55.0, 25.0)
    mpg_highway = st.number_input("MPG Highway", 10.0, 58.0, 30.0)

    st.subheader("Size")

    seats = st.number_input("Seats", 2, 12, 5)
    doors = st.number_input("Doors", 2, 5, 4)
    height = st.number_input("Height (in)", 46.7, 79.6, 65.0)
    length = st.number_input("Length (in)", 142.0, 254.4, 180.0)
    width = st.number_input("Width (in)", 53.3, 207.4, 70.0)
    wheelbase = st.number_input("Wheelbase (in)", 57.0, 164.0, 105.0)
    clearance = st.number_input("Clearance (in)", 3.5, 12.0, 6.0)

    st.subheader("Key Features")

    air = st.selectbox("Air Conditioner", yes_no)
    abs_sys = st.selectbox("Anti-lock Braking System", yes_no)
    rear_belt = st.selectbox("Rear Seat Belts", yes_no)
    tyre = st.selectbox("Tyre Pressure Monitor", yes_no)
    convertible = st.selectbox("Removable Convertible Top", yes_no)

    submit = st.form_submit_button("Predict Price")


# =====================================================
# Prediction
# =====================================================

if submit:

    # Create EMPTY row with correct columns
    input_df = pd.DataFrame(columns=INPUT_COLUMNS)
    input_df.loc[0] = np.nan

    # Fill only raw inputs
    input_df.loc[0, [
        "Brand","Body.Type","Fuel.Type","Gearbox.Type","Drivetrain",
        "Power.hp","Displacement.l","Torque.lbft","Cylinders",
        "MPG.City","MPG.Highway","Seats","Doors",
        "Height.in","Length.in","Width.in","Wheelbase.in","Clearance.in",
        "Air.Conditioner","AntiLock.Braking.System",
        "Rear.Seat.Belts","Tyre.Pressure.Monitor",
        "Removable.Convertible.Top"
    ]] = [
        brand, body, fuel, gearbox, drivetrain,
        power, displacement, torque, cylinders,
        mpg_city, mpg_highway, seats, doors,
        height, length, width, wheelbase, clearance,
        air, abs_sys, rear_belt, tyre, convertible
    ]

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: $ {prediction:,.2f}")
