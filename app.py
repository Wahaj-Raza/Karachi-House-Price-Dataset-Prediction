import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import Ridge

df = pd.read_csv("Cleaned_Data.csv")
options = df["Address"].unique()

pipe = pickle.load(open("RidgeModel.pkl","rb"))


def check_fields():
    return location !="" and NoOfBedrooms != 0 and NoOfBathrooms!=0 and SqYardArea!=0

st.title('Welcome to Karachi House Price Predictor')
st.text("Want to know the price of a new house in Karachi? Try filling the details below:")
location = st.selectbox("Select the Location", options)
NoOfBedrooms = st.number_input("Enter Number of Bedrooms:")
NoOfBathrooms = st.number_input("Enter Number of Bathrooms:")
SqYardArea = st.number_input("Enter Total Square Yard")

button = st.button("Predict Price",type="primary", disabled=not check_fields(), use_container_width=True)
if button:
    input = pd.DataFrame([[location,NoOfBedrooms,NoOfBathrooms,SqYardArea]],columns=["Address","NoOfBedrooms","NoOfBathrooms","AreaSqYards"])
    prediction = pipe.predict(input)[0]

    st.title("PKR. "+str(abs(int(prediction))))