import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from PIL import Image

# Load data and model
df = pd.read_csv("Cleaned_Data.csv")
options = sorted(df["Address"].unique(), reverse=True)[1:] + ["other"]
pipe = pickle.load(open("RidgeModel.pkl", "rb"))
pipe_new = pickle.load(open("Transform.pkl", "rb"))
x_train_df = pd.read_csv("X_Train.csv")
image = Image.open("header_image.jpg")


# Function to validate input fields
def check_fields():
    return location != "" and NoOfBedrooms != 0 and NoOfBathrooms != 0 and SqYardArea != 0

# App title and description
st.set_page_config(page_title="Karachi House Price Predictor", page_icon=":house:", layout="wide")
st.title('Welcome to Karachi House Price Predictor')
st.markdown("This app predicts the price of a house in Karachi based on its location, number of bedrooms and bathrooms, and total square yard area.")
st.image(image, use_column_width=True)

# Sidebar with input fields
st.sidebar.subheader("Enter House Details")
location = st.sidebar.selectbox("Location", options)
NoOfBedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, value=1)
NoOfBathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, value=1)
SqYardArea = st.sidebar.number_input("Total Square Yard Area", min_value=20)

# Predict button
if st.sidebar.button("Predict Price", key="predict", disabled=not check_fields()):
    # Create input dataframe for prediction
    input = pd.DataFrame([[location, NoOfBedrooms, NoOfBathrooms, SqYardArea]], columns=["Address", "NoOfBedrooms", "NoOfBathrooms", "AreaSqYards"])
    
    # Make prediction
    prediction = pipe.predict(input)[0]

    # Find 5 most similar properties in the training dataset
    x_test = pipe_new.transform(input)
    x_train_transformed = pipe_new.transform(x_train_df)
    kdt = KDTree(x_train_transformed, leaf_size=30, metric='euclidean')
    indices = kdt.query(x_test, k=5, return_distance=False)
    similar_properties = x_train_df.iloc[indices[0]]
    
    # Display prediction and similar properties
    st.subheader("Prediction")
    st.markdown(f"<span style='font-size:30px;font-weight:bold;color:green;'>PKR {int(prediction):,}</span>", unsafe_allow_html=True)

    
    # Update the subheader for similar properties
    st.markdown(f"<span style='font-size:40px;font-weight:bold;color:white;'>Similar Properties (Top 5)</span>", unsafe_allow_html=True)


    # Create a loop to display each property on a separate page
    for i in range(len(similar_properties)):
        property = similar_properties.iloc[i]
        address = property["Address"]
        bedrooms = int(property["NoOfBedrooms"])
        bathrooms = int(property["NoOfBathrooms"])
        area = int(property["AreaSqYards"])

        # Create a unique title for each property page
        page_title = f"{address} | {bedrooms} BR - {bathrooms} BA - {area} SqYd"

        st.write(f"## {page_title}")

        # Display the property information in a dataframe
        to_show = property.to_frame().iloc[1:,:]
        try:
            to_show.columns=[ "Property Value"]
        finally:
            st.write(to_show)

        # Add a horizontal line to separate each property page
        st.write("---")

# Add a footer using HTML and CSS
footer_html = """
<footer style='text-align: center; font-size: 12px; margin-top: 50px;'>
    Made with ❤️ by <a href="https://github.com/wahajraza">Wahaj Raza</a>
</footer>
"""
st.markdown(footer_html, unsafe_allow_html=True)
