import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model= tf.keras.models.load_model('model.h5')
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on their information.")
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])   
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider("Age", 18, 92, 30)
balance= st.number_input("Balance")
credit_score= st.number_input("Credit Score")
estimated_salary= st.number_input("Estimated Salary")
tenure=st.slider("Tenure", 0, 10, 5)
number_of_products=st.slider("Number of Products", 1, 4, 2)
has_cr_card=st.selectbox("Has Credit Card", (0, 1))
is_active_member=st.selectbox("Is Active Member", (0, 1))

inputdata = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']), index=[0])
inputdata=pd.concat([inputdata.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

inputdata_scaled= scaler.transform(inputdata)

#Predict churn
prediction = model.predict(inputdata_scaled)
prediction_proba = prediction[0][0]
st.write(f"Prediction Probability: {prediction_proba}")
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")