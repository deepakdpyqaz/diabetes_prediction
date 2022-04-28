import streamlit as st
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
st.set_page_config(page_title="Dibetes Prediction", page_icon="🤖")
st.title("Diabetes Prediction")
st.sidebar.markdown("""Code at [GitHub](https://github.com/deepakdpyqaz/diabetes_prediction)""")
mode = st.sidebar.selectbox("Select Mode", ["Predict", "View Details"])
MODEL_DIR = "serving_model/dibetes_prediction_pipeline"
DATA_DIR = "processed_data"
available_models = os.listdir(MODEL_DIR)
model_selected = st.sidebar.selectbox("Models",available_models)

with open("scalers/scaler.pickle","rb") as f:
    scaler = pickle.load(f)
def predict(args):
    model_loaded = keras.models.load_model(os.path.join(MODEL_DIR,model_selected))
    inputs = pd.DataFrame([{
        "Pregnancies":float(args["Pregnancies"]),
        "Glucose":float(args["Glucose"]),
        "BloodPressure":float(args["BloodPressure"]),
        "SkinThickness":float(args["SkinThickness"]),
        "Insulin":float(args["Insulin"]),
        "BMI":float(args["BMI"]),
        "DiabetesPedigreeFunction":float(args["DiabetesPedigreeFunction"]),
        "Age":float(args["Age"]),
    }])
    inputs = scaler.transform(inputs)
    prediction = model_loaded.predict([*inputs.reshape(8,-1)])
    return prediction[0][0]

def viewDetails():
    model_loaded = keras.models.load_model(os.path.join(MODEL_DIR,model_selected))
    data = os.path.join(DATA_DIR,os.listdir(DATA_DIR)[0])
    df = pd.read_csv(data)
    y = df["Outcome"]
    df = df.drop("Outcome",axis=1)
    scaled_df = scaler.transform(df)
    predicted = model_loaded.predict([df["Pregnancies"],df["Glucose"],df["BloodPressure"],df["SkinThickness"],df["Insulin"],df["BMI"],df["DiabetesPedigreeFunction"],df["Age"]])
    predicted = predicted.flatten()
    for i in range(len(predicted)):
        if predicted[i]>0.5:
            predicted[i]=1
        else:
            predicted[i]=0
    
    cnf = confusion_matrix(y,predicted)
    clf = classification_report(y,predicted,output_dict=True)
    return cnf,clf

if mode == "Predict":
    st.subheader("Predict whether a patient is diabetic or not")
    with st.form("prediction_form",clear_on_submit=False):
        l_col,r_col = st.columns(2)
        glucose = l_col.text_input("Glucose")
        BMI = l_col.text_input("BMI")
        blood_pressure = r_col.text_input("Blood Pressure")
        skin_thickness = r_col.text_input("Skin Thickness")
        diabetes_pedigree_function = l_col.text_input("Diabetes Pedigree Function")
        age = r_col.text_input("Age")
        insulin = r_col.text_input("Insulin")
        pregnancies = l_col.text_input("Pregnancies")
        submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                prediction = predict({"Glucose":glucose,"BMI":BMI,"BloodPressure":blood_pressure,"SkinThickness":skin_thickness,
                                        "DiabetesPedigreeFunction":diabetes_pedigree_function,"Age":age,"Insulin":insulin,
                                        "Pregnancies":pregnancies,
                                })
                if prediction>0.5:
                    st.write(f"The person is diabetic with confidence {round(prediction*100,2)} %")
                else:
                    st.write(f"The person is non-diabetic with confidence {round((1-prediction)*100,2)} %")
            except Exception as e:
                print(e)
                st.error(str(e))

if mode=="View Details":
    st.subheader("View Details")
    clicked = st.button("Click to view")
    if clicked:
        cnf,clf = viewDetails()
        st.write("Classification Report")
        df = pd.DataFrame(clf)
        st.write(pd.DataFrame(clf).transpose())
        st.write("Heatmap")
        cnf = cnf.astype('float') / cnf.sum(axis=1)
        fig, ax = plt.subplots()
        sns.heatmap(cnf,ax=ax,fmt=".1%",annot=True,linewidths=1.0, square=1)
        st.pyplot(fig)
    