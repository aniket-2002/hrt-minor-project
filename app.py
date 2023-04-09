
#Library imports
from email import header
import numpy as np
from pandas import options
import streamlit as st
import pickle
import pandas as pd
import sklearn 
import os
import base64
import joblib 

from sklearn.preprocessing import StandardScaler
#from keras.models import load_model


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#Loading pickled file.
model = pickle.load(open('./Models/model.pkl', 'rb'))

def predict(model, data):
    scale=StandardScaler()

    
    x_test = scale.fit_transform(data)
    prediction = model.predict(x_test)
    print(prediction)
    return prediction

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_ensembled_model(model_folder_path = './Models'):
    
    # full_paths = []
    # for model_path in os.listdir(model_folder_path):
    #     """
    #     [
    #         knn.pkl,
    #         model.pkl
    #     ]
        
    #     full_paths = [
    #         Models/knn.pkl 
    #         Models/model.pkl 
    #     ]
    #     """
    #     model_full_path = os.path.join(model_folder_path, model_path)
    #     full_paths.append(model_full_path)
        
    
    list_model_paths = [
        os.path.join(model_folder_path, model_path) for model_path in os.listdir(model_folder_path)
    ] 
    
    models = [] 
    for model_path in list_model_paths:
        models.append(
            joblib.load(model_path)
        )
        # with open(model_path, 'rb') as f:
        #     model = pickle.load(f)
        #     models.append(model)
    
    return models 

def predict_ensembled(model_list, X):
    
    predictions = [0] * len(model_list) 
    
    for i, model in enumerate(model_list):
        predictions[i] = predict(model, data=X)
    
    return predictions

    

def set_background(jpg_file):
    bin_str = get_base64(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./images/heartimage.jpg')
model_list = get_ensembled_model(model_folder_path='./model')


with header:
    #Setting Title of App
    st.title("Heart Failure Prediction🫀-")
    st.text("This application allows users to input their data and predicts whether they are likely to have heart disease or not.")
    
    
with dataset:
    def main():
        st.header("Data Input.")
        st.text("This data is used to predict.")
        html_temp = """
        <style>
            .reportview-container .main .block-container{{
                max-width: 90%;
                padding-top: 5rem;
                padding-right: 5rem;
                padding-left: 5rem;
                padding-bottom: 5rem;
            }}
            img{{
                max-width:40%;
                margin-bottom:40px;
            }}
        </style>
        """
        
        st.markdown(html_temp, unsafe_allow_html=True)
        
        age = st.number_input("Please put your age.", 0)
        sex = st.number_input("What is your sexual orientation? 1 for male and 0 for female", 0)
        cp = st.number_input("Input(Chest pain type in number => Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)", 0)
        trtbps = st.number_input("Input(resting blood pressure (in mm Hg))", 0)
        chol = st.number_input("Input(cholestoral in mg/dl fetched via BMI sensor)", 0)
        fps = st.number_input("Input(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)", 0)
        restecg = st.number_input("Input( resting electrocardiographic results(male HR-49 to 100 bpm & female HR- 55 to 108 bpm))", 0)
        thalachh = st.number_input("Input(maximum heart rate achieved)", 0)
        exng = st.number_input("Input(exercise induced angina (1 = yes; 0 = no))", 0)
        caa = st.number_input("Input(number of major vessels (0-3))", 0)
        
        input_list = ["age", "sex", "cp", "trtbps", "chol", "fps", "restecg", "thalach", "exng", "caa"]
        data = pd.DataFrame(data=[[age,sex,cp,trtbps,chol,fps,restecg,thalachh,exng,caa]],columns=input_list)
        
        result=""
        if st.button("Predict"):
            result=predict(model, data)
            ensembled_result = predict_ensembled(model_list, data)
            print("Ensembled result: ", ensembled_result)
            print("Single model result ", result)
            
            for i in result:
                if i==1:
                    st.success('Yes you have heart disease. Please Consult a doctor')
                else:
                    st.success("No you don't have heart disease but take care of your heart.")
        if st.button("About"):
            st.text("This app was built using Streamlit.")
            st.text("The model was trained using the Heart Disease UCI dataset.")
            st.text("MADE BY Aniket,Jyotirmoy,Utkarsh & Amitanshu")
            
            
    

if __name__=='__main__':
    main()
    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 