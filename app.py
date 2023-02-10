# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:15:30 2023

@author: Lab Pc
"""

#######First App ################
import streamlit as st
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#ML 
import pycaret
from pycaret.regression import setup,pull, save_model
#st.write("First App")
#st.write(pd.DataFrame({
#    'first column':[1,2,3,4],
#    'second column':[10,20,30,40]
#    }))

with st.sidebar:
    st.image("https://singularityhub.com/wp-content/uploads/2018/11/multicolored-brain-connections_shutterstock_347864354.jpg")
    st.title("Machine Learning app")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allow to build automated machine learning pipeline using streamlit, pandas profiling and pycaret and it is amazing!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)
    
if choice=="Upload":
    st.title("Upload your Data Here")
    file = st.file_uploader("Upload your dataset")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        
if choice=="Profiling":
    st.title("Explore Your Data in Better Way")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice=="ML":
    st.title("Machine learning")
    target= st.selectbox("Select Your Target", df.columns)
    if st.button("Train model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'Machine Learning Model')

if choice=="Download":
    with open("Machine Learning Model.pkl",'rb') as f:
        st.title("Download your model from here:")
        st.download_button("Download the file",f,"Machine Learning Model.pkl")
