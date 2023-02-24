# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:14:39 2023

@author: Lab Pc
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

image = Image.open(r'tree.jpg')

col1, col2  = st.columns([0.8,0.2])
with col1:
    st.markdown('Upload your Image here')

with col2:
    st.image(image, width=150)
    
st.sidebar.markdown("Image Converter App")
with st.sidebar.expander("About The App"):
    st.write("""
             This app converts your image to a:\n 1. Pencil sketch \n 2. Grayscale image\n 3. Blurring effect on image.\n This app was created for computer vision applications using streamlit. It is amazing!
                 """)
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([0.5,0.5])
    
    with col1:
        st.markdown('Uploaded image')
        st.image(image,width=300)
        
    with col2:
        st.markdown('Changed image')
        filter = st.sidebar.radio('Convert your photo:',['Original','Gray Image','Black and White','Pencil Sketch', 'Blur Effect'])
        if filter == 'Gray Image':
            converted_img = np.array(image.convert('RGB'))
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            st.image(gray_scale, width=300)
        elif filter == 'Black and White':
            converted_img = np.array(image.convert('RGB'))
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
            st.image(blackAndWhiteImage, width=300)
        elif filter == 'Pencil Sketch':
            converted_img = np.array(image.convert('RGB')) 
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            inv_gray = 255 - gray_scale
            slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
            blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
            sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
            st.image(sketch, width=300) 
        elif filter == 'Blur Effect':
            converted_img = np.array(image.convert('RGB'))
            slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
            converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
            st.image(blur_image, channels='BGR', width=300) 
        else: 
                st.image(image, width=300)
st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Hope You Liked it!')
