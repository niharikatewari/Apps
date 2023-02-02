# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:23:40 2023

@author: Lab Pc
"""

import streamlit as st
from pycaret.regression import load_model

pipeline = load_model("Machine Learning Model")
pipeline