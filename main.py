import streamlit as st 
import pandas as pd 

st.markdown("### LinkedIn Usage")
s = pd.read_csv("social_media_usage.csv")
s.shape


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
  
