import streamlit as st 
import pandas as pd 

st.markdown("### LinkedIn Usage")
s = pd.read_csv("social_media_usage.csv")
s.shape
