# root@0404-172430-kvzfls7l-10-139-64-4:/Workspace/Repos/kristof.rabay@hiflylabs.com/med-diagnose-demo# streamlit run app.py

import streamlit as st

prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")