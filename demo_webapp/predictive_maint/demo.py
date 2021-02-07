import streamlit as st
from PIL import Image
import base64
import os


def st_gif(path):
    with open(path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    
    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="display:block;margin-left:auto;\
    margin-right:auto">',
    unsafe_allow_html=True,
    )


def write():
    dirname = os.path.dirname(__file__)

    st.title("Predictive maintenance")

    st.markdown(
        """
        Lorem ipsum (TODO, this is just a placeholder page for now)

        """
    )