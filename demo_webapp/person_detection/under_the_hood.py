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

    st.markdown(
        """
        # Multilabel classification on the edge
        # **Under the hood**

        This section delves deeper into the application, showing how and why the model was chosen,
        the data it was trained on and the architecture overview for the deployed application.

        Remember you can find all the development code at the [person detection section of the
        github repository](https://github.com/PHANzgz/embedded-deep-learning/tree/master/applications/person_detection)
        including the jupyter notebook(generously documented) containing the data gathering, processing and model selection 
        as well as the C files for the deployment on an STM32H743 microcontroller.

        """
    )


    st.title("Architecture overview")

    st.image(Image.open(os.path.join(dirname, "img/person-detection-infogram.png")))

    st.markdown(
        """
        # A homemade demo, running on the STM32H743 MCU
        Inference time on ARM Cortex-M7: 1195 ms  

        """
    )

    st_gif(os.path.join(dirname, "img/homemade-demo.gif"))


    st.markdown(
    """
    # The data  

    Lorem ipsum (TODO)

    """
    )

    st.markdown(
    """
    # Model selection  

    Lorem ipsum (TODO)

    """
    )

    st.markdown(
    """
    # Deployment  

    Lorem ipsum (TODO)

    """
    )