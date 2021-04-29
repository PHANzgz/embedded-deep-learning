import streamlit as st
from PIL import Image
import base64
import os
import pickle
import plotly.graph_objects as go


def st_gif(path):
    with open(path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    
    st.markdown(
    f'<br><img src="data:image/gif;base64,{data_url}" alt="cat gif" style="display:block;margin-left:auto;\
    margin-right:auto"><br>',
    unsafe_allow_html=True,
    )

#@st.cache
def load_plot(filepath):
    with open(filepath, "rb") as f:
        plot = pickle.load(f)
    
    return plot

def write():
    dirname = os.path.dirname(__file__)

    st.markdown(
        """
        # Handwriting recognition using accelerometer data
        # **Under the hood**

        In this application, preprocessing the data and setting the continuous inference pipeline was crucial. 

        This section gives further details on those topics as well as on the model architecture.

        Remember you can find all the development code at the [handwriting recognition section of the
        github repository](https://github.com/PHANzgz/embedded-deep-learning/tree/master/applications/handwriting_recognition)
        including a couple of jupyter notebooks showing the preprocessing steps, data augmentation, and model training as well
        as the deployment C/C++ code, which is quite documented.

        """
    )

    st.markdown(
        """
         # Preprocessing  
        
        Sensor data can be very noisy. It is therefore essential to filter out the high-frequency components to eliminate them.
        Apart from that, handwriting is done at a "normal" pace, meaning those sudden changes in acceleration will not help
        the model at all.  

        With a sampling rate of 100Hz, a second-order Butterworth low-pass filter with a cutoff frequency of 20Hz was chosen. It
        is easy to implement both in Python and C/C++.  

        Intuitively, the model is interested in how that acceleration changes over time and its magnitude. Therefore, the data was
        not scaled. To ease learning, the mean was subtracted for each axis(X, Y, Z), making the data centered around zero.

        With that in mind, it was crucial to develop the preprocessing both in a pythonic way for efficient training and one that
        didn't use any libraries that resembled a C/C++ implementation. Here are the results:

        """
    )

    st.plotly_chart(load_plot("demo_webapp/handwriting_rec/preprocessing_comparison.pkl"), use_container_width=True)

    st.markdown(
        """
        Both implementations produce the same result. The cutoff frequency was chosen so there were no abrupt changes in the 
        data and the model wouldn't overfit. The filter order was set to two to make the computational cost low. 

        The entire preprocessing pipeline consists of getting the last two seconds of data, subtracting the mean, and applying
        the low-pass filter, which takes approximately 10ms on the nRF52840.

        """
    )

    st.markdown(
        """
        # The model  
        
        Lorem Ipsum (TODO)

        """
    )

    st_gif(os.path.join(dirname, "img/model.png"))

    st.markdown(
        """
        # Implementation
        
        Lorem Ipsum (TODO)

        """
    )

    st.markdown(
        """
        # Results
        
        Lorem Ipsum (TODO)

        """
    )