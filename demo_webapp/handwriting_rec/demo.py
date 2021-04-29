import streamlit as st
from PIL import Image
import base64
import os


def st_gif(path):
    with open(path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    
    st.markdown(
    f'<br><img src="data:image/gif;base64,{data_url}" alt="cat gif" style="display:block;margin-left:auto;\
    margin-right:auto"><br>',
    unsafe_allow_html=True,
    )


def write():
    dirname = os.path.dirname(__file__)

    st.title("Handwriting recognition using accelerometer data")

    st.markdown(
        """
        Character recognition is a common practice in different fields and is typically done with optical methods(i.e. through images).
        Sometimes the use of a camera or an optical sensor may be inconvenient, so a different approach is proposed to recognize handwritten
        characters.  

        An accelerometer(IMU) is put on the end of the barrel of a pen(furthest possible from the tip so it doesn't feel uncomfortable when writing)
        and acceleration is measured in three axes. The application processes the data continuously and generates predictions if a pattern matches
        a trained character.  

        This method provides a cheaper and more convenient approach for handwriting recognition. Apart from that, the characters are predicted
        while the strokes are being drawn rather than once the sketch is done, that's why it's called *online* character recognition.
        

        """
    )

    st.markdown(
        """
        # A homemade demo, running on the nRF5240 MCU

        Data is continuously sampled and filtered. The data is fed to the convolutional neural network, which also runs constantly with the latest 
        two seconds of accelerometer data. Once a letter is written on paper and a prediction is made, the information is sent over BLE.  

        In this visual demo, the output BLE data is received by a Raspberry Pi Zero with a little screen on top of it showing the current string.  

        Preprocessing takes approximately 10ms.  
        Inference takes approximately 62ms.  

        """
    )

    st_gif(os.path.join(dirname, "img/demo.gif"))

    st.markdown(
        """
        # Validating the proposed solution  
        
        It may not be completely obvious that the model would be able to learn from the acceleration data, so the first thing I did was to train
        the model on a few letters with a simple, out-of-the-box 1D convolutional model.

        """
    )

    st_gif(os.path.join(dirname, "img/spoiler-umap.gif"))

    st.markdown(
        """
        Creating the dataset from scratch has proven to be quite tedious. Nevertheless, at the time of this test, with 211 examples per class, 
        and six classes(A, B, C, D, E, F) the simple model was trained with 2166 examples in total. The image shows an UMAP visualization to 
        check if the data seems separable. This is handy to check if the features extracted from the raw data actually help the model. 
        It can be easily seen how our data seems indeed to form a [manifold](https://scikit-learn.org/stable/modules/manifold.html#manifold) 
        for each class; For example, take a look at classes E(purple) and F(brown), they are drawn similarly and their manifolds show that. 
        If the model is capable to understand that relationship and unwrap such manifold, it should have no trouble differentiating between them. 

        """
    )

    st.info(
        """
        I'm using [Edge Impulse](https://www.edgeimpulse.com/) to develop this application. It has excellent tools for data acquisition and 
        model deployment. On top of that, you can actually code your own preprocessing blocks and model architectures with tensorflow, which 
        was what convinced me to use this framework.

        """
    )

    st.markdown(
        """
        The results for this test were the following:

        """
    )

    st_gif(os.path.join(dirname, "img/spoiler-classification.png"))

    st.markdown(
        """
        As you can see, without looking too much for the perfect architecture, a traditional convolutional neural network does an amazing
        job already. With these results, the experiment is validated and we can know know that the app will work for all the alphabet(with enough data).

        """
    )


    st.markdown(
        """

        # The model, preprocessing and results
        
        Check the "Under the hood" section of this app to learn more about the model architecture as well as the preprocessing applied to the raw data.  

        Metrics for the final model are also shown there.

        """
    )