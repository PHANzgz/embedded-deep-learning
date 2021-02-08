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
        Lorem ipsum (TODO, this is just a placeholder page for now)  

        """
    )

    st.markdown(
        """
        # Sneak peek
        Although the app is not nearly finished, here are a couple of spoilers on what you should expect. An accelerometer(IMU) is put
        on the end of the barrel of a pen(furthest possible from the tip so it doesn't feel uncomfortable when writing)
        and those accelerations serve as input for the model. This is called online handwriting recognition as the characters are
        predicted while the strokes are being drawn rather than once the sketch is done(this would be done with images).

        """
    )

    st_gif(os.path.join(dirname, "img/spoiler-hardware.gif"))

    st.markdown(
        """
        This is just a quick test I did to check the three channels of the accelerometer. I'm holding the pen by its tip and the MCU is
        attached to the pen with a simple 3D printed structure. Please note that I will not be using Arduino libraries to develop the 
        application even though I did for this test.  

        """
    )

    st_gif(os.path.join(dirname, "img/spoiler-umap.gif"))

    st.markdown(
        """
        Creating the dataset from scratch is proving to be quite tedious. Nevertheless, with 211 examples per class, and six classes for
        now(A, B, C, D, E, F), making 2166 examples in total we can see an UMAP visualization to check if our data seems separable. This
        is handy to check if the features extracted from the raw data actually help the model. It can be easily seen how our data seems
        indeed to form a [manifold](https://scikit-learn.org/stable/modules/manifold.html#manifold) for each class; For example, take a
        look at classes E(purple) and F(brown), they are drawn similarly and their manifolds show that. If the model is capable to
        understand that relation and unwrap such manifold it should have no trouble differentiating between them.  

        """
    )

    st.info(
        """
        I'm using [Edge Impulse](https://www.edgeimpulse.com/) to develop this application. It has excellent tools for data acquisition and 
        model deployment. On top of that, you can actually code your own preprocessing blocks and model architectures with tensorflow, which 
        I did not know  until recently and was what convinced me to use this framework. I'll make a proper review when I finish the application.

        """
    )

    st_gif(os.path.join(dirname, "img/spoiler-classification.png"))

    st.markdown(
        """
        As you can see, without looking too much for the perfect architecture, a traditional convolutional neural network does an amazing
        job already. More on the feature extraction and model architecture in a few weeks. With these results, I have validated the
        experiment and know that the app will work for all the alphabet(with enough data). More updates coming soon!

        """
    )
    
    st.markdown(
            """
            # A homemade demo, running on the nRF5240
            (TODO) 

            """
        )