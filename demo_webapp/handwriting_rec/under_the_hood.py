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
        
        When working with a time series such as accelerometer data, one may think simple RNNs or LSTMs are the best options, but
        they can be very resource-intensive and are not currently supported in Tensorflow Lite. This is a perfect opportunity to
        use a convolutional neural network.

        For the task at hand, we have two seconds of data at 100Hz(i.e. 200 samples) and three features, one for each axis (X, Y, Z).
        As CNNs are usually used for images(although lately they are finding new use-cases) a good visualization is to think of the
        data as a three-channel image of height one and width equal to the number of timesteps(200). If we choose the kernel
        sizes of height one, the kernel will only move along the width axis(i.e. time axis).

        With that considered, the architecture is pretty simple. Consecutive convolutional and max-pooling layers, with deeper layers
        having more filters and fewer timesteps followed by a dense classification head:

        """
    )

    st_gif(os.path.join(dirname, "img/model.png"))

    st.markdown(
        """
        And here is a sample of the code used to generate the model in Keras:

        ```python
        # model architecture
        ...
        model = Sequential()
        model.add(Reshape((1, 200, 3), input_shape=(n_timesteps*n_features, )))
        model.add(Conv2D(8, kernel_size=(1,9), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1,2), strides=2, padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(16, kernel_size=(1,6), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1,2), strides=2, padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(32, kernel_size=(1,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1,2), strides=2, padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(64, kernel_size=(1,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1,2), strides=2, padding='same'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(16, activation='relu',
            activity_regularizer=tf.keras.regularizers.l1(0.00001)))
        model.add(Dense(classes, activation='softmax', name='y_pred'))
        ...
        ```

        This architecture performs pretty well as shown in the results section, and it only takes 62 ms
        to run in an nRF52840 MCU.

        """
    )

    st.markdown(
        """
        # Results

        Before we dive into the results, it is important to note I had to create the dataset myself: I took
        1261 training samples for each class(Letters "A" to "O" plus a "noise" class) and 185 test samples
        for each class. All those samples contain **only** my handwriting but serves as a solid proof of concept
        of what could be achieved with enough data for different handwriting styles.

        The following image summarizes the validation set performance, including the confusion matrix, accuracy
        and F1 score for each class:

        """
    )

    st_gif(os.path.join(dirname, "img/validation_results.png"))

    st.markdown(
        """
        As for the test set results, here are the results:

        """
    )

    st_gif(os.path.join(dirname, "img/test_results.png"))

    st.markdown(
        """
        The performance is great and shows quite well where the model is struggling the most, which I verified with the
        deployed application.

        The letter "G" is the most misrepresented class and gets classified as "C" a fair amount of times. One example
        that gets misclassified and it is not very noticeable on the confusion matrix is the letter "H", which gets
        classified as "A" if it is not strictly well drawn.

        Nevertheless, the model works pretty well and classifies in real-time the drawn letters to their corresponding classes.
        The following section explains how this is achieved on the microcontroller.

        """
    )


    st.markdown(
        """
        # Implementation
        
        Setting up a solid pipeline that correctly classifies real-time data can be tricky. Here are the considerations and design
        decisions I took. I reference 
        [some of the code](https://github.com/PHANzgz/embedded-deep-learning/tree/master/applications/handwriting_recognition/application/online-handwriting-recognition/source)
        which is entirely available on GitHub, which is pretty well documented.

        * The application requires at least two tasks(running on Mbed OS RTOS). One is in charge of getting the accelerometer data with a fixed sampling rate.
        The other task is the inference task, which must retrieve the last two seconds of data and run the preprocessing and classifier.
        * To store the data, a `DataProvider` class(inside `feature_provider.hpp`) contains a simple ring buffer implementation that allows pushing new data or
        retrieving the last `length` samples ordered chronologically.
        * Leveraging the Edge Impulse SDK, the preprocessing implementation was built on top of it and runs along with the classifier. This corresponds to the
        `preprocessing.hpp` file. This allows for flexibility and scalability.
        * Every time a prediction is made, there must be some kind of filtering. This is the job of the `OutputHandler` class(inside `output_handler.cpp`). It computes
        the mean scores for each class over the last `window_size` predictions and when a class mean score is higher than a `threshold_probability` and it is 
        the highest of the mean scores, a valid prediction is found.
        * Once a valid prediction is found, the output handler ensures there have not been any **valid** predictions on the last `suppresion_window_size` runs.
        This ensures that the model does not predict the same event(written letter) more than once due to the nature of the real-time implementation.
        * As an example output handle, the implementation sends the data over BLE with a custom service. This requires running another task in the background
        that checks for BLE events.

        Here is a **simplified** flowchart to understand how it all ties together:
        """
    )

    st_gif(os.path.join(dirname, "img/handwriting_recognition_architecture.png"))

    st.markdown(
        """
        # Conclusion

        The online handwriting recognition problem can be approached with simple accelerometer data and a simple convolutional model. Nevertheless, there are
        key factors to take into account when the deployment target is a microcontroller, like ensuring a correct data pipeline and output handler.

        With enough data, the model could learn to represent the classes better and generalize to all letters and different handwriting styles. This application
        serves as a solid proof of concept for future work.

        """
    )