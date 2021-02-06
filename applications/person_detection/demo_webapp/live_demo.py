import streamlit as st
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from PIL import Image
import random
import requests
import time

def write():
    st.markdown(
    '''
    # Multilabel classification on the edge

    Image classification is a common task in deep learning. A bit more niche but still very important
    application is multiclass image classification, where an image may belong to **one or more** classes.
    This application aims to provide an example of the development and deployment of this task on constrained 
    devices like microcontrollers using tensorflow lite.

    In particular, this application aims to detect whether a person, a car, or both are present in an image. This
    has interesting applications like improving traffic through traffic lights.

    I have chosen to deploy the model on an STM32H743 microcontroller due to its generous specifications like
    2MB of flash memory and 1MB of RAM. Inference takes 1195 ms, resulting in 0.83FPS; Which may not seem like much but
    it is important to remember how cost and power effective microcontrollers are.  

    Nonetheless, other MCUs like the ESP32 or nRF528 series are also perfectly suitable targets for deployment.

    The model weights only 345kB and has a MobileNet feature extractor. For more information about the model and the
    application please check the "Under the hood tab" or
    [the github repository](https://github.com/PHANzgz/embedded-deep-learning).

    '''
    )

    st.image(Image.open("./img/hardware.jpg"), use_column_width=True,
            caption = "Hardware used: NUCLEO-H743ZI2 dev board, OV7670 camera, ILI9341 powered LCD-TFT screen")

    st.markdown(
    '''
    # Live demo

    If you want to try the model right now on your browser you can do it! Just select an image and let the model
    make its predictions.

    You can choose a random image from part of the validation dataset or upload
    an image of your own(JPG). 

    '''
    )

    st.markdown("## Loaded image (padded)")
    loaded_image_ph = st.empty() # placeholder for the loaded image

    left_column, right_column = st.beta_columns(2)

    left_column.text("Randomly pick an image from the\nvalidation dataset...")
    pressed_get_img = left_column.button('Get random image', key="get_img_button")
    uploaded_file  = right_column.file_uploader("Or upload your own...", "jpg")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

    elif pressed_get_img:
        with open("random_filenames.txt", "rt") as f: # Return random line
            line = next(f)
            for num, aline in enumerate(f, 2):
                if random.randrange(num):
                    continue
                line = aline

        url = "https://embedded-ai.000webhostapp.com/" + "person-detection-small-val/" + line[:-1]
        img = Image.open(requests.get(url, stream=True).raw)
        
    else:
        img = Image.open("./img/sample_image.jpg")


    # Display padded loaded image
    desired_size = 640
    old_size = img.size  

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    loaded_image_ph.image(new_im, use_column_width='auto')
    #loaded_image_ph.image(Image.open("./img/sample_image.jpg"), use_column_width='auto') # DEBUG

    # Transform image to model input format
    uc_img_size = (240, 160) # the resolution the OV7670 is halved horizontally to reduce RAM usage
    img_res = tf.image.resize(np.array(img), uc_img_size)
    img_prep = tf.cast(img_res-128., tf.int8)

    st.markdown(
    """
    ## Actual input of the model and prediction
    This simulates the capture of the OV7670 camera. It may seem an odd resolution but to reduce RAM usage
    the resolution is halved horizontally(160x240) from the orignal capture size. Otherwise the model RAM
    usage would not fit into the STM32H7. The performance metrics were barely affected.
    """
    )
    img_ph, pred_ph = st.beta_columns(2) # create placeholders
    img_ph.image(img_prep.numpy()+128, use_column_width='always', clamp=True)

    # Load tflite model
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="uc_final_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_prep, axis=0))
    t1 = time.time()
    interpreter.invoke()
    t2 = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Format scores in pretty progress bars and show prediction
    norm_scores = ((output_data + 128) / 255).reshape(-1)
    car_score, negative_score, person_score = norm_scores.tolist()
    pred_ph.markdown("<br>", unsafe_allow_html=True)
    pred_ph.markdown("**Car score**")
    pred_ph.progress(car_score)
    pred_ph.markdown("**Negative score**")
    pred_ph.progress(negative_score)
    pred_ph.markdown("**Person score**")
    pred_ph.progress(person_score)
    classes = np.array(["Car", "Negative", "Person"])
    predicted_classes = classes[np.where(norm_scores>0.5, True, False)]
    pred_ph.markdown("<br><br>**Predicted classes:** {}".format(", ".join(predicted_classes)), unsafe_allow_html=True)
    pred_ph.markdown("**Inference time(server side):** {} ms".format(int((t2-t1)*1000)), unsafe_allow_html=True)

    expander = st.beta_expander("Image credits")
    expander.markdown(
    """
    All the random images that come from the validation set have been used only 
    for educational and research purposes and come from the following sources:  

    - [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) - Copyright (c) 2015, COCO Consortium  
    - [Indoor scene dataset](http://web.mit.edu/torralba/www/indoor.html) - IEEE Conference on Computer Vision and 
    Pattern Recognition (CVPR), 2009.  
    - [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) - 3D Object Representations for Fine-Grained Categorization
       Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
       4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

    """)

    st.title("Caveats")
    st.markdown(
        """
        As was previously mentioned, the model weights just 345kB and has been trained on a rather generic dataset.  

        The dataset was built using multiple datasets(see "Image Credits" just above) with the purpose of providing
        a close representation of what the model would be exposed when deployed but still with a wide area of operation. 
        For example, since I will only be using this model for educational purposes only I added indoor scenes images as
        I will probably be running the application from home or the university. Otherwise the model would find strange 
        patterns and categorize them as "person" or "car".  

        With that being said, the model still performs quite well given its size, with a 0.89 F1 score on the validation 
        data and "quite good" performance detecting my relatives and I at home from the microcontroller version. Nevertheless,
        it has its limitations; You may have noticed how the default sample image actually contains a car in the background 
        but the model does not detect it. This occurs when the cars are widely occluded by other elements. The reason is 
        probably the data: Since the car class was unbalanced(much fewer examples) I added a lot of car images from the 
        "Cars Dataset", which contains clean, centered and not at all occluded cars.

        Another interesting but natural behavior that the negative class sometimes accompanies a positive class. For
        example, a car photo may give a 1.00 score to car and a 0.98 score to negative. It may seem shocking at first
        but it is completely normal, as the image probably contains thousands of features that do not match any other class.  

        """
    )

