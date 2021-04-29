import streamlit as st
from PIL import Image
import base64
import os
import pandas as pd


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


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info(
        """
        # Note
        As previously mentioned, the [jupyter notebook](https://github.com/PHANzgz/embedded-deep-learning/tree/master/applications/person_detection)
        that accompanies the research process is very well documented and contains detailed information about all aspects of the application.

        The following sections provide a **summary** of the data pipeline, model selection and deployment phases.
        """
    )

    st.markdown(
    """
    # The data  
    
    The nature of the application requires images that contain people, cars, and both, as well as negative images. The data is aggregated using three
    different datasets: [COCO dataset](https://cocodataset.org/#home), [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) and the
    [MIT Indoor dataset](http://web.mit.edu/torralba/www/indoor.html).

    Here are some insights into why multiple datasets were used as well as some considerations:

    * The COCO dataset has both "car" and "person" classes. On top of that, the dataset name means "Common Objects in COntext" so, in theory, the images 
    will be a good representation of the reality the model will be exposed to in the final application.
    * Some images in the COCO dataset that contain the classes we are interested in feature the object in a tiny part of the image, so a 
    `threshold_area` is set for each class, resulting in only getting the samples that meet the criteria
    * With only COCO, the "person" class has more than six times more examples than the "car" class. To deal with this class imbalance, undersampling is
    proposed but the "car" class is misrepresented and the overall performance is not ideal. Therefore, the Cars dataset is added to balance classes.
    * One downside of the Cars dataset is that all of the images contain clean, centered, and not occluded cars. This leads to worse generalization for
    the "car" class.
    * For this demo, the application will be mostly tested at home, university, and indoors in general. Apart from that, a curious observation of the
    COCO dataset is that a lot of the "person" images are indoors. This led to the model to predict "person" when an indoor scene was shown without
    any people on it. To solve this, the "MIT Indoor dataset" is added to represent the negative class better.

    In summary, as this application is made for educational purposes only, creating a dataset from scratch is not an option. Therefore, we get the
    closest publicly available datasets that best resemble the reality the model will be exposed to and aggregate them together.

    All the data was downloaded through their APIs or official download links with several python scripts, allowing for easy flexibility for different
    applications that may require different classes. 

    """
    )

    st.markdown(
    """
    # Model selection  

    Computer vision state-of-the-art deep learning architectures are usually large. The concern of reducing latency and moving the computation to the
    client-side(instead of server-side) have resulted in more constrained architectures like EfficientNet, MobileNet, MobileNetV2, ShuffleNet, etc. 
    which usually target mobile phones or tablets.

    The main target for this application is an STM32H7 microcontroller, with generous 2MB of flash memory and 1MB of RAM. Those are really high-end specifications 
    for a microcontroller. Still, none of the models in that list would fit those constraints, at least not in their original form. Both 
    [EfficentNet](https://arxiv.org/abs/1905.11946) and [NASNet](https://arxiv.org/abs/1707.07012) shine on mobile applications, but 
    [MobileNet](https://arxiv.org/abs/1704.04861) and [MobileNetV2](https://arxiv.org/abs/1801.04381) apart from performing nearly as well, they propose two 
    hyperparameters that affect the model complexity and capacity and, in turn, size and latency.

    The following table summarizes several configurations for the MobileNet and MobileNet V2 architecture with the results obtained:

    """
    )

    st.dataframe(pd.read_csv("demo_webapp/person_detection/model_selection_complete.csv"))

    st.markdown(
    """
    The column `n_dense` refers to the number of neurons in the classification head. The "FT" and "TR" suffixes refer to the fine-tuning and transfer
    learning phases during training. A value of `-1` is used when it was not measured.

    Before comparing models, you may have realized the loss `macro_double_soft_f1`. This is a differentiable version of the F1 score metric which "instead
    of computing the number of true positives, false positives and false negatives as discrete integer values, it computes them as a continuous sum of
    likelihood values". Check [this amazing post](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d)
    for more information about it. It has the benefit of optimizing directly for the F1 score, which is the main metric we will be using as well as the fact that
    the need for choosing a threshold to consider each class positive is implicitly inside the loss and we don't have to tweak those values.

    Indeed, through various experiments and seeing the results of the above table, the "macro_double_soft_f1" loss produced slightly, and sometimes considerable, better
    results.

    The table also shows the model sizes when compressed to 16-bit floating-point weights, although the final application further constrains those weights to 8-bit integers,
    otherwise the model would not fit(not enough RAM). Choosing `alpha`s bigger than `0.25` also resulted in models too large to fit into the microcontroller.

    Fully connected layers are pretty resource-heavy, increasing model size and latency, so it was gradually reduced to `64`, where performance holds quite well.
    In the last iteration, the model was trained on smaller resolution images, producing roughly a four-time speed-up in terms of latency. The model performance
    was not too affected, so this was chosen as the final model.

    It is important to remember that the architectures(and its parameters) to be selected were constrained by the pre-trained(on ImageNet) models available on Keras 
    in order to benefit from transfer learning.

    """
    )

    st.markdown(
    """
    # Deployment 

    The model input accepts RGB888(8 bits per color) images of shape (160,160,3), but the camera attached to the microcontroller captures images in RGB555 with
    a resolution of 240x160. This odd resolution comes from the fact that the captured image is stretched to reduce RAM usage.

    This divergence of format and resolution makes necessary some preprocessing steps. Although the full preprocessing could be done inside the model, some of
    the operations required are not supported in Tensorflow Lite for microcontrollers. In particular, the RGB565 to RGB888 is done in C, which is fairly easy
    to implement with simple masking and remapping, and the image resizing is done with Tensorflow inside the model.

    To add this resizing(and it applies normalization as well), a simple custom layer is created with the Keras subclassing API:
    ```python
    # only shows a portion of the code
    class uc_preprocess(keras.layers.Layer):
    ...
        def call(self, inputs):
            # Resize the image
            res_imgs = tf.image.resize(inputs, self.out_img_size, method='nearest') # method supported by tflite micro
            # Normalize to the range [-1,1]
            norm_imgs = res_imgs*(1/127.5) -1 # multiply by reciprocal as DIV is not supported by tflite micro as of October, 2020
        
            return norm_imgs
    ...
    ```

    After compressing the model with integer quantization using tflite, the performance is barely affected and the model only weights 345kB. Inference
    takes around 1195 ms on an STM32H7, which is quite good for the application purpose: Estimating and improving traffic setting up cameras
    on top of traffic lights.

    """)

    st.markdown(
        """
        The microcontroller implementation is configured as follows: A photo is received from the OV7670 camera and sent to the microcontroller through DMA and with the help
        of the DCMI peripheral. After that, the model input buffer is filled, transforming the received RGB565 image to RGB888. Inference runs and the results are
        displayed on an ILI9341 LCD screen for demonstration purposes. Here is a snippet of the code:

        ```c
        if (new_capture){
            new_capture=0;

            // TENSORFLOW
            // Fill input buffer
            uint16_t *pixel_pointer = (uint16_t *)frame_buffer;
            uint32_t input_ix = 0;

            for (uint32_t pix=0; pix<OV7670_QVGA_HEIGHT*OV7670_QVGA_WIDTH/2; pix++){
                // Convert from RGB55 to RGB888 and int8 range
                uint16_t color = pixel_pointer[pix];
                int16_t r = ((color & 0xF800) >> 11)*255/0x1F - 128;
                int16_t g = ((color & 0x07E0) >> 5)*255/0x3F - 128;
                int16_t b = ((color & 0x001F) >> 0)*255/0x1F - 128;

                model_input->data.int8[input_ix] =   (int8_t) r;
                model_input->data.int8[input_ix+1] = (int8_t) g;
                model_input->data.int8[input_ix+2] = (int8_t) b;

                input_ix += 3;
            }

            // Run inference, measure time and report any error
            timestamp = htim->Instance->CNT;
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
                return;
            }
            timestamp = htim->Instance->CNT - timestamp;
            car_score = model_output->data.int8[0];
            neg_score = model_output->data.int8[1];
            person_score = model_output->data.int8[2];
            // END TENSORFLOW

            // Display inference information
            Draw_Image((unsigned char*) frame_buffer);
            ILI9341_Set_Rotation(SCREEN_HORIZONTAL_2);
            if (person_score > 0) ILI9341_Draw_Text("PERSON", 180, 210, GREEN, 2, BLACK);
            else ILI9341_Draw_Text("PERSON", 180, 210, RED, 2, BLACK);
            if (car_score >  0) ILI9341_Draw_Text("CAR", 80, 210, GREEN, 2, BLACK);
            else ILI9341_Draw_Text("CAR", 80, 210, RED, 2, BLACK);

            // Print inference info
            buf_len = sprintf(buf,
                    "car: %+*d, neg: %+*d, person: %+*d | Duration: %lu ms\\r\\n",
                    4,car_score, 4,neg_score, 4,person_score , timestamp/1000);
            HAL_UART_Transmit(huart, (uint8_t *)buf, buf_len, 100);

            // Capture a new image
            ov7670_startCap(OV7670_CAP_SINGLE_FRAME, (uint32_t)frame_buffer);
        }
        ```


        """
        )

    st.markdown(
        """
        # Conclusion

        Running a state-of-the-art deep learning, computer vision model on a microcontroller can prove to be a challenge. But with the help of small
        pre-trained models and integer quantization it can be done.

        Furthermore, as multi-label classification is not as common as simple classification, setting up a good data pipeline can be time-consuming.

        The resulting model works quite well even in the deployed version, but a better quality dataset that represents the actual data distribution the model
        will be exposed to could lead to even better results, including commercial applications.

        """
    )