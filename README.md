# embedded-deep-learning
 Research and code examples about embedded deep learning applications in different areas and microcontrollers, like nRF52840 or STM32 MCUs.

> **This repository is still under development, but feel free to browse the web-app and the currently developed applications.**  

# Web app

The best way to enjoy this research is through the web app. You can come back later to check the source code.  
~~Please note it may take up to one minute to load the app if the server is asleep.~~  

<p align="center">
    <a href="https://share.streamlit.io/phanzgz/embedded-deep-learning/demo_webapp/app.py">
        <img src="images/launch-webapp-btn.png" alt="launch web app">
    </a>
</p>  



# Demo apps

## Person and car detection(STM32F7, STM32H7, STM32F4, (TODO ESP32) )
To improve (mostly night) traffic and pollution, an embedded device with a camera is put on traffic lights. They serve a similar purpose as the buttons for the pedestrians: when the device doesn't detect a person in a time window, the traffic light may switch to flashing amber, allowing cars to traverse the crossing with precaution. This prevents the agglomeration of vehicles in crossings and, more importantly, saves gas(or battery) as the vehicle does not have to stop, just slow down. The immediate benefit is better circulation, but saving both money and gas are also direct consequences. This in turn decreases the amount of CO2 released by vehicles and provides a very positive environmental impact.

### [Learn more](/applications/person_detection/) 
![webapp-sample](/images/webapp-demo-sample.jpg)
You can also try the model on your browser with the web app.  
<br>
![hardware](/demo_webapp/person_detection/img/hardware.jpg)
Hardware used: NUCLEO-H743ZI2 dev board, OV7670 camera, ILI9341 powered LCD-TFT screen
<br><br>

## Handwriting recognition using accelerometer data (nRF52840)
Character recognition is a common practice in different fields and is typically done with optical methods(i.e. through images).
Sometimes the use of a camera or an optical sensor may be inconvenient, so a different approach is proposed to recognize handwritten
characters.  
<br>
An accelerometer(IMU) is put on the end of the barrel of a pen(furthest possible from the tip so it doesn't feel uncomfortable when writing)
and acceleration is measured in three axes. The application processes the data continuously and generates predictions if a pattern matches
a trained character.  
<br>
This method provides a cheaper and more convenient approach for handwriting recognition. Apart from that, the characters are predicted
while the strokes are being drawn rather than once the sketch is done, that's why it's called *online* character recognition.  
<br>
For this application, the [Edge Impulse](https://www.edgeimpulse.com/) was used. It has excellent tools for data acquisition and 
model deployment. On top of that, you can code your own preprocessing blocks and model architectures with tensorflow and keras, adding
an arbitrary level of customization.  
<br>
### [Learn more](/applications/handwriting_recognition/) 
<br>

![homemade_demo](/demo_webapp/handwriting_rec/img/demo.gif)  
This (homemade) demo, shows the application running on an nRF52840 MCU. 
Data is continuously sampled and filtered. The data is fed to the convolutional neural network, which also runs constantly with the latest 
two seconds of accelerometer data. Once a letter is written on paper and a prediction is made, the information is sent over BLE.  

In this visual demo, the output BLE data is received by a Raspberry Pi Zero with a little screen on top of it showing the current string.  

Preprocessing takes approximately 10ms.  
Inference takes approximately 62ms.  
<br><br>

## Future work
* Motion detection and localization using Wi-Fi CSI signals (ESP32)
* Predictive maintenance (Apollo3, nRF52840)

## Other ideas
* Smart thermostat that learns from user interaction and improves user experience as well as saving energy.
* Pose detection for bin-picking

# Frameworks used

* Tensorflow Lite for microcontrollers
* X-CUBE-AI (STM32)
* Edgeimpulse
