# embedded-deep-learning
 Research and code examples about embedded deep learning applications in different areas and microcontrollers, like ESP32 or STM32 MCUs.

> **This repository is still under development and will be for a while, but if you are eager to see what is currently done feel free to browse.**  
Currently finished(pending revision):  
> * [Web app that contains interactive demos and info about each app](https://embedded-person-car-detection.herokuapp.com/)
> * [Person and car detection app](/applications/person_detection/) (research and deployment)

# Web app

The best way to enjoy this research is through the web app. You can come back later to check the source code.

<p align="center">
    <a href="https://embedded-deep-learning.herokuapp.com/">
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
(TODO, framework edgeimpulse+keras)  

## Motion detection and localization using Wi-Fi CSI signals (ESP32)
(TODO)

## Predictive maintenance (Apollo3, nRF52840)
(TODO)

## Other ideas
* Smart thermostat that learns from user interaction and improves user experience as well as saving energy.
* Pose detection for bin-picking
* Bark detection for dogs (TODO find a better speech recognition application)
* (TODO)

# Frameworks used

* Tensorflow Lite for microcontrollers
* X-CUBE-AI (STM32)
* Edgeimpulse

# Side notes

* Jetson Nano
* Deep learning on FPGAs.
