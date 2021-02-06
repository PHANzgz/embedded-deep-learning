# embedded-deep-learning
 Research and code examples about embedded deep learning applications in different areas and microcontrollers, like ESP32 or STM32 MCUs.

> **This repository is still under development and will be for a while, but if you are eager to see what is currently done feel free to browse.**  
Currently finished(pending revision):  
> * [Person and car detection app](/applications/person_detection/) (research and deployment)

# Demo apps

## Person and car detection (STM32F7, STM32H7, STM32F4, (TODO ESP32) )
To improve (mostly night) traffic and pollution, an embedded device with a camera is put on traffic lights. They serve a similar purpose as the buttons for the pedestrians: when the device doesn't detect a person in a time window, the traffic light may switch to flashing amber, allowing cars to traverse the crossing with precaution. This prevents the agglomeration of vehicles in crossings and, more importantly, saves gas(or battery) as the vehicle does not have to stop, just slow down. The immediate benefit is better circulation, but saving both money and gas are also direct consequences. This in turn decreases the amount of CO2 released by vehicles and provides a very positive environmental impact.

### [Learn more](/applications/person_detection/)
![demo_gif](/applications/person_detection/demo_webapp/img/demo.gif)

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
* Edgeimpulse (?)

# Side notes

* Jetson Nano
* Deep learning on FPGAs.
