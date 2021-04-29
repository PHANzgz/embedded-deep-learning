# Online handwriting recognition

(TODO)

With an accelerometer as a pen cap, data is sampled continuously and filtered. The data is fed to the convolutional neural network which also runs continuously with the latest two seconds of accelerometer data. Once a letter is written on paper and a prediction is made, the information is sent over BLE. 

# A homemade demo, running on an nRF52840
In this visual demo, the output BLE data is received by a Raspberry Pi Zero with a little screen on top of it
showing the current string.  

Preprocessing takes approx 10ms.  
Inference takes approx 62ms.  

![demo](demo.gif)
<br><br>