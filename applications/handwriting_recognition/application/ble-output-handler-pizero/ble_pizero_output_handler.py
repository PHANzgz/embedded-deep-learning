import time
import PIL
import RPi.GPIO as GPIO
from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.oled.device import sh1106
import pygatt


class BLEPredictionHandler():

    # Text font
    fnt = PIL.ImageFont.truetype("Roboto-Regular.ttf", size=12)
    fnt_small = PIL.ImageFont.truetype("Roboto-Regular.ttf", size=9)
    # Constants
    TARGET_MAC_ADDRESS = 'C1:13:F3:34:9D:68'
    OUTPUT_HANDLER_SERVICE_UUID = "a29cee85-4fd7-4118-be91-f77d40db9362"
    PREDICTION_CHAR_UUID = "7f028bda-2032-4982-b1b7-7a121e70c6bc"
    AVG_SCORES_CHAR_UUID = "d10b1962-4a92-4432-90ac-ebe706701d33"
    BUTTON_PIN = 16

    def __init__(self, adapter, screen_device):
        self.adapter = adapter
        self.device = screen_device
        self.string_prediction = ""

        # GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) # Input with pull-up

        # BLE
        self.adapter.start()
        self.ble_device = adapter.connect(self.TARGET_MAC_ADDRESS, 
                                timeout=10, 
                                address_type=pygatt.BLEAddressType.random)
        print("Connected to BLE server")
        self.draw_output()


    def start(self):
        # Add callbacks
        GPIO.add_event_detect(self.BUTTON_PIN, GPIO.RISING, callback = self.on_button_press)
        # Prediction notfications
        self.ble_device.subscribe(self.PREDICTION_CHAR_UUID, callback = self.on_prediction_data)

    def stop(self):
        self.adapter.stop()
        GPIO.cleanup()

    def on_button_press(self, channel):
        # Reset string prediction
        self.string_prediction = ""
        self.draw_output()


    def on_prediction_data(self, handle, value):
        # Extract character from bytearray
        value = int.from_bytes(value, byteorder='big', signed = False)
        letter = chr( ord('A') + value)
        # Add letter to current string
        self.string_prediction += letter
        self.draw_output()
    

    def draw_output(self):

        with canvas(self.device) as draw:
            draw.text((3, 3), "Handwriting recognition", fill="white", font = self.fnt_small)
            draw.text((5, 30), self.string_prediction, fill="white", font = self.fnt)


    def __del__(self):
        self.adapter.stop()
        GPIO.cleanup()



def main():

    # Create serial interface, device and BLE adapter
    serial = spi(device = 0, port = 0)
    device = sh1106(serial, rotate=2) # rotate 180 degrees
    adapter = pygatt.GATTToolBackend()

    handler = BLEPredictionHandler(adapter, device)

    try:
        # Start handler
        handler.start()
        input("Press enter to stop program...\n")
        handler.stop()
    except KeyboardInterrupt:
        handler.stop()



if __name__ == "__main__":
    main()
