
import sys
sys.path.insert(0, "G:\DREAM/from_github/CIFRE_DREAM")
from AcquisitionClass import Acquisition

from config_DAQ import *
from ArduinoControl.python_serial import *

import serial
import time
import matplotlib.pyplot as plt


link = serial.Serial("COM4", 115200)
time.sleep(2.0)
    
pam = Acquisition()
pam.experiment_folder("test_arduino")
pam.p.save_folder = pam.save_folder

pam.generator_analog_channels = []
pam.generator_digital_channels = []
pam.detector_analog_channels = [detector_analog[intensity], detector_analog[fluorescence]]
pam.detector_digital_channels =  [detector_digital[i] for i in [arduino_blue, arduino_green, arduino_purple]]
pam.trig_chan_detector = trigger[arduino_green]
pam.trig_chan_detector_edge = 'RISING'

pam.sample_rate = 1e3 #Hz
pam.acq_time = 60 #s
pam.reading_samples = int(pam.sample_rate * pam.acq_time)

pam.detector_init()
pam.detector_start()

print("Testing the Digital pulse")

reset_arduino(link)

add_digital_pulse(link,12, 0, 1000, 500, 0)
add_digital_pulse(link, 10, 10, 1000, 500, 0)
start_measurement(link)
response = input("Is the blue light blinking ? [Y/n]")
if (response == "n"):
    sys.exit("Check blue light cables")
response = input("Is the green light blinking ? [Y/n]")
if (response == "n"):
    sys.exit("Check green light cables")

stop_measurement(link)
reset_arduino(link)
print("Reset")    

add_digital_pulse(link, pins["green"], 0, 1000, 500, 0)
add_digital_pulse(link, pins["blue"], 10, 1000, 500, 0)
start_measurement(link)
response = input("Is the blue light blinking ? [Y/n]")
if (response == "n"):
    sys.exit("Check blue light cables")
response = input("Is the green light blinking ? [Y/n]")
if (response == "n"):
    sys.exit("Check green light cables")

    

output, time_array = pam.read_detector()
stop_measurement(link)

plt.plot(output[arduino_green], label = "arduino_green")
plt.plot(output[arduino_blue], label = "arduino_blue")
plt.legend()
plt.show()
