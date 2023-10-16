import pandas as pd
from alienlab.init_logger import logger
from serial import *
from tqdm import tqdm
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg

from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *

"""Compare voltage output with NI command and arduino command. Result: 1V is 100mA"""

name = "SP_calibration"

# LED control tool DC4104 Thorlabs
logger.name = name

ctrlLED = ThorlabsDC4100(logger, port = "COM5")
ctrlLED.initialise_fluo()

ctrlLED.set_user_limit(LED_blue, 500)
ctrlLED.set_user_limit(LED_green, 500)
ctrlLED.set_user_limit(LED_purple, 500)

# NATIONAL INSTRUMENTS card SCB-68A

trigger_color = "blue"
sample_rate = 1e4
acq_time = 40
ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)

ni.trigger_init()

ni.generator_analog_channels = ["ao0"] #"Dev1/ao0:1"
ni.generator_digital_channels = [] #"Dev1/port0/line2"

V = 2
ni.excitation_frequency = 2
ni.points_per_period = 1000
ni.amplitude = V
ni.offset = V
ni.update_rates()
signal = ni.square_pattern()


ni.generator_analog_init(signal)
ni.trigger_init()

# Upload code to the Arduino

link = Serial("COM4", 115200)
time.sleep(2.0)
reset_arduino(link)

ctrlLED.set_user_limit(LED_blue, 2*V*100)
add_digital_pulse(link, pins['blue'], 100, 500, 250, 0)



# Acquisition
start_measurement(link)

time.sleep(10)
reset_arduino(link)



print("Switch source")

ni.task_generator_analog.start()
ni.task_trigger.start()

time.sleep(10)
ni.generator_stop()


ctrlLED.disconnect()



#output, time_array = ni.read_detector()
stop_measurement(link)

ni.end_tasks()
link.close()
