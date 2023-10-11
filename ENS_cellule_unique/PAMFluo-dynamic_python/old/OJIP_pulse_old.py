
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


name = "OJIP_pulse"

# LED control tool DC4104 Thorlabs
logger.name = name

ctrlLED = ThorlabsDC4100(logger, port = "COM5")
ctrlLED.initialise_fluo()

ctrlLED.set_user_limit(LED_blue, 400)
ctrlLED.set_user_limit(LED_green, 200)
ctrlLED.set_user_limit(LED_purple, 200)

ctrlLED.disconnect()

# NATIONAL INSTRUMENTS card SCB-68A

trigger_color = "blue"
sample_rate = 2e6
acq_time = 60

ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)
ni.detector_start()

# Upload code to the Arduino

link = Serial("COM4", 115200)
time.sleep(2.0)
reset_arduino(link)

add_digital_pulse(link, pins[trigger_color], 1000, 4*minute, 4*minute, 0)


# Acquisition
start_measurement(link)
output, time_array = ni.read_detector()
stop_measurement(link)

ni.end_tasks()
link.close()


#Results analysis

ni.window = 1000
average_output, downscaled_time = ni.averaging(output, time_array)


ni.p.label_list = detector
ni.p.xlabel = "time (s)"
ni.p.ylabel = "voltage (V)"
ni.p.save_name = "output_plot" 
fig = ni.p.plotting(downscaled_time, [average_output[i] for i in range(average_output.shape[0])])
ni.p.saving(fig)

plt.show()

ni.p.ylog = "semilogx"
ni.p.save_name = "ojip_curve"
fig = ni.p.plotting(downscaled_time, average_output[fluorescence])
ni.p.saving(fig)
plt.show()



