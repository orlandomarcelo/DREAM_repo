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

"""Collect fluorescencce response over a wide range of actinic amplitude"""

name = "AL_calibration"

# LED control tool DC4104 Thorlabs
logger.name = name

ctrlLED = ThorlabsDC4100(logger, port = "COM5")
ctrlLED.initialise_fluo()

ctrlLED.set_user_limit(LED_blue, 0)
ctrlLED.set_user_limit(LED_green, 0)
ctrlLED.set_user_limit(LED_purple, 0)


# NATIONAL INSTRUMENTS card SCB-68A

trigger_color = "blue"
sample_rate = 1e4
acq_time = 400

ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)
ni.detector_start()

# Upload code to the Arduino

link = Serial("COM4", 115200)
time.sleep(2.0)
reset_arduino(link)

# Filter
fwl = FW102C(port=port_filter_wheel)
fwl.move_to_filter(0)
que.put(fwl)


# Acquisition

N_points = 10
D = 45
light_time = 1*sec

for actinic_level in np.linspace(50, 400, N_points):

    ctrlLED.set_user_limit(LED_blue, actinic_level)
    add_digital_pulse(link, pins["blue"], 0, 8*light_time, light_time, 0)
    start_measurement(link)
    time.sleep(2 * light_time / sec)
    stop_measurement(link)
    reset_arduino(link)

x = input("What filter")

for actinic_level in np.linspace(50, 400, N_points):

    ctrlLED.set_user_limit(LED_blue, actinic_level)
    add_digital_pulse(link, pins["blue"], 0, 8*light_time, light_time, 0)
    start_measurement(link)
    time.sleep(2 * light_time / sec)
    stop_measurement(link)
    reset_arduino(link)

x = input("What filter")

for actinic_level in np.linspace(50, 400, N_points):

    ctrlLED.set_user_limit(LED_blue, actinic_level)
    add_digital_pulse(link, pins["blue"], 0, 8*light_time, light_time, 0)
    start_measurement(link)
    time.sleep(2 * light_time / sec)
    stop_measurement(link)
    reset_arduino(link)

output, time_array = ni.read_detector()



ctrlLED.disconnect()
ni.end_tasks()
link.close()


# OUTPUTS
logger.info("Finished acquisition")


a = output[arduino_purple] + output[arduino_blue] + output[arduino_green]
b = np.roll(a, 1)
c = a-b

indices = np.linspace(0, len(c)-1, len(c)).astype(int)
indices = indices[c>=1]

start = indices[0]
list_mean = []
list_std = []
list_mppc = []
list_mppc_std = []
for stop in indices[1:]:
    measuring_pulse = a[start:stop]
    measuring_pulse = measuring_pulse>0
    blank = measuring_pulse == 0
    L_pulse = len(measuring_pulse)
    fluo = output[fluorescence, start:stop]
    fluo = fluo[measuring_pulse]

    mppc = output[intensity, start:stop]
    mppc = mppc[measuring_pulse]


    mean = np.mean(fluo)
    std = np.std(fluo)
    list_mean.append(mean)
    list_std.append(std)

    mean = np.mean(mppc)
    std = np.std(mppc)
    list_mppc.append(mean)
    list_mppc_std.append(std)    

    start = stop

start = indices[0]
list_pulses= []
list_purple = []

for stop in indices[1:]:
    measuring_pulse = a[start:stop]
    measuring_pulse = measuring_pulse>0

    L_pulse = len(measuring_pulse)
    print(L_pulse)

    list_pulses.append(output[fluorescence, start:stop])
    list_purple.append(a[start:stop])

    start = stop


ni.window = 100
average_output, downscaled_time = ni.averaging(output, time_array)



ni.p.label_list = detector
ni.p.xlabel = "time (s)"
ni.p.ylabel = "voltage (V)"
ni.p.save_name = "output_plot" 
fig = ni.p.plotting(downscaled_time, [average_output[i] for i in range(average_output.shape[0])])
ni.p.saving(fig)
plt.show()

ni.p.xlabel = "pulse number"
ni.p.save_name = "IBPC_curve"
fig = ni.p.plotting(np.array(range(len(list_mean))), np.array(list_mean))
ni.p.saving(fig)
plt.show()


k = 3
N = 100
u = mvgavg(list_pulses[k], N); v = mvgavg(list_purple[k], N); plt.plot(u);plt.plot(v);plt.show()
pos = list_purple[k]>0
val = list_pulses[k][pos]
print(list_std[k])
print(np.std(val))