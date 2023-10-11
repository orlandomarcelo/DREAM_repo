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


name = "SP_calibration"

# LED control tool DC4104 Thorlabs
logger.name = name

ctrlLED = ThorlabsDC4100(logger, port = "COM5")
ctrlLED.initialise_fluo()

ctrlLED.set_user_limit(LED_blue, 50)
ctrlLED.set_user_limit(LED_green, 500)
ctrlLED.set_user_limit(LED_purple, 500)

# NATIONAL INSTRUMENTS card SCB-68A

trigger_color = "purple"
sample_rate = 1e4
acq_time = 60*3

ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)
ni.detector_start()

# Upload code to the Arduino

link = Serial("COM4", 115200)
time.sleep(2.0)
reset_arduino(link)

L = 200
D = 45
add_digital_pulse(link, pins['purple'], 100, 2000, L, 1)
add_master_digital_pulse(link, pins['green'], 100 + L, 1000, D, 0)



# Acquisition
start_measurement(link)

N_step = 30
step = 500/N_step
for i in range(N_step):
    ctrlLED.set_user_limit(LED_purple, step*i)
    try: 
        print(crtlLED.get_user_limit(LED_purple))
    except:
        pass
    time.sleep(5)
ctrlLED.disconnect()



output, time_array = ni.read_detector()
stop_measurement(link)

ni.end_tasks()
link.close()



# OUTPUTS
logger.info("Finished acquisition")
#arduino_green = arduino_purple
a = output[arduino_green]
b = np.roll(output[arduino_green], 1)
c = a-b

indices = np.linspace(0, len(c)-1, len(c)).astype(int)
indices = indices[c>=1]

start = indices[0]
list_mean = []
list_std = []
list_blank = []
list_blank_std = []
for stop in indices[1:]:
    start = start
    measuring_pulse = output[arduino_green, start:stop]
    measuring_pulse = measuring_pulse>0
    blank = measuring_pulse == 0
    L_pulse = len(measuring_pulse)
    fluo = output[fluorescence, start:stop]
    fluo = fluo[measuring_pulse]
    fluo_blank = output[fluorescence, start:stop]
    fluo_blank = fluo_blank[blank]
    mean = np.mean(fluo)
    std = np.std(fluo)
    list_mean.append(mean)
    list_std.append(std)

    list_blank.append(np.mean(fluo_blank))
    list_blank_std.append(np.std(fluo_blank))

    start = stop

start = indices[0]
list_pulses= []
list_ground = []
list_green = []

for stop in indices[1:]:
    start = start

    measuring_pulse = output[arduino_green, start:stop]
    measuring_pulse = measuring_pulse>0

    L_pulse = len(measuring_pulse)
    print(L_pulse)
    if L_pulse > 2000:

        list_pulses.append(output[fluorescence, start:stop])
        list_ground.append(output[fluorescence, start:stop])
        list_green.append(output[arduino_green, start:stop])

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



def plot_pulse(k, N):
    u = mvgavg(list_pulses[k], N); v = mvgavg(list_green[k], N); plt.plot(u, label = "%d"%k);plt.plot(v)
    plt.legend()

N = 100
k = 4
plot_pulse(5, N)
plot_pulse(7, N)
plot_pulse(12, N)
plt.show()
pos = list_green[k]>0
val = list_pulses[k][pos]
print(list_std[k])
print(np.std(val))

nev = list_mean[1::2]
eve = list_mean[::2]
fm = np.mean(eve)
fo = np.mean(nev)
(fm-fo)/fm

bis = (np.array(eve)-np.array(nev))/np.array(eve)

plt.plot(bis)
plt.show()

fig = plt.figure()
m = np.copy(list_mean[::2])
b = np.copy(list_blank[1::2])
plt.plot((b-b.min())/(b.max()-b.min()))
plt.plot((m-m.min())/(m.max()-m.min()))
plt.show()

ni.p.save_name = "normalised_intensity_vs_response"
ni.p.saving(fig)
