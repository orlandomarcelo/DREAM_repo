import pandas as pd
from alienlab.init_logger import logger
from serial import *
from tqdm import tqdm
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg
import datetime

from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses

import ipdb

ex = Experiment('SP_calibration', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "SP_calibration"
    trigger_color = "purple"

    N = 15
    acq_time = 11 * N



@ex.automain
def SP_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, N):


    # LED control tool DC4104 Thorlabs
    logger.name = name

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, limit_blue)
    ctrlLED.set_user_limit(LED_green, limit_green)
    ctrlLED.set_user_limit(LED_purple, limit_purple)

     # NATIONAL INSTRUMENTS card SCB-68A

    ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)
    ni.detector_start()

    # Upload code to the Arduino

    link = Serial("COM4", 115200)
    time.sleep(2.0)
    reset_arduino(link)

    add_digital_pulse(link, pins['purple'], 100, 2000, length_SP, 1)
    add_master_digital_pulse(link, pins['green'], 100 + length_SP + 1, 1000, length_ML, 0)

    ctrlLED.set_user_limit(LED_purple, limit_purple)
    start_measurement(link)

    limits_purple = np.linspace(10, 500, N)
    for limit_purple in limits_purple:
        ctrlLED.set_user_limit(LED_purple, int(limit_purple))
        try: 
            print(crtlLED.get_user_limit(LED_purple))
        except:
            pass
        time.sleep(15.1)

    output, time_array = ni.read_detector()
    stop_measurement(link)

    ctrlLED.disconnect()
    ni.end_tasks()
    link.close()


    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_green) 
    

    #for i in range(len(result[0])):
    #    _run.log_scalar("Measure pulse", result[0][i], i)
    #    _run.log_scalar("Fluorescence", result[2][i], i)
    #    _run.log_scalar("MPPC", result[4][i], i)

    fig = plt.figure()
    pulse_green = np.copy(result[0][::2])
    pulse_purple = np.copy(result[4][1::2])
    pulse_fluo_jump = np.copy(result[2][1::2])
    plt.plot(pulse_green, label = 'pulse green')
    plt.plot(pulse_purple, label = "pulse purple")

    pulse_green_norm = (pulse_green-pulse_green.min())/(pulse_green.max()-pulse_green.min())
    pulse_purple_norm = (pulse_purple-pulse_purple.min())/(pulse_purple.max()-pulse_purple.min())
    pulse_fluo_norm = (pulse_fluo_jump-pulse_fluo_jump.min())/(pulse_fluo_jump.max()-pulse_fluo_jump.min())

    plt.close("all")

    ni.p.save_name = "normalised_intensity_vs_response"
    ni.p.saving(fig)
    _run.add_artifact(ni.p.save_path + ni.p.extension)

    blank_level = np.mean(result[2][::2]).mean()

    for i in range(len(pulse_green)-1):
        _run.log_scalar("Fm response", pulse_green[i], pulse_purple[i])
        _run.log_scalar("Fluorescence response", pulse_fluo_jump[i], pulse_purple[i])

        _run.log_scalar("Pulse green norm", pulse_green_norm[i], i)
        _run.log_scalar("Pulse purple norm", pulse_purple_norm[i], i)
        _run.log_scalar("Pulse fluo norm", pulse_fluo_norm[i], i)

        _run.log_scalar("Pulse green", pulse_green[i], i)
        _run.log_scalar("Pulse purple", pulse_purple[i], i)
        _run.log_scalar("Pulse fluo jump", pulse_fluo_jump[i], i)

    ipdb.set_trace()


"""
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
"""