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


def send_activity(name, activities, limit_blue = 50, limit_green =  200, limit_purple = 50, trigger_color = "green", sample_rate = 1e6, acq_time = 60):


    # LED control tool DC4104 Thorlabs
    logger.name = name

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, limit_blue)
    ctrlLED.set_user_limit(LED_green, limit_green)
    ctrlLED.set_user_limit(LED_purple, limit_purple)

    ctrlLED.disconnect()

    # NATIONAL INSTRUMENTS card SCB-68A

    ni = init_NI_handler(name, trigger_color, sample_rate, acq_time)
    ni.detector_start()

    # Upload code to the Arduino

    link = Serial("COM4", 115200)
    time.sleep(2.0)
    reset_arduino(link)

    add_digital_pulse(link, pins['blue'], 15000, 30000, L, 1)
    add_digital_pulse(link, pins['purple'], 5000, 30000, 30000, 1)
    add_master_digital_pulse(link, pins['green'], L, 250, D, 0)


    # Acquisition
    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    ni.end_tasks()
    link.close()


    # OUTPUTS


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
    list_pulses= []
    list_ground = []
    list_green = []

    for stop in indices[1:]:
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

