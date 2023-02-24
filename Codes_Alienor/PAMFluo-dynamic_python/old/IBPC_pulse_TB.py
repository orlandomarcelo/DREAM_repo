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
from torch.utils.tensorboard import SummaryWriter

def analyse_IBPC(ni, output, time_array):

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
        start = start + 5
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
    

    ni.p.xlabel = "pulse number"
    ni.p.save_name = "IBPC_curve"
    fig = ni.p.plotting(np.array(range(len(list_mean))), np.array(list_mean))
    ni.p.saving(fig)
    

    return list_mean, list_blank, list_std, list_blank_std, list_pulses, list_ground, list_green



def send_IBPC(name = "IBPC_pulse", limit_blue = 50, limit_green =  200, limit_purple = 50, trigger_color = "green", sample_rate = 1e6, acq_time = 60, L = 200, D = 25):


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


    return ni, output, time_array


if __name__ == "__main__":

      

    name = "IBPC_pulse"

    limit_blue = 500
    limit_green = 200
    limit_purple = 50

    trigger_color = "green"
    sample_rate = 1e6
    acq_time = 20

    L = 200
    D = 45


    results = {}
    for limit_purple in [50, 100, 200, 300, 400]:
        ni, output, time_array = send_IBPC(name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, L, D)

        results[limit_purple] = analyse_IBPC(ni, output, time_array) 
        tsboard_writer = SummaryWriter("tensorboard/" + name + "/exp_%d"%limit_purple + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_')))  
       
        data_pulses = results[limit_purple]
        for i in range(len(data_pulses[0])):
            tsboard_writer.add_scalar('Mean', data_pulses[0][i], i)
            print(i, data_pulses[0][i])

        tsboard_writer.add_hparams(
            {"limit_blue": limit_blue, 
             "limit_green": limit_green,
             "limit_purple": limit_purple,
             
            "filter_blue": 0.5,
            "filter_green": 0,
            "filter_purple": 1.3,

            },
            {}
        )
        tsboard_writer.close()


    plt.show()
    ni.p.label_list = []
    result_mean = []
    for key in results.keys():
        list_mean = results[key][0]
        result_mean.append(np.array(list_mean))
        ni.p.label_list.append(str(key))
    
    pulses = np.array(range(len(list_mean)))
    fig = ni.p.plotting(pulses, result_mean)
 #   ni.p.saving(fig)

    plt.show()