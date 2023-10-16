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
import time as TIMING
from ArduinoControl import python_comm

from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera


import ipdb

"""Obervation of the temporal response to a light jump: OJIP curve"""

ex = Experiment('OJIP_pulse', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "OJIP_pulse"

    trigger_color = "no_LED_trig"
    color = "blue"
    acq_time = 1.3
    sample_rate = 3e6
    limit_blue= 0
    limit_purple=0
    limit_red=0
    limit_green=0
    actinic_filter=0
    move_plate = False
    move=100


@ex.automain
def OJIP(_run, _log, name, limit_blue, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, actinic_filter, move_plate, move, color):

    # LED control tool DC4104 Thorlabs
    logger.name = name
    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo = True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])
    ni.detector_start()

    add_digital_pulse(link, pins[trigger_color], 0, 4*minute, 4*minute, 0)
    add_digital_pulse(link, pins[color], 30, 4*minute, 4*minute, 0)


    # Acquisition
    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    ni.end_tasks()
    link.close()


    #Results analysis

    ni.window = 50
    average_output, downscaled_time = ni.averaging(output, time_array)
    

    ni.p.label_list = detector
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "output_plot" 
    fig = ni.p.plotting(downscaled_time, [average_output[i] for i in range(average_output.shape[0])])
    ni.p.saving(fig)
    _run.add_artifact((ni.p.save_path + ni.p.extension))

    #plt.show()

    ni.p.ylog = "semilogx"
    ni.p.save_name = "ojip_curve"
    F = output[fluorescence]
    ind = F/F.max()>0.1
    timea = time_array[ind]
    timea -= timea[0]
    fig = ni.p.plotting(timea, output[fluorescence][ind])
    ni.p.saving(fig)
    #plt.show()
    _run.add_artifact((ni.p.save_path + ni.p.extension))
  
    fluo = average_output[fluorescence]
    ind = fluo/fluo.max()>0.1
    logger.critical(len(fluo))
    fluo=fluo[ind]
    logger.critical(len(fluo))
    j = 0

    while j < len(fluo):
        window = 1 + int(np.log(j+ 1))
        data = fluo[j:j+window]
        _run.log_scalar("OJIP_response", data.mean(), np.log(j))
    
        j = j + window
        print(j)
        
    for j in range(len(fluo)//10):
        _run.log_scalar("OJIP_response_linear", fluo[10*j], j)

    f = open(ni.p.save_folder + "/intensity_OJIP.txt", "w")
    f.write(str(limit_blue) + ',' + str(actinic_filter))
    f.close()
    
    
    plt.close("all")
    
    if move_plate:
        print("platform movement")
        motors = all_links[4]
        python_comm.move_dx(motors, move)
        TIMING.sleep(2)
        #ipdb.set_trace()
        #python_comm.move_dy(motors, 200)

    close_all_links(*all_links)
    #ipdb.set_trace()


