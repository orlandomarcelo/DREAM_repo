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

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses, mean_by_indices
from Initialize_setup import start_and_stop, initialize, close_all_links

import ipdb

ex = Experiment('ML_calibration', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "ML_calibration"
    N = 3
    acq_time = 70*N
    trigger_color = "green"
    periods_ML = [2000, 250, 100, 250, 2000]


@ex.automain
def ML_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, periods_ML, N):
    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]
    voltint = all_links[4]

    ni.detector_start()

    lengths_ML = np.linspace(10, 60, N)
    for length_ML in lengths_ML:
        for period_ML in periods_ML:

            add_digital_pulse(link, pins[trigger_color], 0, period_ML, int(length_ML), 0)
            start_measurement(link)
            time.sleep(10.1 * period_ML // 1000)
            stop_measurement(link)
            reset_arduino(link)

    output, time_array = ni.read_detector()
    
    close_all_links(*all_links)


    # OUTPUTS
    logger.info("Finished acquisition")

    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_green) 

    split_step = (len(result[0])+1)//N


    np.savetxt(save_figure_folder + "id_%05d_%s_measure_pulse.csv"%(_run._id, name), np.array(result[0]), delimiter=",")
    np.savetxt(save_figure_folder + "id_%05d_%s_fluorescence.csv"%(_run._id, name), np.array(result[2]), delimiter=",")
    np.savetxt(save_figure_folder + "id_%05d_%s_MPPC_intensity.csv"%(_run._id, name), np.array(result[4]), delimiter=",")


    pulse_number = [0, 11, 13+11, 21+11+13, 13*2+11+21, 11*2+13*2+21]

    blank_level = np.mean(result[2])
    pulse_values = result[0] - blank_level

    for i in range(len(pulse_values)):
        ind = i//split_step
        j = i%split_step
        if j == 0 and i < len(pulse_values)-1:
            meaned_pulses = mean_by_indices(pulse_number, pulse_values[i: i + split_step])
            for k in range(len(meaned_pulses)):
                _run.log_scalar("Mean pulse %d"%lengths_ML[ind], meaned_pulses[k], periods_ML[k] )
            _run.log_scalar("ratio_end_begin", meaned_pulses[-1]/meaned_pulses[0], lengths_ML[ind] )
            _run.log_scalar("ratio_max_begin", meaned_pulses[2]/meaned_pulses[0], lengths_ML[ind] )



      
        _run.log_scalar("Measure pulse %d"%lengths_ML[ind], result[0][i], j)
        _run.log_scalar("Fluorescence %d"%lengths_ML[ind], result[2][i], j)
        _run.log_scalar("MPPC %d"%lengths_ML[ind], result[4][i], j)




