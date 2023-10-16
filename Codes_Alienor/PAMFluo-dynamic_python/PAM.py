from alienlab.init_logger import logger
import time
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('PAM', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")



@ex.config
def my_config():
    name = "PAM"
    limit_blue = 90
    acq_time = 10*60



@ex.automain
def send_IBPC(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML):


    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    add_digital_pulse(link, pins['purple'], 30000, 30000, length_SP, 1) #saturating
    add_digital_pulse(link, pins['blue'], 40000, 10*minute, 5*minute, 1) #actinic
    add_master_digital_pulse(link, pins['green'], length_SP, period_ML, length_ML, 0) #measuring


    # Acquisition
    ni.detector_start()

    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    close_all_links(*all_links)


    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_green) 
    

    blank_level = np.mean(result[2][0:15])

    for i in range(len(result[0])):
        _run.log_scalar("Measure pulse", result[0][i] - blank_level, i)
        _run.log_scalar("Fluorescence", result[2][i], i)
        _run.log_scalar("MPPC", result[4][i], i)

    
