
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
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('Dronpa2_calibration', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():

    name = "Dronpa2_calibration"
    acq_time = 20
    limit_blue = 50
    limit_green = 50
    limit_purple = 50
    trigger_color = "purple"
    sample_rate = 1e4
    acq_time = 60*2

@ex.automain
def Dronpa2_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time):

    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)


    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]


    add_digital_pulse(link, pins[trigger_color], 1000, minute//2, minute//4, 0)
    add_digital_pulse(link, pins["blue"], 1000, 10*minute, 10*minute, 0)


    # Acquisition
    ni.detector_start()
    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    close_all_links(*all_links)


    #OUTPUTS


    tau_1, tau_2 = ni.get_tau(output[arduino_purple], output[fluorescence], time_array, 3.3, threshold_ticks = 0.5)

    sigma_480 = 198 #m2/mol
    sigma_405 = 415 #m2/mol

    I_480 = 1e6*(1/tau_1[1:].mean()-0.014)/sigma_480
    I_405 = 1e6*(1/tau_2.mean() - 0.014 -1/tau_1[1:].mean())/sigma_405 

    _run.log_scalar("I_480_mean", I_480, limit_blue)
    _run.log_scalar("I_405_mean", I_405, limit_purple)

    for i in range(len(tau_1)):
        _run.log_scalar("I_480", 1e6*(1/tau_1[i].mean()-0.014)/sigma_480, limit_blue)
    
    for i in range(len(tau_2)):
        _run.log_scalar("I_405", 1e6*(1/tau_2[i] - 0.014 -1/tau_1[1:].mean())/sigma_405 , limit_purple)

    ni.end_exp(__file__)
