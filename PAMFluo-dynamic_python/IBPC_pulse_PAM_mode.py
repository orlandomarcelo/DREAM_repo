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
from Ingredients import pulse_ingredient, read_pulses, lock_in
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('IBPC_pulse', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "IBPC_pulse"
    acq_time = 20
    limit_blue = 10
    period_ML = 50
    length_ML = 3
    actinic_filter = 0

@ex.automain
def send_IBPC(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML,actinic_filter):


    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])


    add_digital_pulse(link, pins['purple'], 15000, 30000, length_SP, 0)
    add_digital_pulse(link, pins['blue'], 5000, 30000, 30000, 0)
    add_master_digital_pulse(link, pins['green'], length_SP + 1, period_ML, length_ML, 0)


    # Acquisition
    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    ni.end_tasks()
    link.close()

    sin_lock, cos_lock, radius_lock, phase_lock = lock_in(output, time_array, 1/(50*1e-3), 0, 100000)
    ipdb.set_trace()



    ni.window = 150000
    sin, t = ni.averaging(sin_lock, time_array)
    cos, t = ni.averaging(cos_lock, time_array)
    
    radius = np.sqrt(sin[1]**2 + cos[1]**2)
    plt.plot(radius)
        

    for i in range(len(radius)):
        _run.log_scalar("Modulated amplitude", radius[i], t[i])
