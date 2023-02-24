

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

import ipdb

ex = Experiment('SP_decay', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "SP_decay"
    N = 1
    acq_time = 20*N

    limit_purple=150
    length_ML = 300
    trigger_color = "purple"


@ex.automain
def ML_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, N):
    # LED control tool DC4104 Thorlabs
    logger.name = name

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, limit_blue)
    ctrlLED.set_user_limit(LED_green, limit_green)
    ctrlLED.set_user_limit(LED_purple, limit_purple)


    # NATIONAL INSTRUMENTS card SCB-68A

    ni = init_NI_handler(name, trigger_color, sample_rate, acq_time, _run)
    ni.detector_start()

    # Upload code to the Arduino

    link = Serial("COM4", 115200)
    time.sleep(2.0)
    reset_arduino(link)

    # Acquisition

    ctrlLED.set_user_limit(LED_green, limit_green)


    lengths_ML = [length_ML]
    for length_ML in lengths_ML:
            add_digital_pulse(link, pins['purple'], 100, 12000, length_SP, 1)
            add_master_digital_pulse(link, pins['green'], 100 + length_SP + 1, 6000, length_ML, 0)
            start_measurement(link)
            time.sleep(20)
            stop_measurement(link)
            reset_arduino(link)

    output, time_array = ni.read_detector()
    
    ctrlLED.disconnect()
    ni.end_tasks()
    link.close()


    # OUTPUTS
    logger.info("Finished acquisition")

    tau_1, tau_2 = ni.get_tau(output[arduino_green], output[fluorescence], 
                        time_array, length_ML, threshold_ticks = 0.1, 
                        averaging = True, N_avg = 50)

    ipdb.set_trace()

    decays = output[fluorescence][output[arduino_green]>0]
    time_decays = time_array[0:len(decays)]
    window = 10000
    decays = mvgavg(decays, window)
    time_decays = mvgavg(time_decays, window)*1000

    fig = ni.p.plotting(time_decays, decays)
    ni.p.saving(fig)
    _run.add_artifact(ni.p.save_path + ni.p.extension)

    for i in range(len(tau_1)):
        _run.log_scalar("tau ", tau_1[i]/sample_rate, i)


