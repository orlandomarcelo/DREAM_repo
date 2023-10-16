
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
from   scipy import optimize


from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('LED_dynamics', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "LED_dynamics"
    acq_time = 0.5
    limit_blue = 500
    limit_green = 500
    limit_purple = 500
    sample_rate = 2e6
    actinic_filter = 1
    trigger_color = "green"

@ex.automain
def send_IBPC(_run, name, limit_blue, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, actinic_filter):

    def sigmoid(parameters, xdata):
        '''
        Calculate an exponential decay of the form:
        S= a * exp(-xdata/b)
        '''
        A = parameters[0]
        tau = parameters[1]
        y0 = parameters[2]
        t0 = parameters[3]
        return A * (1 / (1 + np.exp(-(xdata-t0)/tau))) + y0

    def residuals(parameters, x_data, y_observed, func):
        '''
        Compute residuals of y_predicted - y_observed
        where:
        y_predicted = func(parameters,x_data)
        '''
        return func(parameters,x_data) - y_observed    

    def make_fit(x0, time_transition, fluo_transition):
        OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                                args = (time_transition, fluo_transition, sigmoid))
        parameters_estimated = OptimizeResult.x
        
        y_data_predicted = sigmoid(parameters_estimated, time_transition)
        return parameters_estimated, y_data_predicted

                

    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, limit_red=limit_red,
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])


    add_digital_pulse(link, pins['purple'], 30, 120, 20, 0)
    add_digital_pulse(link, pins['blue'], 70, 120, 20, 0)
    add_master_digital_pulse(link, pins['green'], 0, 120, 20, 0)


    # Acquisition
    start_measurement(link)
    output, time_array = ni.read_detector()
    stop_measurement(link)

    ni.end_tasks()
    link.close()

    ni.p.label_list = detector
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "output_plot" 
    fig = ni.p.plotting(time_array, [output[i] for i in range(output.shape[0])])
    ni.p.saving(fig)
    _run.add_artifact((ni.p.save_path + ni.p.extension))
    NI_curves = {}

    d = 3000
    x0 = [0.01,0.01,0.01, 4]
    NI_curves[arduino_purple] = {}
    curve = output[:, 59500:59500+d]
    NI_curves[arduino_purple]["up_y"] = curve
    NI_curves[arduino_purple]["down_y"] = output[:, 99500:99500+d]

    NI_curves[arduino_blue] = {}
    NI_curves[arduino_blue]["up_y"] = output[:, 139500:139500+d]
    NI_curves[arduino_blue]["down_y"] = output[:, 179500:179500+d]

    NI_curves[arduino_green] = {}
    NI_curves[arduino_green]["up_y"] = output[:, 239500:239500+d]
    NI_curves[arduino_green]["down_y"] = output[:, 279500:279500+d]   

    for part in ["up_y", "down_y"]:
        fig, axs = plt.subplots(3)
        for i, k in enumerate(NI_curves.keys()):
            MPPC_intensity = NI_curves[k][part][intensity]
            L = len(MPPC_intensity)
            x_axis = np.linspace(0, L-1, L)/sample_rate*1e3
            plt.xlabel("time (ms)")
            axs[i].plot(x_axis, MPPC_intensity)#, label = detector[k])
            param, pred = make_fit(x0, x_axis, MPPC_intensity)
            axs[i].plot(x_axis, pred, "--", label = "tau = %0.2e s"%(1e-3*param[1]))

            arduino_intensity = NI_curves[k][part][k]
            axs[i].plot(x_axis, arduino_intensity * MPPC_intensity.max()/arduino_intensity.max(), label = "arduino")


            fluo_intensity = NI_curves[k][part][fluorescence]
            axs[i].plot(x_axis, fluo_intensity * MPPC_intensity.max()/fluo_intensity.max(), label = "fluorescence")
            param, pred = make_fit(x0, x_axis, fluo_intensity)
            axs[i].plot(x_axis, pred, "--", label = param[1])
            axs[i].legend() 
        ni.p.save_name = part
        ni.p.saving(fig)
        _run.add_artifact((ni.p.save_path + ni.p.extension))        


    plt.show()

    ipdb.set_trace()