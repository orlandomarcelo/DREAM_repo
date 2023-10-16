from concurrent.futures.process import _threads_wakeups
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
from skimage.transform import rescale, resize, downscale_local_mean


from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera

from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from scipy import optimize


import ipdb

"""Obervation of the temporal response to a light jump: OJIP curve"""

ex = Experiment('OJIP_pulse', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "OJIP_pulse"
    camera_heat_delay = 250 #s
    trigger_color = "blue"
    acq_time = 1
    sample_rate = 2e6
    limit_blue= 300
    actinic_filter=1
    x0=200
    y0=200
    height=200
    width=200
    clock=474
    binning_factor=2
    subsampling_factor=8
    
    trigger_freq = 350 #Hz
    exposure= 1000/trigger_freq - 0.1 #ms 

@ex.automain
def OJIP(_run, _log, name, limit_blue, limit_green, limit_purple, limit_red, trigger_color, 
         sample_rate, acq_time, length_SP, length_ML, period_ML, actinic_filter, height, width, x0, y0, subsampling_factor, binning_factor,
         clock, trigger_freq, exposure, camera_heat_delay):
    
    def exp_decay(parameters, xdata):
        '''
        Calculate an exponential decay of the form:
        S= a * exp(-xdata/b)
        '''
        A = parameters[0]
        tau = parameters[1]
        y0 = parameters[2]
        return A * np.exp(-xdata/tau) + y0

    def residuals(parameters, x_data, y_observed, func):
        '''
        Compute residuals of y_predicted - y_observed
        where:
        y_predicted = func(parameters,x_data)
        '''
        return func(parameters,x_data) - y_observed

    @delayed
    @wrap_non_picklable_objects
    def get_tau(fluo, window_fit, time_array, volt, sample_rate, 
                residuals, exp_decay, threshold_ticks = 1e0, averaging = False, 
                N_avg = 0, plot = False):
        first=1
        second = first + window_fit
        fluo_transition = fluo[first:second]
        time_transition = np.linspace(0, second - first - 1, second - first)
        x0 = [1e5, (second-first)/8, 1]
        OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                            args = (time_transition, fluo_transition, exp_decay))
        parameters_estimated = OptimizeResult.x
        tau = parameters_estimated[1]
        
        #conditions on tau too low or too high for the second, more accurate, fit, because we will fit on a signal that lasts 5*tau
        if tau >  (second-first)//10: #if too high
            tau =  (second-first)//10
        if tau < 2: #if too low, increase it
            tau = 3
        x0 = parameters_estimated #initial guess: parameters from previous fit
        #second fit
        OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                            args = (time_transition[0:int(tau*5)], fluo_transition[0: int(tau*5)], exp_decay))
        parameters_estimated = OptimizeResult.x
    

        # How good are the parameters I estimated?
        parameters_estimated[1] = parameters_estimated[1]/sample_rate
        return parameters_estimated

    
    
    
    name = "OJIP_pulse"

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
    thread, cam = init_camera(exposure = exposure, num_frames = trigger_freq*acq_time, gain =  100, x=x0, height=height,
                              y=y0, width=width, subsampling_factor=subsampling_factor, clock=clock, binning_factor=binning_factor, trigger_frequency=trigger_freq, trigger=False)

    add_digital_pulse(link, pins['no_LED_trig'], 15+camera_heat_delay*sec, sec/trigger_freq, 20, 1) #trigger
    add_digital_pulse(link, pins[trigger_color], 15+camera_heat_delay*sec, 4*minute, 4*minute, 0)


    ni.detector_start()

    # Acquisition
    start_measurement(link)
    time.sleep(camera_heat_delay)
    thread.start()


    output, time_array = ni.read_detector()
    stop_measurement(link)
    thread.stop_thread = True
    thread.join()
    cam.exit()

    video, timing, L, H  =  cam.return_video(ni)
    
    ni.end_tasks()
    link.close()


    #Results analysis

    ni.window = 500
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
    fig = ni.p.plotting(time_array, output[fluorescence])
    ni.p.saving(fig)
    #plt.show()
    _run.add_artifact((ni.p.save_path + ni.p.extension))
  
    fluo = average_output[fluorescence]
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
    close_all_links(*all_links)
    
    t = 1000*(np.array(timing)-timing[0])
    plt.plot(t, np.mean(video, axis=(1,2)))
    plt.savefig(ni.save_folder + '/OJIP.pdf')
    plt.figure()
    plt.plot(np.mean(video, axis=(1,2)))

    plt.show()
    window_fit = int(input())

    
    video_downscaled = []
    for i in range(video.shape[0]):
        video_downscaled.append(downscale_local_mean(video[i], (10, 10)))
    video_downscaled = np.array(video_downscaled)
    init_shape = np.copy(video_downscaled.shape)
    L, H = init_shape[1:]
    video_downscaled = video_downscaled.reshape(video_downscaled.shape[0], -1)
    i = video_downscaled.shape[1]//2
    

    trig = output[arduino_blue]

    mean_trace = np.mean(video, axis=(1,2))
    params_init = Parallel(n_jobs = -1 )(get_tau(video_downscaled[:,j], window_fit, time_array, 3.3,
        trigger_freq, residuals, exp_decay, 0.5) for j in [i])
    fluo_trace = video_downscaled[1:,i]
    ipdb.set_trace()
    param_copy = np.copy(params_init[0])
    param_copy[1] *= trigger_freq
    plt.plot(exp_decay(param_copy, np.linspace(0, len(fluo_trace)-1, len(fluo_trace))))
    plt.plot(fluo_trace, '.')
    plt.show()
    
    params = Parallel(n_jobs = -1 )(get_tau(video_downscaled[:,i], window_fit, time_array,
                                              3.3, trigger_freq, residuals, exp_decay, 0.5) for i in range(video_downscaled.shape[1]))
    ipdb.set_trace()
    params = np.array(params)
    tau_list = params[:,1]
    amp_list = params[:,0] 
    tau_list = np.array(tau_list).reshape(init_shape[1:])
    amp_list = np.array(amp_list).reshape(init_shape[1:])
    
    m = np.copy(amp_list)
    m[m<100]=0
    l = 1/np.copy(tau_list)
    l[l>1000]=0
    l[l<0]=0
    if True: 
        fig, axs = plt.subplots(1,3, figsize=(15, 6))
        axs[0].imshow(l)
        axs[1].imshow(m)
        axs[2].imshow(video[3])
        plt.savefig(ni.save_folder + '/tau_image.pdf')
        plt.show()    
    ipdb.set_trace()

"""
if True:
    window_fit=40
    kk=30
    params = get_tau(video_downscaled[:,kk], window_fit, time_array,3.3,trigger_freq, residuals, exp_decay, 0.5)
    params[1]*=trigger_freq
    plt.plot(exp_decay(params, np.linspace(0, len(fluo_trace)-1, len(fluo_trace))))
    plt.plot(video_downscaled[:,kk])
    plt.show()
    
"""