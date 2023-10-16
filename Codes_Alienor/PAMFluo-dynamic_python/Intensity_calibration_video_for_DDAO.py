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
from skimage.transform import rescale, resize, downscale_local_mean
import tifffile as tiff

from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from scipy import optimize

from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera

import ipdb

@start_and_stop.config
def update_cfg():
    initial_filter = filters[0.5]

ex = Experiment('D2_calib_video', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "D2_calib_video"
    acq_time = 16*60//5
    sample_rate = 2
    trigger_color = "purple"
    limit_purple = 100
    limit_blue = 400
    exposure = 100
    focus_pos = 0

@ex.automain
def send_IBPC(_run, _log, name, limit_blue, limit_green, limit_red, limit_purple, trigger_color, sample_rate, acq_time, 
        length_SP, length_ML, period_ML, exposure, focus_pos):




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
    def get_tau(trig, fluo, time_array, volt, points_per_period, sample_rate, residuals, exp_decay, threshold_ticks = 1e0, averaging = False, N_avg = 0, plot = False):
#if True:
#        threshold_ticks = 5
#        points_per_period=60
    
        trig_shift = np.roll(trig, 1)
        diff = trig-trig_shift
        tau_list = []
        
        trig_diff = np.abs(diff) > threshold_ticks #PARAMETER HERE

        ticks = np.nonzero(trig_diff)[0].tolist()
        ticks.append(-1)
        first = ticks[0]
        for second in ticks[1:]:
            if second - first < (points_per_period // 2):  #PARAMETER HERE
                ticks.remove(second)
            else: 
                first = second
        
        print(ticks)
        first = ticks[0]
        for second in ticks[1:]:
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
            if tau < 3: #if too low, increase it
                tau = 5
            x0 = parameters_estimated #initial guess: parameters from previous fit
            #second fit
            OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                                args = (time_transition[0:int(tau*5)], fluo_transition[0: int(tau*5)], exp_decay))
            parameters_estimated = OptimizeResult.x
        
            #plt.plot(time_transition+first, fluo_transition)
            #plt.plot(time_transition+first, exp_decay(parameters_estimated, time_transition))
            # How good are the parameters I estimated?
            tau_list.append(parameters_estimated[1]*(time_array[-1]-time_array[0])/len(time_array))
            first = second

        tau_1 = tau_list[0::2]
        tau_2 = tau_list[1::2]
        print(tau_1)
        print(tau_2)
        return [np.asarray(tau_1[1:-1]).mean(), np.asarray(tau_1[1:-1]).std(), np.asarray(tau_2[:-1]).mean(), np.asarray(tau_2[:-1]).std()]



    # LED control tool DC4104 Thorlabs
    logger.name = name

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, limit_red = limit_red,
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo = True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]
    KP = all_links[5]
    
    KP.SetOutputVoltage(focus_pos)


    ni.mongo = False

    points_per_period = 60//2
    ni.points_per_period = points_per_period
    add_digital_pulse(link, pins[trigger_color], 1000, 4*minute//4, minute//4, 0)
    add_digital_pulse(link, pins["blue"], 1000, 10*minute, 10*minute, 0)
    add_digital_pulse(link, pins["no_LED_trig"], 1000, 1/sample_rate*1000, 20, 0)


    thread, cam = init_camera(exposure = exposure, num_frames = acq_time*sample_rate, gain = 80)


    # Acquisition
    thread.start()
    ni.detector_start()
    start_measurement(link)


    output, time_array = ni.read_detector()
    thread.stop_thread = True
    thread.join()
    cam.exit()

    stop_measurement(link)

    close_all_links(*all_links)
    cam.exit()


    try: #color camera vs grey
        L, H = cam.video[0].shape
        video = np.array(cam.video)

    except:
        L, H, d = cam.video[0].shape
        video = np.array(cam.video)[:,:,:, 0]


    tiff.imwrite(ni.save_folder + "/video.tiff", video, photometric='minisblack')

    timing = cam.timing
    np.save(ni.save_folder + "/video_timing.npy", timing)
    tau_480_map = np.zeros((L, H))
    tau_405_map = np.zeros((L, H))


    video_downscaled = []
    for i in range(video.shape[0]):
        video_downscaled.append(downscale_local_mean(video[i], (10, 10)))
    video_downscaled = np.array(video_downscaled)
    init_shape = np.copy(video_downscaled.shape)
    L, H = init_shape[1:]
    video_downscaled = video_downscaled.reshape(video_downscaled.shape[0], -1)

    ni.p.save_name = "curve_on_off"
    ni.p.xlabel = "time"
    ni.p.ylabel = 'fluo'
    fig = ni.p.plotting(np.linspace(0, len(video_downscaled), len(video_downscaled)), np.mean(video_downscaled, axis = 1))
    ni.p.saving(fig)

    trig = output[arduino_purple]
    points_per_period = 60//2
    
 
    tau_lists = Parallel(n_jobs = -1 )(get_tau(trig, video_downscaled[:,i], time_array, 3.3, points_per_period, sample_rate, residuals, exp_decay, 0.5) for i in range(video_downscaled.shape[1]))
    #for i in range(L):        
    #    for j in range(H):
    #      tau_1, tau_2 = ni.get_tau(output[arduino_purple], video_downscaled[:,i,j], time_array, 3.3, threshold_ticks = 0.5)
    #      tau_480_map[i,j], tau_405_map[i,j] = tau_1[1:].mean(), tau_2.mean()

    tau_lists = np.array(tau_lists)
    tau_480_map = tau_lists[:,0]
    tau_480_std = tau_lists[:,3]
    tau_405_map = tau_lists[:,2]
    tau_405_std = tau_lists[:,1]

    sigma_480 = 198 #m2/mol
    sigma_405 = 415 #m2/mol

    I_480_map = 1e6*(1/tau_480_map - 0.014)/sigma_480
    I_405_map = 1e6*(1/tau_405_map - 0.014 -1/tau_480_map)/sigma_405 

    I_480_map = I_480_map.reshape(init_shape[1:])
    I_405_map = I_405_map.reshape(init_shape[1:])

    I_480_map[I_480_map <=0 ] = 1
    I_480_map[I_480_map >= 1.5*np.median(I_480_map) ] = 1 


    I_405_map[I_405_map <=0 ] = 1
    I_405_map[I_405_map >= 1.5*np.median(I_405_map) ] = 1 
    
    fig = plt.figure()
    plt.imshow(I_480_map) 
    plt.colorbar(orientation="horizontal")


    ni.p.save_name = "I_480_map"
    ni.p.saving(fig)
    np.save(ni.p.save_path[:-4]+'.npy', I_480_map)
    fig = plt.figure()
    plt.imshow(I_405_map)
    plt.colorbar(orientation="horizontal")


    ni.p.save_name = "I_405_map"
    ni.p.saving(fig)
    np.save(ni.p.save_path[:-4]+'.npy', I_405_map)

    fig = plt.figure()
    plt.imshow(np.log(I_480_map)) 
    plt.colorbar(orientation="horizontal")

    ni.p.save_name = "I_480_map_log"
    ni.p.saving(fig)
    fig = plt.figure()
    plt.imshow(np.log(I_405_map))
    plt.colorbar(orientation="horizontal")

    ni.p.save_name = "I_405_map_log"
    ni.p.saving(fig)


    fig = plt.figure()
    plt.plot(I_480_map[[20,25,30], :].T)
    ni.p.save_name = "tranche_480_x"
    ni.p.saving(fig)
    fig = plt.figure()
    plt.plot(I_480_map[:, [40,50,60]])
    ni.p.save_name = "tranche_480_y"
    ni.p.saving(fig)
    fig = plt.figure()
    plt.plot(I_405_map[[20,25,30], :].T)
    ni.p.save_name = "tranche_405_x"
    ni.p.saving(fig)
    fig = plt.figure()
    plt.plot(I_405_map[:, [40,50,60]])
    ni.p.save_name = "tranche_405_y"
    ni.p.saving(fig)

    plt.show()

    with open(ni.p.save_folder + '/zpos.txt', 'w') as f:
        f.write('%f'%focus_pos)
    f.close()

    ipdb.set_trace()

    #OUTPUTS
    """
    sigma_480 = 198 #m2/mol
    sigma_405 = 415 #m2/mol

    I_480 = 1e6*(1/tau_1[1:].mean()-0.014)/sigma_480
    I_405 = 1e6*(1/tau_2.mean() - 0.014 -1/tau_1[1:].mean())/sigma_405   
    """

    ni.end_exp(__file__)
