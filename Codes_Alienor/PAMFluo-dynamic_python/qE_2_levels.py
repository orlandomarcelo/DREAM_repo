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
import tifffile as tiff
from Ingredients import pulse_ingredient, read_pulses
from Segmentation import segmentation, preprocess_movie, segment_movie, trajectories
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

ex = Experiment('qE_2_levels', ingredients=[pulse_ingredient, start_and_stop, segmentation])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "qE_2_levels"

    trigger_color = "no_LED_trig"
    induction = 60
    relaxation = induction
    num_period = 3
    acq_time = (2* induction + 10) * num_period
    sample_rate = 2e4
    limit_blue_high = 450
    limit_blue_low = 3
    gain_induction = 50
    gain_relaxation = 100
    exposure_induction = 400
    exposure_relaxation = 900
    limit_green = 0
    limit_purple = 0
    frame_rate = 1
    n_samples = acq_time*frame_rate
    move_plate = True
    xmove=100
    actinic_filter=1

@ex.automain
def OJIP(_run, _log, name, limit_OJIP, limit_blue_low, limit_blue_high, limit_green, limit_purple, limit_red, trigger_color, sample_rate, 
        acq_time, length_SP, induction, relaxation, exposure_induction, exposure_relaxation, gain_induction, gain_relaxation, n_samples,
        move_plate, xmove, actinic_filter, num_period):

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue_low, limit_green = limit_green, limit_red = limit_red,
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, initial_filter = filters[actinic_filter])

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    output_sequence = {}
    time_sequence = {}
    video_sequence = {}


    add_master_digital_pulse(link, pins['no_LED_trig'], 1000, sec*induction//60, 20, 0) #trigger
    add_digital_pulse(link, pins["blue"], 10000, 20*minute, acq_time*sec, 0)
    thread, cam = init_camera(exposure = exposure_relaxation, num_frames = n_samples, gain =  gain_relaxation)

    # Acquisition
    ni.detector_start()
    thread.start()

    start_measurement(link)

    LED = LED_blue

    for i in range(num_period):
        ctrlLED.set_user_limit(LED, int(limit_blue_low))
        time.sleep(induction)
        cam.set_Gain(gain_induction)
        cam.set_Exposure(exposure_induction)
        time.sleep(2)
        ctrlLED.set_user_limit(LED, int(limit_blue_high))
        time.sleep(induction)
        cam.set_Gain(gain_relaxation)
        cam.set_Exposure(exposure_relaxation)
        time.sleep(4)
    ctrlLED.set_user_limit(LED, int(limit_blue_low))
      

    output, timeframe = ni.read_detector()
    stop_measurement(link)
    thread.stop_thread = True
    thread.join()
    cam.exit()
    try: #color camera vs grey
        L, H = cam.video[0].shape
        video = np.array(cam.video)

    except:
        L, H, d = cam.video[0].shape
        video = np.array(cam.video, dtype = "uint16")[:,:,:, 2]
    
    tiff.imwrite(ni.save_folder + "/video_high.tiff", video, photometric='minisblack')
    timing = cam.timing
    np.save(ni.save_folder + "/video_timing_high.npy", timing)
    ni.end_tasks()

    reset_arduino(link)

    if move_plate:
        print("platform movement")
        motors = all_links[4]
        python_comm.move_dx(motors, xmove)
        #time.sleep(2)
        #python_comm.move_dy(motors, 200)


    link.close()
    ni.end_tasks()
    ni.window = 1000
    average_output, downscaled_time = ni.averaging(output, timeframe)

    for i in range(len(average_output[1])):
        _run.log_scalar("qE decay", average_output[1][i], downscaled_time[i])
        _run.log_scalar("MPPC", average_output[0][i], downscaled_time[i])
        

    ni.p.ylog = "none"

    ni.p.label_list = ['amplitude', 'qE_decay']
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "qE_decay"
    fig = ni.p.plotting(downscaled_time, [average_output[0], average_output[1]])
    ni.p.saving(fig)
    _run.add_artifact((ni.p.save_path + ni.p.extension))

    
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    FO = preprocess_movie(ni.save_folder + "/video_high.tiff", ni.p, show = True)
    ni.g.col_num = 2
    watershed_im_mask, FO = segment_movie(FO, ni.g)
    items_dict = trajectories(watershed_im_mask, FO)

    fluo_MPPC = output[1]
    amplitude_MPPC = output[0]

    axs[0][0].plot(timeframe, fluo_MPPC)
    axs[0][0].set_title("MPPC high, voltage")

    v_high = np.mean(video, axis = (1,2))
    v_time_high = np.array(timing)
    t0 = v_time_high[0]
    v_time_high = v_time_high -t0

    zone = []
    time_zone = []
    video_zone = []


    pulse_pos = np.abs(amplitude_MPPC-np.roll(amplitude_MPPC, 1)) 
    pulse_pos = pulse_pos> 0.5 *np.max(pulse_pos)
    indices = np.linspace(0, len(pulse_pos)-1, len(pulse_pos)).astype(int)
    indices = indices[pulse_pos]
    ratio = len(amplitude_MPPC)/len(v_time_high)
    indices = [0] + list(indices/ratio) + [-1]
    indices_clean = []
    j = 0
    indices_clean.append(indices[0])
    for i in range(len(indices)-1):
        ind = int(indices[i])
        if np.abs(indices_clean[j]-ind) > induction*0.8:
            j+=1
            indices_clean.append(ind)
            
    indices_clean.append(-1)

    print("time zone area HERE", indices_clean, "ratio", ratio)      
    for i in range(len(indices_clean)-1):
        zone.append(v_high[indices_clean[i]+1:indices_clean[i+1]-4])
        time_zone.append(v_time_high[indices_clean[i]+1:indices_clean[i+1]-4] - v_time_high[indices_clean[i]+1])
        video_zone.append(video[indices_clean[i]+1:indices_clean[i+1]-4])
        for k in range(len(zone[i])):
            _run.log_scalar("%d"%i, zone[i][k] , time_zone[i][k])


    ni.p.save_name = "low_light"  
    ni.p.label_list = ["%d"%i for i in range(len(indices_clean))]   
    ni.p.xlabel = "time"
    ni.p.ylabel = "amplitude"
    ni.p.yval = zone[::2]
    ni.p.xval = time_zone[::2]
    fig = ni.p.plotting(ni.p.xval, ni.p.yval)
    ni.p.saving(fig)
        
    ni.p.save_name = "high_light"     
    ni.p.yval = zone[1::2]
    ni.p.xval = time_zone[1::2]
    fig = ni.p.plotting(ni.p.xval, ni.p.yval)
    ni.p.saving(fig)
    #ipdb.set_trace()
    close_all_links(*all_links)