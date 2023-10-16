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

ex = Experiment('qE_OJIP', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "qE_OJIP"

    trigger_color = "no_LED_trig"
    acq_time = 150
    sample_rate = 2e4
    limit_blue = 500
    gain_blue = 10
    limit_pulses = 500
    gain_pulses = 10
    limit_green = 0
    limit_purple = 0
    exposure = 50
    period_ML = 0.3*minute
    length_ML = 100
    minutes_relaxation = 10
    frame_rate = 1
    n_samples = acq_time*frame_rate
    move_plate = True
    xmove=100

@ex.automain
def OJIP(_run, _log, name, limit_OJIP, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, 
        acq_time, length_SP, length_ML, period_ML, exposure, n_samples, limit_pulses,gain_blue, gain_pulses,
        minutes_relaxation, move_plate, xmove):

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]

    output_sequence = {}
    time_sequence = {}
    video_sequence = {}


    ctrlLED.set_user_limit(LED_blue, int(limit_blue))
    fwl.move_to_filter(filters[1])
    add_master_digital_pulse(link, pins['no_LED_trig'], 1000, 1*sec, 20, 0) #trigger
    add_digital_pulse(link, pins["blue"], 1000, 10*minute, acq_time*sec, 0)
    thread, cam = init_camera(exposure = exposure, num_frames = n_samples, gain =  gain_blue)

    # Acquisition
    ni.detector_start()
    thread.start()

    start_measurement(link)
    output_sequence['high'], time_sequence['high'] = ni.read_detector()
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
    video_sequence["high"] = video
    tiff.imwrite(ni.save_folder + "/video_high.tiff", video, photometric='minisblack')
    timing = cam.timing
    np.save(ni.save_folder + "/video_timing_high.npy", timing)
    ni.end_tasks()
    ctrlLED.set_user_limit(LED_blue, int(limit_pulses))

    ni.acq_time = 60*minutes_relaxation #s
    ni.reading_samples = int(ni.sample_rate * ni.acq_time)

    ni.detector_init()
    ni.trigger_init()

    reset_arduino(link)

    
    add_master_digital_pulse(link, pins['no_LED_trig'], 10, period_ML, 20, 0) #trigger
    add_digital_pulse(link, pins["blue"], 5, period_ML, length_ML, 0)
    thread, cam = init_camera(exposure = exposure, num_frames = n_samples, gain = gain_pulses)

    # Acquisition
    ni.detector_start()
    thread.start()

    start_measurement(link)
    output_sequence['pulses'], time_sequence['pulses'] = ni.read_detector()
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
    video_sequence["pulses"] = video
    timing = cam.timing
    ni.end_tasks()


    reset_arduino(link)

    
    np.save(ni.save_folder + "/video_timing_pulses.npy", timing)
    tiff.imwrite(ni.save_folder + "/video_recovery.tiff", video, photometric='minisblack')

    if move_plate:
        print("platform movement")
        motors = all_links[4]
        python_comm.move_dx(motors, xmove)
        #time.sleep(2)
        #python_comm.move_dy(motors, 200)


    link.close()
    ni.end_tasks()

    #Results analysis
    ni.window = 1000
    average_output, downscaled_time = ni.averaging(output_sequence["high"], time_sequence['high'])
    fluo = average_output[fluorescence]
    j = 0

    ojip_curve = []
    ojip_time = []
    while j < len(fluo):
        window = 1 + int(np.log(j+ 1))
        data = fluo[j:j+window]
        time = downscaled_time[j:j+window]
        _run.log_scalar("OJIP_response", data.mean(), np.log(time.mean()))
        j = j + window
        ojip_curve.append(data.mean())
        ojip_time.append(time.mean())

    ni.p.ylog = "semilogx"

    ni.p.label_list = 'ojip_curve'
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "ojip_curve"
    fig = ni.p.plotting(np.array(ojip_time), np.array(ojip_curve))
    ni.p.saving(fig)
    _run.add_artifact((ni.p.save_path + ni.p.extension))

    for i in range(len(average_output[1])):
        _run.log_scalar("qE decay", average_output[1][i], downscaled_time[i])
    
    #ipdb.set_trace()

    ni.p.ylog = "none"

    ni.p.label_list = 'qE_decay'
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "qE_decay"
    fig = ni.p.plotting(downscaled_time, average_output[1])
    ni.p.saving(fig)
    _run.add_artifact((ni.p.save_path + ni.p.extension))
    #ipdb.set_trace()
    result = read_pulses(ni = ni, output = output_sequence["pulses"], time_array = time_sequence["pulses"], step_delay = 10, arduino_color = arduino_blue, arduino_amplitude = arduino_blue) 

    np.savetxt(ni.p.save_folder + "/measure_pulse.csv", np.array(result[0]), delimiter = ",")
    t0 = downscaled_time.max()
    np.savetxt(ni.p.save_folder + "/time_pulse.csv", np.linspace(t0, t0 + len(result[0])*period_ML/1000, len(result[0])))


    np.savetxt(save_figure_folder + "/id_%05d_%s_measure_pulse.csv"%(_run._id, name), np.array(result[0]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_fluorescence.csv"%(_run._id, name), np.array(result[2]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_MPPC_intensity.csv"%(_run._id, name), np.array(result[4]), delimiter=",")

    t0 = downscaled_time.max()
    for i in range(len(result[0])):
        _run.log_scalar("Measure pulse", result[0][i], i*period_ML/1000 + t0)
        _run.log_scalar("Fluorescence", result[2][i], i*period_ML/1000 + t0)
        _run.log_scalar("MPPC", result[4][i], i*period_ML/1000 + t0)


    ipdb.set_trace()

    close_all_links(*all_links)