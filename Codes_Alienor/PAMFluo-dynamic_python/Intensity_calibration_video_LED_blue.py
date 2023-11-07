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
from ingredient_process_video import process_video, make_maps

import ipdb

@start_and_stop.config
def update_cfg():
    initial_filter = filters[0]

ex = Experiment('D2_calib_video', ingredients=[pulse_ingredient, start_and_stop, process_video])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "D2_calib_video"
    acq_time = 12
    trigger_color = "blue"
    limit_purple = 250
    limit_blue = 400
    exposure = 50
    focus_pos = 0
    gain=100
    cam_period = 200

@ex.automain
def send_IBPC(_run, _log, name, limit_blue, limit_green, limit_red, limit_purple, trigger_color, sample_rate, acq_time, 
        length_SP, length_ML, period_ML, exposure, focus_pos ,gain, cam_period):


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
    #add_digital_pulse(link, pins[trigger_color], 1000, 6*minute//10, 2*minute//10, 0) 

    add_digital_pulse(link, pins["blue"], 1000, 10*minute, 10*minute, 0)
    add_digital_pulse(link, pins["no_LED_trig"], 0, cam_period, 20, 0)

    sample_rate = 1/(cam_period/1000)

    thread, cam = init_camera(exposure = exposure, num_frames = acq_time*sample_rate, gain = gain)


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

    video, timing, L, H  = cam.return_video(ni.save_folder, extend_name = "")

    fname = ni.save_folder + "/video.tiff"
    tiff.imwrite(fname, video[:,:,:],photometric="minisblack")
            
    fname_t = ni.save_folder + '/video_timing.csv'
    pd.DataFrame(timing).to_csv(fname_t)

    _run.add_artifact(fname, "video.tiff")
    _run.add_artifact(fname, "video_timing.csv")

    for i, frame in enumerate(video):
        _run.log_scalar("Fluorescence", np.mean(frame), i)
        _run.log_scalar("Time", i*cam_period/1000, i)

    make_maps(_run, ni.save_folder, video, timing)

    with open(ni.p.save_folder + '/zpos.txt', 'w') as f:
        f.write('%f'%focus_pos)
    f.close()

    ni.end_exp(__file__)
