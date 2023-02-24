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
    acq_time = 30
    sample_rate = 2e4
    limit_blue = 50
    limit_green = 0
    limit_purple = 0
    exposure = 800
    frame_rate = 1
    n_samples = acq_time*frame_rate

@ex.automain
def OJIP(_run, _log, name, limit_OJIP, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, exposure, n_samples):

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

    filter_sequence = [2,0,2]
    gain_sequence = [100, 100, 100]
    actinic_sequence = [120, 50, 120]
    for i in range(3):
        ctrlLED.set_user_limit(LED_blue, int(actinic_sequence[i]))

        fwl.move_to_filter(filters[filter_sequence[i]])
        add_master_digital_pulse(link, pins['no_LED_trig'], 999, 1*sec, 20, 0) #trigger
        add_digital_pulse(link, pins["blue"], 1000, 10*minute, acq_time*sec, 0)
        thread, cam = init_camera(exposure = exposure, num_frames = n_samples, gain = gain_sequence[i])

        # Acquisition
        ni.detector_start()
        thread.start()

        start_measurement(link)
        output_sequence[i], time_sequence[i] = ni.read_detector()
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
        video_sequence[i] = video
        ni.end_tasks()

        ni.detector_init()
        ni.trigger_init()

        reset_arduino(link)

        tiff.imwrite(ni.save_folder + "/video_%d.tiff"%i, video, photometric='minisblack')


    link.close()
    ni.end_tasks()
    ipdb.set_trace()

    #Results analysis

    for k in range(3):

        ni.window = 1000
        average_output, downscaled_time = ni.averaging(output_sequence[k], time_sequence[k])
        
        """
        ni.p.label_list = detector
        ni.p.xlabel = "time (s)"
        ni.p.ylabel = "voltage (V)"
        ni.p.save_name = "output_plot_%d"%j 
        fig = ni.p.plotting(downscaled_time, [average_output[i] for i in range(average_output.shape[0])])
        ni.p.saving(fig)
        _run.add_artifact((ni.p.save_path + ni.p.extension))

        plt.show()

        ni.p.ylog = "semilogx"
        ni.p.save_name = "ojip_curve_%d"%j
        fig = ni.p.plotting(downscaled_time, average_output[fluorescence])
        ni.p.saving(fig)
        plt.show()
        _run.add_artifact((ni.p.save_path + ni.p.extension))
        """
        fluo = average_output[fluorescence]
        j = 0

        ojip_curve = []
        ojip_time = []
        while j < len(fluo):
            window = 1 + int(np.log(j+ 1))
            data = fluo[j:j+window]
            time = downscaled_time[j:j+window]
            _run.log_scalar("OJIP_response_%d"%k, data.mean(), np.log(time.mean()))
            j = j + window
            ojip_curve.append(data.mean())
            ojip_time.append(time.mean())
        #ipdb.set_trace()

        ni.p.ylog = "semilogx"

        ni.p.label_list = 'ojip_curve'
        ni.p.xlabel = "time (s)"
        ni.p.ylabel = "voltage (V)"
        ni.p.save_name = "ojip_curve_%d"%k
        fig = ni.p.plotting(np.array(ojip_time), np.array(ojip_curve))
        ni.p.saving(fig)
        _run.add_artifact((ni.p.save_path + ni.p.extension))



