from alienlab.init_logger import logger
import time
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg

from PyQt5 import QtGui

import tifffile as tiff

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera
from Segmentation import segmentation, preprocess_movie, segment_movie, trajectories
import ipdb
from VoltageIntensityClass import VoltageIntensity


ex = Experiment('Ek_video', ingredients=[pulse_ingredient, start_and_stop, segmentation])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")



@ex.config
def my_config():
    name = "Ek_video"
    trigger_color = "green"

    sample_rate = 1e5
    gain = 10

    N = 36
    n_samples = 6
    frame_rate = 1

    length_SP = 10 #s
    period_SP = 30 #s
    delay_trig = 1 #ms
    filter_list = [3,2, 1]

    limit_blue = 0

    acq_time = 11#(n_samples * period_SP + 3) * (N//2)*2 + len(filter_list)*15

    exposure = 800




@ex.automain
def SP_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, frame_rate, 
                    acq_time, gain, length_SP, length_ML, period_ML, N, period_SP, delay_trig, n_samples, filter_list,
                    exposure):

    voltint = VoltageIntensity()
  
    all_links = initialize(logger = logger, name = name, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]



    add_digital_pulse(link, pins['blue'], 0, period_SP*sec, length_SP*sec, 0)
    add_master_digital_pulse(link, pins['no_LED_trig'], 0+delay_trig, 0.9*sec, 20, 0) #trigger
    add_master_digital_pulse(link, pins['green'], 0, 150000, length_ML, 0)


    ctrlLED.set_user_limit(LED_blue, limit_blue)

    #limits_blue_low = np.linspace(10, 45, N//(2*len(filter_list)))
   # limits_blue_high = np.linspace(50, 250, N//(2*len(filter_list)))
    #limits_blue = np.concatenate((limits_blue_low, limits_blue_high), axis = None)
    limits_blue = np.array([0, 20, 50, 80, 100, 150, 200, 250, 300, 350])
    
    value_intensity = []
    value_MPPC = []

    data = {}
    start = time.time()
    # a thread that waits for new images and processes all connected views
    for filter in filter_list:
        data[filter] = {}
        fwl.move_to_filter(filters[filter])
        for limit_blue in limits_blue:
            data[filter][limit_blue] = {}
            ctrlLED.set_user_limit(LED_blue, int(limit_blue))
            try: 
                print(ctrlLED.get_user_limit(LED_blue))
            except:
                pass
            thread, cam = init_camera(exposure = exposure, num_frames = N*n_samples)

            ni.detector_start()
            thread.start()

            start_measurement(link)
            time.sleep(length_SP-0.1)
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

            output, time_array = ni.read_detector()
            ni.detector_stop()
            tiff.imwrite(ni.save_folder + "/video_%d_%0.2f.tiff"%(filter, limit_blue), video, photometric='minisblack')

            data[filter][limit_blue]["output"] = output
            data[filter][limit_blue]["video"] = video

            time.sleep(60)
            add_digital_pulse(link, pins['blue'], 0, period_SP*sec, length_SP*sec, 0)
            add_master_digital_pulse(link, pins['no_LED_trig'], 0+delay_trig, 0.9*sec, 20, 0) #trigger
            add_master_digital_pulse(link, pins['green'], 0, 150000, 20, 0)




    stop = time.time()

    logger.info("recording time: %0.2s"%(stop-start))


    close_all_links(*all_links)
    np.save(ni.save_folder + "/dict_full.npy",data)
    ipdb.set_trace()

    """"

    amplitude_diff = 1
    step_delay = 1
    a = output[arduino_blue]
    b = np.roll(output[arduino_blue], amplitude_diff)
    cut = a-b

    indices = np.linspace(0, len(cut)-1, len(cut)).astype(int)
    indices = indices[cut>=1]

    start = indices[0]
    mean_intense = []
    blank_intense = []
    mean_fluo = []
    blank_fluo = []
    for stop in indices[1:]:
        start = start + step_delay
        intensity_pulse = output[arduino_blue, start:stop]
        intensity_pulse = intensity_pulse > 0
        blank = intensity_pulse == 0

        intense = output[intensity, start:stop]
        intense = intense[intensity_pulse]
        fluo = output[fluorescence, start:stop]
        fluo = fluo[intensity_pulse]
        plt.plot(intense)
        mean_intense.append(np.mean(intense))
        mean_fluo.append(np.mean(fluo))
        
        intense_blank = output[intensity, start:stop]
        intense_blank = intense_blank[blank]
        blank_intense.append(np.mean(intense_blank))
        fluo_blank = output[fluorescence, start:stop]
        fluo_blank = fluo_blank[blank]
        blank_fluo.append(np.mean(blank_fluo))
        start = stop

    mean_fluo_video = np.mean(cam.video, axis = (1,2))
    mean_intense = mean_intense# - np.mean(intense_blank)
    mean_fluo = mean_fluo# - np.mean(fluo_blank)

    np.savetxt(ni.save_folder + "/mean_intense.csv", mean_intense, delimiter=",")
    np.savetxt(ni.save_folder + "/mean_fluo.csv", mean_fluo, delimiter=",")
    np.savetxt(ni.save_folder + "/mean_fluo_video.csv", mean_fluo_video, delimiter=",")
    np.savetxt(ni.save_folder + "/value_intensity.csv", value_intensity, delimiter = ",")
    np.savetxt(ni.save_folder + "/value_MPPC.csv", value_MPPC, delimiter = ",")

    for i in range(1,min(len(mean_fluo_video), len(mean_fluo))):
        _run.log_scalar("mean_intensity", mean_intense[i], i)
        _run.log_scalar("mean_fluo", mean_fluo[i], i)
        _run.log_scalar("mean_fluo_video", mean_fluo_video[i], i)
        _run.log_scalar("MPPC_fluo_curve", mean_fluo[i], mean_intense[i])
        _run.log_scalar("video_fluo_curve", mean_fluo_video[i], mean_intense[i])

    ipdb.set_trace()

    FO = preprocess_movie(ni.save_folder + "/video.tiff", ni.p)

    watershed_im_mask, FO = segment_movie(FO, ni.g)

    items_dict = trajectories(watershed_im_mask, FO)


    MPPC_voltage = []
    intensity_voltage = []
    intensity_MPPC = []
    i = 0
    for f in filter_list:
        for limit_blue in limits_blue:
            MPPC_voltage.extend([voltint.get_MPPC_voltage('blue', f, limit_blue/100)]*n_samples)
            intensity_voltage.extend([voltint.get_intensity_voltage('blue', f, limit_blue/100)]*n_samples)
            intensity_MPPC.extend([voltint.get_intensity_MPPC('blue', f, mean_intense[i])]*n_samples)
            i += 1
    figure_names =  ["mean_intense", "MPPC_voltage", "intensity_voltage", "intensity_MPPC"]
    for i, x in enumerate([mean_intense, MPPC_voltage, intensity_voltage, intensity_MPPC]):
        ni.p.label_list = []
        xval = []
        yval = []
        for k in items_dict:
            if k!= 0:
                curve = items_dict[str(k)]['mean']

                xval.append(x)
                yval.append(curve/curve.max())
                ni.p.label_list.append(k)
        ni.p.ylog = 'loglog'
        fig = ni.p.plotting(xval, yval)
        ni.p.save_name="curve_%s"%figure_names[i]
        ni.p.saving(fig)

    
    ipdb.set_trace()
    print("the end")
    """