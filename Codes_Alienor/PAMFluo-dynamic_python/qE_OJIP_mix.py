import pandas as pd
from alienlab.init_logger import logger
from tqdm import tqdm
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg
import datetime
import tifffile as tiff

from ArduinoControl import python_comm

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera
import ipdb

ex = Experiment('qE_calib', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "qE_calib"
    camera_heat_delay= 250 #s
    HL_time = 0.5
    periods=1
    acq_time = periods*2*HL_time*60 + 2# + camera_heat_delay
    actinic_filter = 1
    move_plate = False
    length_SP = 200
    period_SP= 20*sec
    limit_purple = 150
    limit_blue_high = 300
    limit_blue_low = 0
    trigger_color = "no_LED_trig"
    sample_rate=1e3
    exposure = 170
    gain  = 70


@ex.automain
def send_IBPC(_run, name, camera_heat_delay, limit_blue_high, HL_time, gain, exposure, periods, limit_blue_low, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, period_SP, length_SP, period_ML, actinic_filter, move_plate):


    # LED control tool DC4104 Thorlabs
    logger.name = name


    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue_high, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo=True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])

    ratio = 5
    add_digital_pulse(link_arduino, pins['no_LED_trig'], camera_heat_delay*sec - 2*sec, sec/ratio, 20, 1) #trigger

    add_digital_pulse(link_arduino, pins['blue'], camera_heat_delay*sec + period_SP//2, 1000000, 1000000, 1)

    add_master_digital_pulse(link_arduino, pins['purple'], camera_heat_delay*sec, period_SP, length_SP, 0)


    thread, cam = init_camera(exposure = exposure, num_frames = acq_time*ratio, gain =  gain)


    # Acquisition

    ni.detector_start()
    thread.start()

    LED = LED_blue
    start_measurement(link_arduino)
    time.sleep(1 + period_SP//(2*1000))
    for i in range(periods):
        time.sleep(HL_time*60 +camera_heat_delay)
        ctrlLED.set_user_limit(LED, int(limit_blue_low))
        #time.sleep(HL_time*60)
        #ctrlLED.set_user_limit(LED, int(limit_blue_high))

    output, time_array = ni.read_detector()

    stop_measurement(link_arduino)
    thread.stop_thread = True
    thread.join()
    cam.exit()
    try: #color camera vs grey
        L, H = cam.video[0].shape
        video = np.array(cam.video)

    except:
        L, H, d = cam.video[0].shape
        video = np.array(cam.video, dtype = "uint16")[:,:,:, 2]
    
    tiff.imwrite(ni.save_folder + "/video.tiff", video, photometric='minisblack')
    timing = cam.timing
    np.save(ni.save_folder + "/video_timing.npy", timing)

    #ipdb.set_trace()

    fluo = output[1]
    arduino_purple_array = output[4]
    arduino_blue_array = output[2]
    indices = (arduino_purple_array<1)*(arduino_blue_array>0.5)
    direct_fluo = fluo[indices]
    time_fluo = time_array[indices]

    fig = plt.figure()
    ni.p.xlabel = "time (s)"
    ni.p.ylabel = "voltage (V)"
    ni.p.save_name = "fluorescence_selectively" 
    fig = ni.p.plotting(mvgavg(time_fluo, 100, binning=True), mvgavg(direct_fluo, 100, binning=True))
    ni.p.saving(fig)
    _run.add_artifact(ni.p.save_path + ni.p.extension)
    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_purple, arduino_amplitude = arduino_blue) 
    
    result[2][0] = result[2][1]
    blank_level = np.mean(result[2][0:15])
    logger.critical(blank_level)


    np.savetxt(save_figure_folder + "/id_%05d_%s_measure_pulse.csv"%(_run._id, name), np.array(result[0]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_fluorescence.csv"%(_run._id, name), np.array(result[2]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_MPPC_intensity.csv"%(_run._id, name), np.array(result[4]), delimiter=",")


    for i in range(len(result[0])):
        _run.log_scalar("Measure pulse", result[0][i] - blank_level, i*period_ML/1000)
        _run.log_scalar("Fluorescence", result[2][i], i*period_ML/1000)
        _run.log_scalar("MPPC", result[4][i], i*period_ML/1000)

    #ipdb.set_trace()
    
    FM = np.array(result[0][0]- blank_level).mean()

    actinic_level = np.mean(result[4][40:56])
    plt.close("all")
    #
    #ipdb.set_trace()

    if move_plate:
        print("platform movement")
        motors = all_links[4]
        python_comm.move_dx(motors, 200)
        #time.sleep(2)
        #python_comm.move_dy(motors, 200)

        for i in range(len(result[0])):
            _run.log_scalar("norm Measure pulse", (result[0][i] - blank_level)/FM, i*period_ML/1000)
            _run.log_scalar("norm Fluorescence", result[2][i]/FM, i*period_ML/1000)

    close_all_links(*all_links)
    
    ipdb.set_trace()

    #plt.plot(output[1]);plt.plot(output[3]);plt.show()
    """
        from tkinter.filedialog import askopenfilename, askopenfilenames
        file = askopenfilename()

        import pandas as pd

        f1 = pd.read_csv(file)


        file = askopenfilename()
        f2 = pd.read_csv(file)

        name = "voltage pulse_mean"

        import matplotlib.pyplot as plt

        import numpy as np
        u = np.asarray(f1[name])
        v = np.asarray(f2[name])

        plt.plot(u-v)
        plt.show()

    """