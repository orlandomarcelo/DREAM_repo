import pandas as pd
from alienlab.init_logger import logger
from tqdm import tqdm
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg
import datetime

from ArduinoControl import python_comm

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('IBPC_RLC', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "IBPC_RLC"
    acq_time = 60*6+10
    actinic_filter = 2
    move_plate = False
    limit_blue = 0

@ex.automain
def send_IBPC(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, actinic_filter, move_plate):


    # LED control tool DC4104 Thorlabs
    logger.name = name


    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])


    add_digital_pulse(link_arduino, pins['purple'], 5000, 60000, length_SP, 1)
    add_digital_pulse(link_arduino, pins['blue'], 0, 2000000, 2000000, 1)
   
    add_master_digital_pulse(link_arduino, pins['green'], length_SP, period_ML, length_ML, 0)


    # Acquisition

    ni.detector_start()
    start_measurement(link_arduino)
    time.sleep(5)
    for limit_blue in [0,100, 200, 300, 400, 500]:
        ctrlLED.set_user_limit(LED_blue, int(limit_blue))
        time.sleep(59.75)
    output, time_array = ni.read_detector()
    stop_measurement(link_arduino)


    ipdb.set_trace()

    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_green, arduino_amplitude = arduino_blue) 
    

    blank_level = np.mean(result[2][30:100])
    logger.critical(blank_level)


    np.savetxt(save_figure_folder + "/id_%05d_%s_measure_pulse.csv"%(_run._id, name), np.array(result[0]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_fluorescence.csv"%(_run._id, name), np.array(result[2]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_MPPC_intensity.csv"%(_run._id, name), np.array(result[4]), delimiter=",")


    for i in range(len(result[0])):
        _run.log_scalar("Measure pulse", result[0][i] - blank_level, i*period_ML/1000)
        _run.log_scalar("Fluorescence", result[2][i], i*period_ML/1000)
        _run.log_scalar("MPPC", result[4][i], i*period_ML/1000)

    #ipdb.set_trace()
    
    F0 = np.array(result[0][0:15]- blank_level).mean()
    FM = np.array(result[0][58:63] - blank_level).max()
    FS = np.array(result[0][40:56]- blank_level).mean()
    F_direct = np.array(result[2][40:56]- blank_level).mean()
    print(result[4][40:56])
    print(result[4][0:15])

    actinic_level = np.mean(result[4][40:56])
    plt.close("all")
    #ipdb.set_trace()

    if move_plate:
        print("platform movement")
        motors = all_links[4]
        python_comm.move_dx(motors, 200)
        #time.sleep(2)
        #python_comm.move_dy(motors, 200)

        for i in range(len(result[0])):
            _run.log_scalar("norm Measure pulse", (result[0][i] - blank_level)/F0, i*period_ML/1000)
            _run.log_scalar("norm Fluorescence", result[2][i]/F0, i*period_ML/1000)

    close_all_links(*all_links)
    return float(actinic_level), float((FM-F0)/FM), float(F0), float(FS), float(FM), float(F_direct)

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