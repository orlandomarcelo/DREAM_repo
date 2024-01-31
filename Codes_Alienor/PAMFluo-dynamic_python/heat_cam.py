    
import NIControl.BodeDiagram
import numpy as np
import alienlab
import matplotlib.pyplot as plt
import time
from alienlab.init_logger import logger
from scipy import optimize
import alienlab.regression_func
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from config_DAQ import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from ingredient_process_video import process_video, make_maps

from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera

import pandas as pd


import ipdb

ex = Experiment('Activation_curve', ingredients=[pulse_ingredient, start_and_stop, process_video])

ex.observers.append(MongoObserver())

ex.logger = logger

ex.add_config("config.json")

@ex.config
def my_config():
    name = "cam_heating"

    acq_time = 60
    gen_ana = [NI_blue]
    detect_input = arduino_blue
    limit_blue = 1000
    limit_green = 0
    limit_red = 0
    limit_purple = 0
    sleep_time = 5 #s
    trigger_color = "blue"
    sample_rate = 1/4 #Hz
    actinic_filter = 1
    exposure = 180 #ms sets the exposure time of the camera and also actual framerate
    gain = 100
    N = 10


@ex.automain
def activation_curve(_run, name, gen_ana, sleep_time, N, detect_input, limit_blue, limit_red, limit_green, 
                     limit_purple, trigger_color, acq_time, sample_rate, actinic_filter, exposure, gain):
    
    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo=False)

    ctrlLED = all_links[0]
    ni0 = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]
    
    fwl.move_to_filter(filters[actinic_filter]) 
    


    ni = NIControl.BodeDiagram.BodeDiagram() #NI card control tool
    ni.experiment_folder(name) #folder name
    ni.p.save_folder = ni.save_folder #where to save images

    logger.name = name

    ni.generator_analog_channels = [generator_analog[color] for color in gen_ana] #LED control channel
    ni.generator_digital_channels = []
    ni.trig_chan_detector_edge = 'FALLING' #trigger RISING or FALLING

    ni.writing_rate =  1000 #Hz
    ni.acq_time = 300 #s
    ni.writing_samples = int(ni.acq_time*ni.writing_rate)

    ni.sample_rate = 1000
    ni.reading_samples = ni.sample_rate*ni.acq_time
        
    #LED
    sat = 3.4 #V
    duration_sat = 0.5 #s 
    start = 5
    stop = 15

    def fitness_sequence(ni, offset, sat, start, stop, duration_sat):
    
        sample = np.linspace(0, ni.acq_time, num = ni.writing_samples , endpoint=False)
        actinic = np.zeros(sample.shape)
        actinic[start*ni.writing_rate:stop*ni.writing_rate] = offset
        pulse = np.zeros(sample.shape)
        pulse[stop*ni.writing_rate:stop*ni.writing_rate + int(duration_sat*ni.writing_rate)] = sat

        signal = pulse + actinic

        return sample, signal


    fig = plt.figure()
    
    all_outputs = []
    all_times = [] 
    offsets = [0]
    
    for i in range(len(offsets)):
        thread, cam = init_camera(exposure = exposure, num_frames = 15*ni.acq_time, gain =  gain, trigger=False, trigger_frequency=1000/exposure)
        thread.start()
        
        sample, signal = fitness_sequence(ni, offsets[i], sat, start, stop, duration_sat)

        ni.generator_analog_init(signal)
        ni.detector_init()
        #self.trigger_frequency = freq
        ni.trigger_init()

        ni.task_generator_analog.start()
        ni.detector_start()
        ni.task_trigger.start()

        output, time_array = ni.read_detector()
        thread.stop_thread = True
        thread.join()
        cam.exit()
        all_outputs.append(output[1])
        all_times.append(time_array)
        ni.end_tasks()    
        ni.shut_down_LED(1)         

        names = ['Intensity', 'Fluo', 'Blue','Purple','Green', 'Trigger', 'jspq']
        
        data_dict = {'time':time_array}   
            
        for j, k  in enumerate(names):
            data_dict[k] = output[j]
            
        file_name = ni.p.save_folder + f"/outputs_{offsets[i]:.2f}V.csv"
            
        pd.DataFrame(data_dict).to_csv(file_name, sep = ',', index=False)
  
        cam.return_video(ni.p.save_folder, extend_name = f"_{offsets[i]:.2f}V")
        #make_maps(_run, f"{ni.save_folder}_{offsets[i]:.2f}V", np.array(cam.video[0:210]), np.array(cam.timing[0:210]), smoothing = 5, limits = (0,0), filter = "none")
        
        time.sleep(sleep_time)


    close_all_links(*all_links)
    