    
import NIControl.BodeDiagram
import numpy as np
import pandas as pd
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

from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera


import ipdb

ex = Experiment('Oscillation_Protocol', ingredients=[pulse_ingredient, start_and_stop])

#ex.observers.append(MongoObserver())

ex.logger = logger

ex.add_config("config.json")

@ex.config
def my_config():
    name = "Oscillation_Protocol"

    acq_time = 20 #s

    gen_ana = [NI_blue]
    detect_input = arduino_blue
    limit_blue = 1000
    limit_green = 0
    limit_red = 0
    limit_purple = 0
    sleep_time = 1
    trigger_color = "blue"
    sample_rate = 1/4 #Hz
    actinic_filter = 1
    
    exposure = 60 #ms sets the exposure time of the camera
    gain = 100
    
    N = 10


@ex.automain
def spectral_content(_run, name, gen_ana, sleep_time, N, detect_input, limit_blue, limit_red, limit_green, limit_purple, trigger_color, acq_time, sample_rate, actinic_filter, exposure, gain):
    
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

    # ni diagram fequency

    ni.generator_analog_channels = [generator_analog[color] for color in gen_ana]
    ni.generator_digital_channels = []
    ni.excitation_frequency = 1
    ni.trig_chan_detector_edge = 'FALLING'

    frequencies = [0.5, 1/16, 1/64]
    acq_per_period = [10, 25, 50]
    acq_time = 3600

    ##### Quick test #####
    #frequencies = [0.5]
    #acq_per_period = [10]
    #acq_time = 20

    ni.excitation_frequency = frequencies[0]
    ni.num_period = int(frequencies[0]*acq_time)
    ni.points_per_period = 2000
    ni.writing_samples = 10000
    ni.update_rates()
        
        
    #LED
    
    ni.offset =  0.455
    ni.amplitude = 0.2275
    signal = ni.sine_pattern()
    all_outputs = []
    all_times = [] 

    fig = plt.figure()
    data = {}
    
    
    
    for i, freq in enumerate(frequencies):
        frame_rate = acq_per_period[i]*freq
        thread, cam = init_camera(exposure = exposure, num_frames = ni.acq_time*frame_rate, gain =  gain, trigger=False, trigger_frequency=frame_rate)
        thread.start()
        
        ni.excitation_frequency = freq
        ni.num_period = int(freq*acq_time)
        ni.update_rates() 
        ni.generator_analog_init(signal)
        ni.detector_init()
        #self.trigger_frequency = freq 
        ni.trigger_init()

        ni.task_generator_analog.start()
        ni.detector_start()
        ni.task_trigger.start()
        
        output, time_array = ni.read_detector()

        all_outputs.append(output)
        all_times.append(time_array)
        ni.end_tasks()    
        ni.shut_down_LED(1)        
        time.sleep(5)

        
        names = ['Intensity', 'Fluo', 'Blue','Purple','Green', 'Trigger', 'jspq']
        
        dico = {'time':time_array}   
            
        for j, k  in enumerate(names):
            dico[k] = output[j]
            
        file_name = ni.p.save_folder + "/outputs_%03d_%03d.csv"%(frequencies[i], i)
            
        pd.DataFrame(dico).to_csv(file_name, sep = ',', index=False)
        _run.add_artifact(file_name)

        cam.return_video(ni.p.save_folder, extend_name = "%03d_%03d_"%(frequencies[i], i))
        
        file_name = ni.p.save_folder + "/%03d_%03d_video.tiff"%(frequencies[i], i)
        _run.add_artifact(file_name)

        cam.exit()
    close_all_links(*all_links)
    
    #ipdb.set_trace()


    """
    data = {}
    
    offsets = 10**np.linspace(0, 3, N)[::-1]

    for i, freq in enumerate(frequencies):
        
        ni.excitation_frequency = freq
        ni.update_rates() 
        ni.generator_analog_init(signal)
        ni.detector_init()
        #self.trigger_frequency = freq
        ni.trigger_init()

        ni.task_generator_analog.start()
        ni.detector_start()
        ni.task_trigger.start()


        output, time_array = ni.read_detector()

        (f, Y) = FFT(time_array, output[1])
        fft_fluo.append(Y)
        (f, Y) = FFT(time_array, output[0])
        fft_int.append(Y)
        (f_in, Y_in) = FFT(time_array, output[3])
        fft_in.append(Y_in)


        Y[0] = 0
        Y_in[0]=0
        data[freq]={}
        data[freq]["intensity"] = Y
        data[freq]["input"] = Y_in
        data[freq]["output"] = output
        data[freq]["time"] = time_array
        plt.plot(Y[:100], label = freq)
        ni.end_tasks()
        ni.set_level_LED(signal.mean(), None, sleep_time)
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            _run.log_scalar("Harmonics intensity %.0f Hz"%freq, Y[10*i]/Y[10], i) #10 corresponds to the sampling 2e6, should be changed if the sampling rate evolves. 
            _run.log_scalar("Harmonics input %.0f Hz"%freq, Y_in[10*i]/Y_in[10], i) #10 corresponds to the sampling 2e6, should be changed if the sampling rate evolves. 

    plt.legend()
    ni.p.save_name = 'spectral_content'
    ni.p.saving(fig)
    _run.add_artifact(ni.p.save_path +  ni.p.extension)
    ipdb.set_trace()
    
    np.save(ni.save_folder + "/data_dict.npy", data)
    

    ni.end_exp(__file__)
"""