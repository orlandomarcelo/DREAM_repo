    
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

from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera


import ipdb

ex = Experiment('cross_validation', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger

ex.add_config("config.json")

@ex.config
def my_config():
    name = "cross_validation"

    acq_time = 60

    gen_ana = [NI_blue]
    detect_input = arduino_blue
    limit_blue = 1000
    limit_green = 0
    limit_red = 0
    limit_purple = 0
    sleep_time = 1
    trigger_color = "blue"
    sample_rate = 1/4 #Hz
    acquisition_time = 30 #s
    actinic_filter = 1
    
    exposure = 60
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



    ni.writing_rate =  1000 #Hz
    ni.acq_time = 62 #s
    ni.writing_samples = int(ni.acq_time*ni.writing_rate)

    ni.sample_rate = 1000
    ni.reading_samples = ni.sample_rate*ni.acq_time
        
        
    #LED
    offset =  1 #V
    sat = 3.4 #V
    duration_sat = 0.2 #ms 
    start = 5
    stop = 15

    def fitness_sequence(ni, offset, sat, start, stop, duration_sat):
       

        sample = np.linspace(0, ni.acq_time, num = ni.writing_samples , endpoint=False)
        actinic = np.zeros(sample.shape)
        actinic[start*ni.writing_rate:stop*ni.writing_rate] = offset
        actinic[(start+30)*ni.writing_rate:(stop+30)*ni.writing_rate] = offset
        pulse = np.zeros(sample.shape)
        pulse[stop*ni.writing_rate:stop*ni.writing_rate + int(duration_sat*ni.writing_rate)] = sat
        pulse[(stop+30)*ni.writing_rate:(stop+30)*ni.writing_rate + int(duration_sat*ni.writing_rate)] = sat

        signal = pulse + actinic

        return sample, signal


    fig = plt.figure()
    
    all_outputs = []
    all_times = [] 
    offsets = np.array([0, 6.63106037, 10.63690732, 13.84158488, 21.85327878,
       29.06380328, 49.09303803, 70.72461155, 83.54332179, 97.16320142]
                       + [0, 6.63106037, 10.63690732, 13.84158488, 21.85327878,
       29.06380328, 49.09303803, 70.72461155, 83.54332179][::-1])/100
    
    for i in range(len(offsets)):
        thread, cam = init_camera(exposure = exposure, num_frames = 15*ni.acq_time, gain =  gain, trigger=False, trigger_frequency=15)
        thread.start()
        
        sample, signal = fitness_sequence(ni, offsets[i], sat, start, stop, duration_sat)
        #plt.plot(sample, signal)
        #plt.show()
        
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
        time.sleep(60)
  
        np.save(ni.p.save_folder + "/outputs_%03d_%03d.npy"%(100*offsets[i], i), output[1:3])
        np.save(ni.p.save_folder + "/time_%03d_%03d.npy"%(100*offsets[i], i), time_array)

        cam.return_video(ni.p.save_folder, extend_name = "%03d_%03d_"%(100*offsets[i], i))
        
    close_all_links(*all_links)
    
    ipdb.set_trace()


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