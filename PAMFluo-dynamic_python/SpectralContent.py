    
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
import ipdb

ex = Experiment('spectral_content', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger

@ex.config
def my_config():
    name = "spectral_content"

    acq_time = 60

    gen_ana = [NI_blue]
    detect_input = arduino_blue
    sleep_time = 1
    N = 20


@ex.capture
def FFT(t, y):
    #source: https://stackoverflow.com/questions/56797881/how-to-properly-scale-frequency-axis-in-fast-fourier-transform
    n = len(t)
    Δ = (max(t) - min(t)) / (n-1)
    k = int(n/2)
    f = np.arange(k) / (n*Δ)
    Y = np.abs(np.fft.fft(y))[:k]
    return (f, Y)


@ex.automain
def spectral_content(_run, name, gen_ana, sleep_time, N, detect_input):

    # LED control tool DC4104 Thorlabs

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, 800)
    ctrlLED.set_user_limit(LED_green, 800)
    ctrlLED.set_user_limit(LED_purple, 800)

    ctrlLED.disconnect()

    ni = NIControl.BodeDiagram.BodeDiagram()
    ni.experiment_folder(name)
    ni.p.save_folder = ni.save_folder

    logger.name = name

    # ni diagram cut-off fequency

    ni.generator_analog_channels = [generator_analog[color] for color in gen_ana]
    ni.generator_digital_channels = []
    ni.excitation_frequency = 1
    ni.trig_chan_detector_edge = 'FALLING'

    frequencies = 10**np.linspace(0, 3, N)[::-1]

    ni.excitation_frequency = 1
    ni.num_period = 10
    ni.points_per_period = 2000
    ni.writing_samples = 10000
    ni.update_rates()
    #LED
    ni.offset =  5
    ni.amplitude = 5
    signal = ni.sine_pattern()
    fft_int = []
    fft_fluo = []
    fft_in = []

    fig = plt.figure()
    data = {}
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