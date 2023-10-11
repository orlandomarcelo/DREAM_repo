    
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
from ArduinoControl.python_serial import *
from serial import *


from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
import ipdb

ex = Experiment('Bode_IBPC', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger

ex.add_config("config.json")


@ex.config
def my_config():
    name = "Bode_IBPC"

    trigger_color = "blue"
    acq_time = 60
    sample_rate = 2e6

    gen_ana = [NI_blue]
    sleep_time = 2
    N = 5
    min_freq = 0
    max_freq = 3
    


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
def spectral_content(_run, _log, name, gen_ana, sleep_time, N, length_ML, length_SP, period_ML, min_freq, max_freq, limit_blue, limit_green, limit_purple):

    # LED control tool DC4104 Thorlabs

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, limit_blue)
    ctrlLED.set_user_limit(LED_green, limit_green)
    ctrlLED.set_user_limit(LED_purple, limit_purple)


    ni = NIControl.BodeDiagram.BodeDiagram()
    ni.experiment_folder(name)
    ni.cop.save_folder = ni.save_folder

    logger.name = name

    # ni diagram cut-off fequency

    ni.generator_analog_channels = [generator_analog[color] for color in gen_ana]
    ni.generator_digital_channels = []
    ni.detector_analog_channels = [detector_analog[intensity], detector_analog[fluorescence]]
    ni.detector_digital_channels = [detector_digital[i] for i in [arduino_blue, arduino_green, arduino_purple]]
    ni.excitation_frequency = 1
    ni.trig_chan_detector_edge = 'FALLING'

    frequencies = 10**np.linspace(min_freq, max_freq, N)#[::-1]

    ni.excitation_frequency = 1
    ni.num_period = 10
    ni.points_per_period = 2000
    ni.update_rates()
    
    #LED
    ni.offset = 0.15
    ni.amplitude = 0.1
    signal = ni.sine_pattern()
    fft_int = []
    fft_fluo = []
    link = Serial("COM4", 115200)
    time.sleep(2.0)

    reset_arduino(link)


    for i, freq in enumerate(frequencies):
        

        add_digital_pulse(link, pins['purple'], 500, 30000, length_SP, 1)
        #add_digital_pulse(link, pins['blue'], 1000 + length_ML, 30000, 30000, 1) #actinic
        add_master_digital_pulse(link, pins['green'], length_SP, period_ML, length_ML, 0)

        ni.excitation_frequency = freq
        ni.update_rates() 
        ni.acq_time += 4
        ni.reading_samples += int(ni.sample_rate*4)
        ni.generator_analog_init(signal)
        ni.detector_init()
        #self.trigger_frequency = freq
        ni.trigger_init()

        ni.task_generator_analog.start()

        _log.critical("Frequency excitation")

        ni.detector_start()
        ni.task_trigger.start()

        time.sleep(ni.num_period/ni.excitation_frequency)

        ni.task_generator_analog.stop()
        ni.task_generator_analog.close()
        ni.task_trigger.stop()
        ni.task_trigger.close()
        ni.set_level_LED(0*signal.mean(), None, 0.001)
        
        _log.critical("Quantum yield")

        start_measurement(link)
        time.sleep(5)
        stop_measurement(link)

        _log.critical("Break")

        ni.set_level_LED(0*signal.mean(), None, sleep_time)

        
        output, time_array = ni.read_detector()

        ni.end_tasks()
        reset_arduino(link)

        result = read_pulses(ni, output, time_array, arduino_green)
        blank_level = np.mean(result[2][0:2])

        for i in range(len(result[0])):
            _run.log_scalar("Measure pulse-%0.1e"%freq, result[0][i] - blank_level, i*period_ML/1000)
            _run.log_scalar("Fluorescence", result[2][i], i*period_ML/1000)
            _run.log_scalar("MPPC", result[4][i], i*period_ML/1000)

    
        F0 = np.array(result[0][0:2] - blank_level).mean()
        FM = np.array(result[0] - blank_level).max()

        _run.log_scalar("Quantum yield", (FM-F0)/FM, np.log10(freq))



    ctrlLED.disconnect()

    plt.legend()
    plt.show()
    ipdb.set_trace()
    link.close()

    ni.end_exp(__file__)



    