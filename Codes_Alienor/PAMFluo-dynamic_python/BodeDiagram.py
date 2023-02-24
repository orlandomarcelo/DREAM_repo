    
import NIControl.BodeDiagram
import numpy as np
import alienlab
import matplotlib.pyplot as plt
import time
from alienlab.init_logger import logger
from scipy import optimize
import alienlab.regression_func
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from ThorlabsControl.FW102 import FW102C

from config_DAQ import *
from ArduinoControl.python_serial import *
from serial import *

from NIControl import DronpaClass
from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
import ipdb

ex = Experiment('bode_diagram', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger

@ex.config
def my_config():
    name = "spectral_content"

    trigger_color = "blue"
    acq_time = 60

    gen_ana = [NI_blue, NI_purple]
    sleep_time = 1
    N = 20
    min_freq = -1.5
    max_freq = 4.8

    amplitude = 2.5
    offset = 2.5

    actinic_filter = 2

@ex.automain
def bode_diagram(_run, _log, name, gen_ana, sleep_time, N, min_freq, max_freq, amplitude, offset, actinic_filter):

    # LED control tool DC4104 Thorlabs

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, 800)
    ctrlLED.set_user_limit(LED_green, 0)
    ctrlLED.set_user_limit(LED_purple,0)

    ctrlLED.disconnect()

    fwl = FW102C(port="COM3")
    fwl.move_to_filter(actinic_filter)
    fwl.filter = actinic_filter

    fwl.move_to_filter(filters[actinic_filter])


    bode = NIControl.BodeDiagram.BodeDiagram()
    bode.experiment_folder("bode_diagram")
    bode.cop.save_folder = bode.save_folder

    logger.name = "Bode_diagram"

    #Clear arduino
    link = Serial("COM4", 115200)
    time.sleep(2.0)
    reset_arduino(link)

    # Bode diagram cut-off fequency

    bode.generator_analog_channels = [generator_analog[color] for color in gen_ana]
    bode.generator_digital_channels = []
    bode.excitation_frequency = 1
    bode.trig_chan_detector_edge = 'FALLING'

    
    frequencies = 10**np.linspace(min_freq, max_freq, N)
    frequencies = np.append(frequencies,  frequencies[::-1])
    frequencies = np.append(frequencies, frequencies)
    frequencies = np.append(1, frequencies)

    bode.excitation_frequency = 1
    bode.num_period = 10
    bode.points_per_period = 50
    bode.update_rates()
    #480
    bode.offset =  offset
    bode.amplitude = amplitude
    signal_480 = bode.square_pattern()
    #405
    bode.offset = offset
    bode.amplitude = amplitude
    signal_405 = bode.sine_pattern()

    signal_12 = np.stack((signal_480, signal_405), axis = 0)

    radius, phase, all_outputs, sin_lo, cos_lo, all_outputs_3D = bode.bode_diagram(frequencies, signal_12, 2)
    x0 = [0.1, 0.1, 0]

    pulse = 2 * np.pi * frequencies
    parameters_estimated = optimize.least_squares(alienlab.regression_func.residuals,  x0, bounds = (-1e5,1e5),
                                        args = (pulse, sin_lo[:,1], alienlab.regression_func.band_pass)).x
    tau_sin = parameters_estimated[1]
    _log.debug(tau_sin)

    parameters_estimated = optimize.least_squares(alienlab.regression_func.residuals,  x0, bounds = (-1e5,1e5),
                                        args = (pulse, cos_lo[:,1], alienlab.regression_func.low_pass)).x
    tau_cos = parameters_estimated[1]
    _log.debug(tau_cos)
    parameters_estimated = optimize.least_squares(alienlab.regression_func.residuals,  x0, bounds = (-1e5,1e5),
                                        args = (pulse, radius[:,1], alienlab.regression_func.amplitude)).x
    tau_radius = parameters_estimated[1]
    _log.debug(tau_radius)



    for i in range(len(radius[:,0])):
        _run.log_scalar("Intensity", radius[i,0], np.log10(frequencies[i]))
        _run.log_scalar("Fluorescence", radius[i,1], np.log10(frequencies[i]))
        _run.log_scalar("Corrected Fluorescence", radius[i,1]/radius[i,0], np.log10(frequencies[i]))

    from mvgavg import mvgavg
    plt.plot(all_outputs[2])
    plt.figure()
    plt.plot(mvgavg(all_outputs[0], 5))
    plt.figure()
    plt.plot(mvgavg(all_outputs[1], 5))
    ipdb.set_trace()
    """
    bode.p.ylog = "none"

    bode.p.label_list = ['amplitude', 'qE_decay']
    bode.p.xlabel = "time (s)"
    bode.p.ylabel = "voltage (V)"
    bode.p.save_name = "qE_decay"
    fig = bode.p.plotting(, [average_output[0], average_output[1]])
    bode.p.saving(fig)
    _run.add_artifact((bode.p.save_path + bode.p.extension))
    """

    """
    test = DronpaClass.DronpaIntensity()
    test.save_folder = bode.save_folder
    test.p.save_folder = bode.save_folder
    p = test.p
    test.cop.save_folder = bode.save_folder
    cop = test.cop
    logger.name = "Dronpa2_intensity"

    test.generator_analog_channels = ["ao0", "ao1"]
    test.generator_digital_channels = []
    test.excitation_frequency = 1
    test.trig_chan_detector_edge = 'FALLING'

    test.num_period = 5
    test.points_per_period = 10000
    N_points = 10
    voltage = np.linspace(0.1, 2, N_points)
    filter_LED = 1

    tau_480_tot, tau_480_405_tot, val_480, val_405 = test.intensity_range(voltage, N_points, filter_LED)
    ipdb.set_trace()

    low_480, low_405 = test.analyse_results(voltage, tau_480_tot, tau_480_405_tot, val_480, val_405)

    #int_480 = utils.light_intensity.set_intensity(4, low_480['I_480'], 2*voltage)
    #int_405 = utils.light_intensity.set_intensity(2, low_405['I_405'], voltage)

    #sigma_480 = 198
    #sigma_405 = 415 #m²/mol
    #kdelta = 0.014

    #expected_tau_480 = sigma_480 * int_480 + kdelta
    #expected_tau_405_480 = expected_tau_480 + sigma_405 * int_405

    #logger.critical(tau_cos, tau_sin, tau_radius, 1 / expected_tau_480, 1 / expected_tau_405_480)
    """
    bode.end_exp(__file__)