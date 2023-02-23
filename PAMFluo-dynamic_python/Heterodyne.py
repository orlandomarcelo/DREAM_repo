    
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

from NIControl import DronpaClass, HeterodynClass
from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
import ipdb
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera

ex = Experiment('bode_diagram', ingredients=[pulse_ingredient])

ex.observers.append(MongoObserver())

ex.logger = logger

@ex.config
def my_config():
    name = "heterodyne"

    trigger_color = "blue"
    acq_time = 60

    gen_ana = [NI_blue, NI_purple]
    sleep_time = 1
    N = 5
    min_freq = 2
    max_freq = 3
    points_per_period = 200
    amplitude = 1
    offset = 2
    dw = 10
    actinic_filter = 4
    
    exposure = length_SP - 30
    trigger_color = "no_LED_trig"
    frequency = 1
    limit_blue = 50
    x0=400
    y0=0
    height=500
    width=400
    clock=474
    binning_factor=2
    subsampling_factor=1
    

@ex.automain
def bode_diagram(_run, _log, name, gen_ana, sleep_time, N, min_freq, max_freq, amplitude, offset, actinic_filter, points_per_period, dw, exposure, gain, frequency, x0, y0, height, width,      binning_factor, subsampling_factor, clock):

    # LED control tool DC4104 Thorlabs

    ctrlLED = ThorlabsDC4100(logger, port = "COM5")
    ctrlLED.initialise_fluo()

    ctrlLED.set_user_limit(LED_blue, 800)
    ctrlLED.set_user_limit(LED_green, 0)
    ctrlLED.set_user_limit(LED_purple,800)

    ctrlLED.disconnect()

    fwl = FW102C(port="COM3")
    fwl.move_to_filter(actinic_filter)
    #fwl.filter = filters[actinic_filter]

    #fwl.move_to_filter(filters[actinic_filter])

    thread, cam = init_camera(exposure = exposure, num_frames = frequency*acq_time, gain =  gain, x=x0, height=height,
                              y=y0, width=width, subsampling_factor=subsampling_factor, clock=clock, binning_factor=binning_factor)


    bode = HeterodynClass.Heterodyne()
    bode.experiment_folder("heterodyne")
    bode.cop.save_folder = bode.save_folder

    logger.name = "heterodyne"

    #Clear arduino
    link = Serial("COM4", 115200)
    time.sleep(2.0)
    reset_arduino(link)

    # Bode diagram cut-off fequency

    bode.generator_analog_channels = [generator_analog[color] for color in gen_ana]
    bode.generator_digital_channels = []
    bode.excitation_frequency = 1
    bode.trig_chan_detector_edge = 'FALLING'

    
    frequencies = 10**np.linspace(min_freq, max_freq, N)[::-1]

    bode.excitation_frequency = 10
    bode.num_period = 10
    bode.points_per_period = points_per_period
    bode.update_rates()
    #480
    bode.offset =  offset
    bode.amplitude = amplitude
    signal_480 = bode.sine_pattern_dw(5, 0.5, 0.5)
    #ipdb.set_trace()
    #405
    bode.offset = offset
    bode.amplitude = amplitude
    signal_405 = bode.sine_pattern_dw(5, 0.5, 0.5)

    signal_12 = np.stack((signal_480, signal_405), axis = 0)

    #radius, phase, all_outputs, sin_lo, cos_lo, all_outputs_3D, all_times = bode.fixed_dw(frequencies, dw, offset, amplitude, 0.#5, 2)
    
    radius, phase, all_outputs, sin_lo, cos_lo, all_outputs_3D, all_times = bode.LEDs_2_dw(frequencies, dw, offset, amplitude, 0.5, 2)
                                                                                          
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
    
    
    
    """"
    if True:
    	import mvgavg.mvgavg as avg
	I = all_outputs[0]
	plt.plot(avg(I, 100))
	
	
		
	def FFT(t, y):
		#source: https://stackoverflow.com/questions/56797881/how-to-properly-scale-frequency-axis-in-fast-fourier-transform
		n = len(t)
		Δ = (max(t) - min(t)) / (n-1)
		k = int(n/2)
		f = np.arange(k) / (n*Δ)
		Y = np.abs(np.fft.fft(y))[:k]
		#print("frequency resolution:", f[1]-f[0])
		return (f, Y)
		
	I = all_outputs_3D[3][0]
	t = all_times[3]
	
	
if True:
    
    
    plt.figure()
    
    for i in [0]: #"range(5):
            
    	I = all_outputs_3D[i][0]
        F = all_outputs_3D[i][1]
	    t = all_times[i]
        a = FFT(t, I)
        b = FFT(t, F)
        f = a[0][1:]
        Y = a[1][1:]
        Y = Y/Y.max()
        Z = b[1][1:]
        Z = Z/Z.max()
        plt.plot(f, Y, "k")#, marker = "o")
        plt.plot(f, Z, 'r')#, marker = "o")
        #plt.plot(f, Z/Y)
	plt.show()
	
	
if True:
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))
    plt.title("dw=%d"%dw)
    for j in range(5):
        

        for i in [0, 1]:
            from mvgavg import mvgavg
            I = all_outputs_3D[j][i]
            t = all_times[j]
            axs[j][i].plot(mvgavg(t, 20), mvgavg(I, 20), label = "%d"%frequencies[j])
            axs[j][i].legend()
            #plt.plot(t, I)
    plt.show()
	
    """