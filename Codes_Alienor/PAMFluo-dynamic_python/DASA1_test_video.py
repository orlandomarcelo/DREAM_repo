from re import S
import pandas as pd
from alienlab.init_logger import logger
from alienlab.regression_func import exp_decay, residuals
from scipy import optimize
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

ex = Experiment('IBPC_pulse', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

def exponential_fit(time, fluorescence):
    
    delta_time = time[-1]-time[0]
    dt = delta_time/len(time)
    
    x0 = [1e5, delta_time/8, 1]
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                        args = (time, fluorescence, exp_decay))
    #logger.critical(OptimizeResult.x)
    parameters_estimated = OptimizeResult.x
    tau = parameters_estimated[1]
    
    #conditions on tau too low or too high for the second, more accurate, fit, because we will fit on a signal that lasts 5*tau
    if tau >  delta_time//10: #if too high
        tau =  delta_time//10
    if tau < 3*dt: #if too low, increase it
        tau = 5*dt
    x0 = parameters_estimated #initial guess: parameters from previous fit
    #second fit
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
                                        args = (time[0:int(tau*5/dt)], fluorescence[0: int(tau*5/dt)], exp_decay))
    parameters_estimated = OptimizeResult.x

    
    # Estimate data based on the solution found
    y_data_predicted = exp_decay(parameters_estimated, time)
        
    return y_data_predicted, parameters_estimated



@ex.config
def my_config():

    name = "IBPC_pulse"
    actinic_filter = 1
    move_plate = False
    sample_rate = 1e6
    exposure = 0.09
    gain = 100
    trigger_color = "no_LED_trig"
    period = 0.1
    
    duration = 0.1
    
    
    red_level = 80
    limit_red=0
    limit_green=0
    limit_purple=0
    move_plate=False
    x0=0
    y0=0
    height=700
    width=600
    clock=474
    binning_factor=2
    subsampling_factor=1
    
    
    trigger_freq = 5*red_level/100 #Hz
    exposure= 1000/trigger_freq - 0.1 #ms 
    acq_time = 1/trigger_freq * 40 + 1

    
@ex.automain
def send_IBPC(_run, name, limit_blue, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, actinic_filter, move_plate, exposure, gain, red_level, duration, period, height, width, x0, y0, subsampling_factor, binning_factor,trigger_freq,clock):


    # LED control tool DC4104 Thorlabs
    logger.name = name


    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = red_level, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo = True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])
    #ipdb.set_trace()
    thread, cam = init_camera(exposure = exposure, num_frames = trigger_freq*acq_time, gain =  100, x=x0, height=height,
                              y=y0, width=width, subsampling_factor=subsampling_factor, clock=clock, binning_factor=binning_factor, trigger_frequency=trigger_freq, trigger=False)


    add_digital_pulse(link_arduino, pins['no_LED_trig'], 15, 300*sec, 300*sec, 1) #trigger

    add_digital_pulse(link_arduino, pins['blue'], 1*sec, 300*sec, 300*sec, 0)
   


    # Acquisition

    ni.detector_start()
    thread.start()

    start_measurement(link_arduino)
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


    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_blue, arduino_amplitude = arduino_blue) 
    
    
    result[2][0] = result[2][1]
    blank_level = np.mean(result[2][0:15])
    logger.critical(blank_level)


    np.savetxt(save_figure_folder + "/id_%05d_%s_measure_pulse.csv"%(_run._id, name), np.array(result[0]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_fluorescence.csv"%(_run._id, name), np.array(result[2]), delimiter=",")
    np.savetxt(save_figure_folder + "/id_%05d_%s_MPPC_intensity.csv"%(_run._id, name), np.array(result[4]), delimiter=",")




    #ipdb.set_trace()
    

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

    
    
    time = np.array(result[10])
    fluorescence = np.array(result[0])
    y_data_predicted, parameters_estimated = exponential_fit(time, fluorescence)
    
    for i in range(len(result[0])):
        _run.log_scalar("Measure pulse", result[0][i] - blank_level, result[10][i])
        _run.log_scalar("Fluorescence", result[2][i], result[10][i])
        _run.log_scalar("MPPC", result[4][i], result[10][i])
        _run.log_scalar("Fit", y_data_predicted[i]- blank_level, result[10][i])

    tau = parameters_estimated[1]
    # Plot all together
    ni.cop.title = "Voltage = %0.3f"%red_level
    ni.cop.xlabel = 'time (s)'
    ni.cop.ylabel = 'voltage (V)'
    ni.cop.y2label = 'residuals (V)'
    ni.cop.label_list = ['Real Data','Voltage = %0.3f, tau = %0.2f'%(red_level, tau), "MPPC"]
    ni.cop.label2_list = ['Residuals']

    ni.cop.xval = time
    ni.cop.yval = [fluorescence, y_data_predicted,result[4]]
    ni.cop.x2val = time
    ni.cop.y2val = y_data_predicted - fluorescence

    f = ni.cop.coplotting()
    ni.cop.save_name = 'exponential_fit_%d'%red_level
    ni.cop.saving(f)
    
    #ipdb.set_trace()


    return float(parameters_estimated[1])