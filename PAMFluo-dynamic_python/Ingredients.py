import numpy as np
from sacred import Ingredient
from config_DAQ import *
import matplotlib.pyplot as plt
import ipdb
from mvgavg import mvgavg
from ArduinoControl.python_serial import reset_arduino, add_master_digital_pulse, add_digital_pulse, start_measurement, stop_measurement


"""basic bricks for experiments"""


pulse_ingredient = Ingredient('pulses')

@pulse_ingredient.config
def cfg():
    step_delay = 1500
    average_window = 100
    amplitude_diff = 1


@pulse_ingredient.capture
def read_pulses(ni, output, time_array, arduino_color, _run, step_delay, average_window, amplitude_diff, arduino_amplitude = arduino_purple):

    a = output[arduino_color]
    b = np.roll(output[arduino_color], amplitude_diff)
    c = a-b

    indices = np.linspace(0, len(c)-1, len(c)).astype(int)
    indices = indices[c>=1]

    start = indices[0]
    list_mean = []
    list_std = []
    list_blank = []
    list_blank_std = []
    list_intense = []
    list_intense_std = []
    list_pulses = []
    list_pulses_blank = []
    list_arduino = []
    list_SP = []
    list_time = []

    for stop in indices[1:]:
        start = start + step_delay
        measuring_pulse = output[arduino_color, start:stop]
        measuring_pulse = measuring_pulse > 0
        blank = measuring_pulse == 0

        fluo = output[fluorescence, start:stop]
        fluo = fluo[measuring_pulse]
        print(len(fluo))
        list_mean.append(np.mean(fluo))
        list_std.append(np.std(fluo))
        
        fluo_blank = output[fluorescence, start:stop]
        fluo_blank = fluo_blank[blank]
        list_blank.append(np.mean(fluo_blank))
        list_blank_std.append(np.std(fluo_blank))

        list_pulses.append(fluo)
        list_pulses_blank.append(output[fluorescence, start:stop])
        list_arduino.append(output[[arduino_blue, arduino_purple, arduino_green], start:stop])

        fluo_pulse = output[fluorescence, start:stop]
        amplitude_pulse = output[arduino_amplitude, start:stop] > 0
        fluo_pulse = fluo_pulse[amplitude_pulse]
        list_SP.append(np.mean(fluo_pulse))

        intense = output[intensity, start:stop]
        intense = intense[amplitude_pulse]    
        list_intense.append(np.mean(intense))
        list_intense_std.append(np.std(intense))

        list_time.append(np.mean(time_array[start:stop]))
        start = stop

    ni.window = average_window
    average_output, downscaled_time = ni.averaging(output, time_array)

    if False: 

        fig = plt.figure()
        ni.p.xlabel = "time (s)"
        ni.p.ylabel = "voltage (V)"
        ni.p.save_name = "fluorescence" 
        fig = ni.p.plotting(time_array, output[fluorescence])
        ni.p.saving(fig)
        #_run.add_artifact(ni.p.save_path + ni.p.extension)

    if False:
        #plt.pause(2)
        ni.p.label_list = detector
        ni.p.xlabel = "time (s)"
        ni.p.ylabel = "voltage (V)"
        ni.p.save_name = "output_plot" 
        fig = ni.p.plotting(time_array, [output[i] for i in range(output.shape[0])])
        
        ni.p.saving(fig)
        _run.add_artifact(ni.p.save_path + ni.p.extension)

    result = list_mean, list_std, list_blank, list_blank_std, list_intense, list_intense_std, list_pulses, list_pulses_blank, list_arduino, list_SP, list_time
    elems =  []
    for elem in result[:6]:
        elem = np.array(elem)
        elem[elem != elem] = 0 #remove nan to avoid plotting error
        elems.append(elem)
    if True: 
        ni.p.xlabel = "pulse number"
        ni.p.save_name = ni.experiment_name
        x_val = np.array(range(len(list_mean)))
        ni.p.ylabel = "voltage"
        ni.p.label_list = ['pulse_mean', 'pulse_std', 'blank_mean', 'blank_std', 'intensity_mean', 'intensity_std']
        fig = ni.p.plotting([x_val]*6, [np.array(elem) for elem in elems[0:6]])
        
        ni.p.saving(fig)
        _run.add_artifact(ni.p.save_path + ni.p.extension)
        _run.add_artifact(ni.p.save_path + ".csv")


    return result


@pulse_ingredient.capture
def lock_in(measured_signal, 
                time_array, 
                frequency, phase_shift, mvg):
    cos_ref = np.cos(2*np.pi * frequency * time_array - phase_shift)
    cos_ref = np.stack([cos_ref] * measured_signal.shape[0])
    sin_ref = np.sin(2*np.pi * frequency * time_array - phase_shift)
    sin_ref = np.stack([sin_ref] * measured_signal.shape[0])
    cos_lock = 2 * np.multiply(measured_signal, cos_ref)
    cos_lock  = mvgavg(cos_lock.T, mvg)
    sin_lock =  2 * np.multiply(measured_signal, sin_ref)
    sin_lock = mvgavg(sin_lock.T, mvg)
    radius_lock = np.sqrt(sin_lock**2 + cos_lock**2)
    phase_lock = np.arctan(sin_lock.mean(axis = 1)/cos_lock.mean(axis = 1))
    
    return sin_lock, cos_lock, radius_lock.T, phase_lock


@pulse_ingredient.capture
def speed_time_response_image(task, frames, kwargs):
        return np.asarray(Parallel(n_jobs=-1)(delayed(task)(skimage.util.img_as_ubyte(frames[i]), **kwargs) 
                                              for i in self.selected_inds))    

@pulse_ingredient.capture
def mean_by_indices(list_indices, array):
    output_array = np.zeros(len(list_indices)-1)
    for i in range(len(list_indices)-1):
        output_array[i] = np.mean(array[list_indices[i]:list_indices[i+1]])
    return output_array


@pulse_ingredient.capture
def saturating_pulse(ni, ctrlLED, LED_sat, link, period_SP, length_SP, limit_sat):
    
    ni.trig_chan_detector = trigger["no_LED_trig"]
    ctrlLED.set_user_limit(LED_sat, int(limit_sat))

    ni.reading_samples = int(ni.sample_rate * ni.acq_time)

    ni.detector_init()
    ni.trigger_init()

    reset_arduino(link)

    add_master_digital_pulse(link, pins['no_LED_trig'], 0, period_SP, 20, 0) #trigger
    add_digital_pulse(link, pins["blue"], 1000, period_SP, length_SP, 0)

    # Acquisition
    ni.detector_start()

    start_measurement(link)
    output, time = ni.read_detector()
    stop_measurement(link)

    ni.end_tasks()
    reset_arduino(link)


    sat_pulse = output[LED_sat] > 0

    blank_level = np.mean(output[fluorescence][1-sat_pulse])

    pulse_val = np.mean(output[fluorescence][sat_pulse])
    plt.plot(time, output[fluorescence])
    return pulse_val, blank_level



