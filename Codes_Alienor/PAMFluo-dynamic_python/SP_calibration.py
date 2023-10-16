from alienlab.init_logger import logger
import time
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links
import ipdb

ex = Experiment('SP_calibration', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")


@ex.config
def my_config():
    name = "SP_calibration"
    trigger_color = "green"

    N = 15
    acq_time = 60 * N


@ex.automain
def SP_calibration(_run, name, limit_blue, limit_green, limit_purple, trigger_color, sample_rate, acq_time, length_SP, length_ML, period_ML, N):

    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link = all_links[2]
    fwl = all_links[3]


    add_digital_pulse(link, pins['blue'], 15000, 30000, length_SP, 1)
    add_master_digital_pulse(link, pins['green'], length_SP + 1, 15000, length_ML, 0)
    LED = LED_blue
    ctrlLED.set_user_limit(LED, limit_purple)

    ni.detector_start()
    start_measurement(link)

    limits_purple_low = np.linspace(1, 45, N//2)
    limits_purple_high = np.linspace(50, 500, N//2)
    limits_purple = np.concatenate((limits_purple_low, limits_purple_high), axis = None)
    for limit_purple in limits_purple:
        ctrlLED.set_user_limit(LED, int(limit_purple))
        try: 
            print(crtlLED.get_user_limit(LED))
        except:
            pass
        time.sleep(60)

    output, time_array = ni.read_detector()
    stop_measurement(link)

    close_all_links(*all_links)



    result = read_pulses(ni = ni, output = output, time_array = time_array, arduino_color = arduino_green, arduino_amplitude = arduino_blue) 
    
    ipdb.set_trace()
    #for i in range(len(result[0])):
    #    _run.log_scalar("Measure pulse", result[0][i], i)
    #    _run.log_scalar("Fluorescence", result[2][i], i)
    #    _run.log_scalar("MPPC", result[4][i], i)

    fig = plt.figure()
    blank_level_fluo = np.mean(result[2][1::2])

    list_pulses = result[7]
    list_arduino = result[8]
    pulse_green = np.copy(result[0][1::2]) - blank_level_fluo
    pulse_f0 = np.copy(result[0][0::2]) - blank_level_fluo
    pulse_fluo_jump = np.copy(result[2][0::2]) - blank_level_fluo

    blank_level_intensity_index = np.sum(list_arduino[0], axis = 0) == 0
    blank_level_intensity = np.mean(list_pulses[0][blank_level_intensity_index])

    pulse_purple = result[9][0::2] - blank_level_fluo
    pulse_intensity = result[4][0::2] - blank_level_intensity

    pulse_green_norm = (pulse_green-pulse_green.min())/(pulse_green.max()-pulse_green.min())
    pulse_purple_norm = (pulse_purple-pulse_purple.min())/(pulse_purple.max()-pulse_purple.min())
    pulse_fluo_norm = (pulse_fluo_jump-pulse_fluo_jump.min())/(pulse_fluo_jump.max()-pulse_fluo_jump.min())

    plt.close("all")




    for i in range(len(pulse_green)-1):
        _run.log_scalar("Fo response", pulse_f0[i], pulse_intensity[i] )
        _run.log_scalar("Fm response", pulse_green[i], pulse_intensity[i])

        _run.log_scalar("Quantum Yield", (pulse_green[i]-pulse_f0[0])/pulse_green[i], pulse_intensity[i])
        _run.log_scalar("Fluorescence response", pulse_purple[i], pulse_intensity[i])

        _run.log_scalar("Pulse green norm", pulse_green_norm[i], i)
        _run.log_scalar("Pulse purple norm", pulse_purple_norm[i], i)
        _run.log_scalar("Pulse fluo norm", pulse_fluo_norm[i], i)

        _run.log_scalar("Pulse green", pulse_green[i], i)
        _run.log_scalar("Pulse purple", pulse_intensity[i], i)
        _run.log_scalar("Pulse fluo jump", pulse_purple[i], i)


    ni.p.save_name = "normalised_intensity_vs_response"
    ni.p.xval = pulse_intensity
    ni.p.yval = pulse_green
    fig = ni.p.plotting(ni.p.xval, ni.p.yval)

    ipdb.set_trace()

    ni.p.saving(fig)


"""
def plot_pulse(k, N):
    u = mvgavg(list_pulses[k], N); v = mvgavg(list_green[k], N); plt.plot(u, label = "%d"%k);plt.plot(v)
    plt.legend()

N = 100
k = 4
plot_pulse(5, N)
plot_pulse(7, N)
plot_pulse(12, N)
plt.show()
pos = list_green[k]>0
val = list_pulses[k][pos]
print(list_std[k])
print(np.std(val))

nev = list_mean[1::2]
eve = list_mean[::2]
fm = np.mean(eve)
fo = np.mean(nev)
(fm-fo)/fm

bis = (np.array(eve)-np.array(nev))/np.array(eve)

plt.plot(bis)
plt.show()

fig = plt.figure()
m = np.copy(list_mean[::2])
b = np.copy(list_blank[1::2])
plt.plot((b-b.min())/(b.max()-b.min()))
plt.plot((m-m.min())/(m.max()-m.min()))
plt.show()

ni.p.save_name = "normalised_intensity_vs_response"
ni.p.saving(fig)
"""