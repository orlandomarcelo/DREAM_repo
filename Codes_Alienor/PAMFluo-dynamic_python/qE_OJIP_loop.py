#from qE_OJIP_Benjamin import ex
from qE_OJIP_calib import ex as ex
import numpy as np
import time
import ipdb 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from VoltageIntensityClass import VoltageIntensity
from config_DAQ  import *

"""

lengths_pulse = [300, 150, 50]
periods_pulse = [0.1*minute, 0.5*minute, 1*minute]
move_plate = True

for i, length in enumerate(lengths_pulse): 
    for j, period in enumerate(periods_pulse):

        r = ex.run(config_updates={'length_ML': int(length), 'period_ML':int(period), 'exposure':length, 'move_plate': move_plate})

                

move_plate = False#True
NN = 40
filter_list = np.array([1]*NN)#[::-1]
#limits_blue = [0, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450]#np.array(list(np.linspace(0, 450, NN)))
#limits_blue = [0, 5, 10, 15, 20, 30,  50, 100, 150, 200, 300,  400]#np.array(list(np.linspace(0, 450, NN)))

limits_blue_high = [10, 20, 30, 40, 50, 100, 150, 160, 170, 180, 200, 220, 230, 250, 280, 300, 350, 400, 150]

for i, intensity in enumerate(limits_blue_high):#[20, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        r = ex.run(config_updates={'limit_blue_high': int(intensity), 'limit_blue_low': int(max(0, intensity - 150)), 'gain': 100, 'actinic_filter' : filter_list[i], 'move_plate': move_plate})
        time.sleep(60)
"""

move_plate = False#True
NN = 40
filter_list = np.array([1]*NN)#[::-1]
#limits_blue = [0, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450]#np.array(list(np.linspace(0, 450, NN)))
#limits_blue = [0, 5, 10, 15, 20, 30,  50, 100, 150, 200, 300,  400]#np.array(list(np.linspace(0, 450, NN)))

limits_blue_high = [10, 20, 30, 40, 50, 100, 150, 160, 170, 180, 200, 220, 230, 250, 280, 300, 350, 400, 150]

for i, intensity in enumerate(limits_blue_high):#[20, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        r = ex.run(config_updates={'camera_heat_delay': 15, 'HL_time': 3, 'limit_blue_high': int(intensity), 'limit_blue_low': 0, 'gain': 100, 'actinic_filter' : filter_list[i], 'move_plate': move_plate})
        time.sleep(60)
