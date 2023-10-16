from IBPC_pulse import ex
import numpy as np
import time
import ipdb 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from VoltageIntensityClass import VoltageIntensity
from config_DAQ  import *

quantum_evolve = []
actinic_evolve =  []
fm = []
fs = []
f0 = []
V = VoltageIntensity()

#NN = 8
#filter_list = np.array([2]*NN + [1]*NN)#[::-1]
#limits_blue = np.array(list(np.linspace(0, 450, NN))+list(np.linspace(0, 450, NN)))#[::-1]
green_list = np.tile([150,0],30)
filter_list = np.repeat([3,3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1,1,0,0], 3)
limits_blue = np.repeat([0, 50, 95, 150, 34, 61, 82, 140, 190, 420, 140, 170, 330, 5, 10], 3)

limits_blue = np.repeat([0, 3, 8, 10, 13, 19, 33, 45, 84, 121, 146, 255], 2)

filter_list = np.repeat([1], len(limits_blue))

limits_actinic = []
f_direct = []
filter_val = []
blue_val = []
move_plate=False
assert len(limits_blue)==len(filter_list)
#ipdb.set_trace()
for i, f in enumerate(filter_list): 
                        limit_blue = limits_blue[i]
                        r = ex.run(config_updates={"limit_green": int(green_list[i]), 'limit_blue': int(limit_blue), 'actinic_filter': f, 'move_plate': move_plate})
                        actinic_evolve.append(r.result[0])
                        quantum_evolve.append(r.result[1])
                        f0.append(r.result[2])
                        fs.append(r.result[3])
                        fm.append(r.result[4])
                        f_direct.append(r.result[5])

                        
                        time.sleep(55)
                        limits_actinic.append(V.get_intensity_voltage('blue', f, limit_blue))
                        filter_val.append(f)
                        blue_val.append(limit_blue)

df = pd.DataFrame(list(zip(actinic_evolve, quantum_evolve, f0, fs, fm, f_direct, limits_actinic, filter_val, blue_val)),
               columns =['Actinic evolve', 'quantum evolve','f0', 'fs', 'fm', 'f_direct', 'limits_actinic', 'filter', 'blue'])
df["calibration"] = V.experiment_folder

date = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_'))
df.to_csv("G:/DREAM/from_github/PAMFluo/Validation/saturation_evolution/" + date + "_vs_IBPC_evolution.csv")
#ipdb.set_trace()
print("pouet")