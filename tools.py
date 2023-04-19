import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import os

def clean_spaces(vector):
    clean_vector = []
    for k in vector:
        try:
            clean_vector.append(float(k.strip()))
        except ValueError:
            clean_vector.append(float('nan'))
    return clean_vector

def turbidostat_data(turbido_data, data_list):
    lights = []
    OD_680 = []
    OD_720 = []
    pump_vol = []
    temperature = []
    times =  [ [] for _ in range(len(data_list)) ]
    aux_time = np.array(turbido_data.iloc[:,0])
    for p, i in enumerate(data_list):
        for j, k in enumerate(turbido_data.keys()):
            if i in k:
                A = np.array(clean_spaces(turbido_data.iloc[:,j]))
                indices = np.invert(np.isnan(A))
                times[p].append(aux_time[indices])
                if i == 'light':
                    times[p].append(aux_time[indices])
                    lights.append(A[indices])
                elif i == '680':
                    OD_680.append(A[indices])
                elif i == '720':
                    OD_720.append(A[indices])
                elif i == 'consumption':
                    pump_vol.append(A[indices])
                elif i == 'temperature':
                    temperature.append(A[indices])    
    return lights, OD_680, OD_720, pump_vol, temperature, times


