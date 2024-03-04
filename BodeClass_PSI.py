import numpy as np
import pandas as pd
import os
import sys
import re
import importlib
import glob
from scipy.signal import find_peaks
from scipy.signal import windows as wd
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import tools
import bode_tools
import math_functions as mf
from ExperimentClass import Experiment

importlib.reload(tools)
importlib.reload(bode_tools)

class BodeClass_PSI(Experiment):
    def __init__(self, name, bode_records = None, frequency_list = None, normalization = "F_stat", Fm_calib = 1, index_start = 0, time_start = None, median_filtering_window_size = 1, 
                 windowing = None, padding = False, padding_value = None, phase_threshold = 5, pic_search_window = 2, number_of_harmonics = 3, detrend = True):
        super().__init__(name, "PSI", DataType = ".csv", sep = ';')
        
        self.date = re.findall(r'\d+', name)
        
        self.number_of_harmonics = number_of_harmonics
        
        if bode_records is None:
            self.bode_records = ["P0.0078125s", "P0.015625s", "P0.03125s", "P0.0625s", "P0.125s", "P0.25s", "P0.5s", "P1s_5", "P2s", "P4s", "P8s", "P16s", "P32s", "P64s", "P128s"]
        else:           
            self.bode_records = bode_records
        
        if frequency_list is None:
            self.frequency_list = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128][::-1]
        else:
            self.frequency_list = frequency_list
        self.median_filtering_window_size = median_filtering_window_size
        self.detrend = detrend
        self.windowing = windowing
        self.padding = padding
        self.padding_value = padding_value
        self.phase_threshold = phase_threshold
        self.pic_search_window = pic_search_window
        self.Fm_calib = Fm_calib
        self.normalization = normalization
        
        if time_start is None:
            self.index_start = index_start
        else:
            self.index_start = np.where(self.Time > time_start)[0][0]
        
        self.bode_times = []
        self.bode_data = []
        self.F_stat = []

        for i in self.bode_records:
            index = self.records.index(i)
            self.bode_times.append(self.clean_times[index][self.index_start:])
            self.F_stat.append(np.mean(self.clean_data[index][int(self.index_start/2):self.index_start]))
            if self.normalization == "F_stat":
                self.bode_data.append(tools.median_filter(self.clean_data[index][self.index_start:],self.median_filtering_window_size)/self.F_stat[-1])
            elif self.normalization == "F_max":
                self.bode_data.append(tools.median_filter(self.clean_data[index][self.index_start:],self.median_filtering_window_size)/self.Fm_calib)
            else:
                self.bode_data.append(tools.median_filter(self.clean_data[index][self.index_start:],self.median_filtering_window_size))
                
        self.filtered_data = bode_tools.band_pass_filter(self)
        
        self.signal, self.detrend_fit, self.fft_freq, self.fft_amp, self.fft_phase, self.harmonics = bode_tools.get_bode_diagram(self)
            
        