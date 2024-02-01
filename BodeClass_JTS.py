import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import importlib
import glob
from scipy.signal import find_peaks
from scipy.signal import windows as wd



module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import EkClass_JTS
import bode_tools

import tools
import math_functions as mf
importlib.reload(tools)
importlib.reload(bode_tools)

class BodeClass:
    def __init__(self, name, rec_string, frequency_list, flash_calib, index_start = 0, median_filtering_windos_size = 1, windowing = None, padding = False, padding_value = None, phase_threshold = 5, pic_search_window = 2, number_of_harmonics = 3):
        self.name = name
        
        self.date = re.findall(r'\d+', name)
        
        self.flash_calib = flash_calib
        self.index_start = index_start
        self.rec_string = rec_string
        self.frequency_list = frequency_list
        self.median_filtering_windos_size = median_filtering_windos_size
        self.windowing = windowing
        self.padding = padding
        self.padding_value = padding_value
        self.phase_threshold = phase_threshold
        self.pic_search_window = pic_search_window
        self.number_of_harmonics = number_of_harmonics
        self.folder = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/JTS/{name}/separated_files/"
        self.fig_folder = f"{self.folder}Figures"
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
               
        self.files = self.files = sorted(glob.glob(self.folder + "*.dat"), key=os.path.getmtime)
        self.clean_times = []
        self.clean_data = []
        self.recordings = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, index_col=False)
            aux_time = np.array(df.iloc[:,0])
            for key in df.keys()[1:]:
                self.recordings.append(key.replace(" ",""))   
                self.clean_times.append(aux_time[self.index_start:])
                self.clean_data.append(np.array(df[key])[self.index_start:])
        
        
        self.bode_records = tools.create_record_list(self.rec_string)
        self.bode_times = []
        self.bode_data = []

        for i in self.bode_records:
            index = np.where(np.array(self.recordings) == i)[0][0]
            self.bode_times.append(self.clean_times[index]/1000)
            self.bode_data.append(tools.median_filter(self.clean_data[index]-self.clean_data[index][0],self.median_filtering_windos_size)/self.flash_calib)
        
        self.filtered_data = bode_tools.band_pass_filter(self)
        
        
        self.signal, self.fft_freq, self.fft_amp, self.fft_phase, self.harmonics = bode_tools.get_bode_diagram(self)