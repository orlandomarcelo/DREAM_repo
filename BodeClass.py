import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import importlib
import glob
from scipy.signal import find_peaks

import tools
import math_functions as mf

class BodeClass:
    def __init__(self, name, rec_string, frequency_list, median_filtering_windos_size = 1, padding = False, padding_value = None, pic_search_window = 2):
        self.name = name
        self.rec_string = rec_string
        self.frequency_list = frequency_list
        self.median_filtering_windos_size = median_filtering_windos_size
        self.padding = padding
        self.padding_value = padding_value
        self.pic_search_window = pic_search_window
        self.folder = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/JTS/{name}/separated_files/"
        self.files = self.files = sorted(glob.glob(self.folder + "*.dat"), key=os.path.getmtime)
        self.clean_times = []
        self.clean_data = []
        self.recordings = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, index_col=False)
            aux_time = np.array(df.iloc[:,0])
            for key in df.keys()[1:]:
                self.recordings.append(key.replace(" ",""))   
                self.clean_times.append(aux_time)
                self.clean_data.append(np.array(df[key]))
        
        
        self.bode_records = tools.create_record_list(self.rec_string)
        self.bode_times = []
        self.bode_data = []

        for i in self.bode_records:
            index = np.where(np.array(self.recordings) == i)[0][0]
            self.bode_times.append(self.clean_times[index])
            self.bode_data.append(tools.median_filter(self.clean_data[index]-self.clean_data[index][0],self.median_filtering_windos_size))
            
        self.fund_freq = []
        self.fund_amp = []
        self.index_fund = []

        for i, k in enumerate(self.bode_records):
            F, A, _ = tools.FFT(self.bode_times[i]/1000, self.bode_data[i], padding, padding_value)
            self.index_fund.append(tools.closest_index(F, frequency_list[i]))
            peak = np.argmax(A[self.index_fund[i] - pic_search_window:self.index_fund[i] + pic_search_window]) + self.index_fund[i] - pic_search_window
            self.fund_freq.append(F[peak])
            self.fund_amp.append(A[peak])
            
            
        


                    
        