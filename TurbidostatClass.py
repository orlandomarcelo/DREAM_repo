import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import pdb


class Turbidostat:
    
    def clean_spaces(self, vector):
        clean_vector = []
        for k in vector:
            try:
                clean_vector.append(float(k.strip()))
            except:
                clean_vector.append(float('nan'))
                
        indices = np.invert(np.isnan(clean_vector))
        return indices, np.array(clean_vector)               
    
    def __init__(self, name):
        self.name = name
        self.folder = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/Turbidostat/{self.name}"
        self.file = f"{self.folder}/{self.name}.csv"
        self.all_data = pd.read_csv(self.file, sep = ';', index_col=False)
        self.fig_folder = f"{self.folder}/Figures"
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
        self.mesurements = ['od-680', 'od-620', 'light', 'consumption']
        self.columns = self.all_data.keys()
        
        self.data = {'temperature': {}, 'ch1': {}, 'ch2': {}, 'ch3': {}, 'ch4': {}, 'ch5': {}, 'ch6': {}, 'ch7': {}, 'ch8': {}}
        
        for i, measurement in enumerate(self.mesurements):
            channel = 1
            for j, column in enumerate(self.columns):
                if measurement in column:
                    A = np.array(self.all_data.iloc[:,j])
                    indices, values = self.clean_spaces(A)
                    aux_time = np.array(self.all_data.iloc[:,0])
                    df = pd.DataFrame({'time':aux_time[indices], 'val': values[indices]})
                    self.data[f'ch{channel}'][measurement] = df
                    channel = channel + 1
                if 'temperature' in column:
                    A = np.array(self.all_data.iloc[:,j])
                    indices, values = self.clean_spaces(A)
                    aux_time = np.array(self.all_data.iloc[:,0])
                    df = pd.DataFrame({'time':aux_time[indices], 'val': values[indices]})
                    self.data['temperature'] = df
        
        
    
    
