import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import importlib
import glob
from scipy.signal import find_peaks
from scipy.signal import windows as wd

import tools
import math_functions as mf

class BodeClass:
    def __init__(self, name, rec_string, frequency_list, flash_calib, index_start = 0, median_filtering_windos_size = 1, windowing = None, padding = False, padding_value = None, pic_search_window = 2):
        self.name = name
        self.flash_calib = flash_calib
        self.index_start = index_start
        self.rec_string = rec_string
        self.frequency_list = frequency_list
        self.median_filtering_windos_size = median_filtering_windos_size
        self.windowing = windowing
        self.padding = padding
        self.padding_value = padding_value
        self.pic_search_window = pic_search_window
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
            self.bode_times.append(self.clean_times[index])
            self.bode_data.append(tools.median_filter(self.clean_data[index]-self.clean_data[index][0],self.median_filtering_windos_size)/self.flash_calib)
            
        self.fund_freq = []
        self.fund_amp = []
        self.index_fund = []
        self.freqs = []
        self.amps = []
        self.signal = []

        for i, k in enumerate(self.bode_records):
            if self.windowing is None:
                self.signal.append(self.bode_data[i])
            elif self.windowing == "flat-top":
                self.signal.append(self.bode_data[i] * wd.flattop(len(self.bode_data[i])))
                
            F, A, _ = tools.FFT(self.bode_times[i]/1000, self.signal[i], padding, padding_value)
            self.freqs.append(F)
            self.amps.append(A)
            self.index_fund.append(tools.closest_index(F, frequency_list[i]))
            peak = np.argmax(A[self.index_fund[i] - pic_search_window:self.index_fund[i] + pic_search_window]) + self.index_fund[i] - pic_search_window
            self.fund_freq.append(F[peak])
            self.fund_amp.append(A[peak])
            
    def plot_bode(self, fig = None, ax = None, leg = None, fmt = 'o', line = 2.5, marker = 4, marker_color = "black", 
                  fig_tile = None, show_fit = True):
        if ax is None:
            fig, ax = plt.subplots()
        
        if fig_tile is None:
            fig_title = "Bode plot of the fundamental harmonic"
            
        if leg is None:
            leg =  f"{self.bode_records[0]}-{self.bode_records[-1]}"
        
        orange = [250/255, 116/255, 79/255]
        
        if show_fit:

            self.popt_RC, self.pcov_RC, self.ffit_RC, self.afit_RC = tools.RC_transf_fit(self.fund_freq, 
                                                                                        self.fund_amp, 0.007, 130, 1000, p0 =  [10, 0.1])
            ax.plot(self.ffit_RC, self.afit_RC, linewidth=line, color = orange, label = "RC model")
            
            RC_err_neg=[mf.RC_transfer(x,*(self.popt_RC-2*tools.my_err_vec(x, self.pcov_RC))) for x in self.ffit_RC]
            RC_err_pos=[mf.RC_transfer(x,*(self.popt_RC+2*tools.my_err_vec(x, self.pcov_RC)))for x in self.ffit_RC]

            ax.fill_between(np.linspace(0.007, 130, 1000), RC_err_neg, RC_err_pos, alpha=0.2, color = orange)
        
        ax.plot(self.fund_freq , self.fund_amp, fmt, markersize=marker, color = marker_color, label = leg)

        ax = tools.bode_plot_axes(ax)

        fig.tight_layout()

        fig.savefig(f"{self.fig_folder}/{fig_tile}_{self.bode_records[0]}-{self.bode_records[-1]}.png", dpi=300)
        return ax
    
    def plot_all_TF(self):
        
        for i, k in enumerate(self.bode_records):
            fig, ax = plt.subplots(1,2, figsize = (10,5))
            
            if self.frequency_list[i] < 1:
                fig_title = f"{k}, P = {1/self.frequency_list[i]:n} s"
            elif self.frequency_list[i] == 1:
                fig_title = f"{k}, P = {1/self.frequency_list[i]:n} s, F = {self.frequency_list[i]} Hz "
            elif self.frequency_list[i] > 1:
                fig_title = f"{k}, F = {self.frequency_list[i]} Hz "
                    
            fig.suptitle(fig_title, fontsize = 16)
            ax[0].plot(self.bode_times[i]/1000, self.bode_data[i], "o-", markersize=1, linewidth=0.5)
            ax[1].plot(self.freqs[i], self.amps[i], "o-", markersize=1, linewidth=0.5)
            ax[1].plot(self.fund_freq[i], self.fund_amp[i], "x", markersize=3)            
            
            ax[0].set_xlabel("Time (s)", fontsize = 14)
            ax[0].set_ylabel("$ΔA_{ 520 nm} (r. u. e^- PS^{-1})$", fontsize = 12)
            ax[0].set_title("Time-domain signal", fontsize = 14)
            ax[1].set_xlabel("Frequency (Hz)", fontsize = 14)
            ax[1].set_ylabel("Amplitude (a. u.)", fontsize = 14)
            ax[1].set_title("Fourier transform", fontsize = 14)
            
            fig.tight_layout()
            
            fig.savefig(f"{self.fig_folder}/{fig_title}.png")
            
    def plot_record_TF(self, record, color = None, leg = None, fig = None, ax = None, fmt = '0-', line = 0.5, marker = 1):
        if ax is None:
            fig, ax = plt.subplots(1,2, figsize = (10,5))
        if leg is None:
            leg = record
        if color is None:
            color = "C0"
            
        i = np.where(np.array(self.bode_records) == record)[0][0]

        ax[0].plot(self.bode_times[i]/1000, self.bode_data[i], "o-", color = color, markersize=marker, linewidth=line)
        ax[1].plot(self.freqs[i], self.amps[i], "o-",color = color, markersize=marker, linewidth=line,label = leg)
        ax[1].plot(self.fund_freq[i], self.fund_amp[i], "x", markersize=10)            
        
        ax[0].set_xlabel("Time (s)", fontsize = 14)
        ax[0].set_ylabel("$ΔA_{ 520 nm} (r. u. e^- PS^{-1})$", fontsize = 12)
        ax[0].set_title("Time-domain signal", fontsize = 14)
        ax[1].set_xlabel("Frequency (Hz)", fontsize = 14)
        ax[1].set_ylabel("Amplitude (a. u.)", fontsize = 14)
        ax[1].set_title("Fourier transform", fontsize = 14)
        
        return ax
    
            
            
            
        


                    
        