import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import importlib
import glob
from scipy.signal import find_peaks
from scipy.signal import windows as wd
import ExperimentClass

import tools
import math_functions as mf

class BodeClass(ExperimentClass.Experiment):
    
    def __init__(self, name, bode_records, frequency_list, Fm_calib, index_start = 0, time_start = None, median_filtering_windos_size = 1, windowing = None, padding = False, padding_value = None, pic_search_window = 2):
        super().__init__(name, "PSI", DataType = ".csv", sep = ';')
        self.bode_records = bode_records
        self.frequency_list = frequency_list
        self.median_filtering_windos_size = median_filtering_windos_size
        self.windowing = windowing
        self.padding = padding
        self.padding_value = padding_value
        self.pic_search_window = pic_search_window
        self.recordings = self.records
        self.Fm_calib = Fm_calib
        
        if time_start is None:
            self.index_start = index_start
        else:
            self.index_start = np.where(self.Time > time_start)[0][0]
        
        self.bode_times = []
        self.bode_data = []

        for i in self.bode_records:
            index = self.recordings.index(i)
            self.bode_times.append(self.clean_times[index][self.index_start:])
            self.bode_data.append(tools.median_filter(self.clean_data[index][self.index_start:],self.median_filtering_windos_size)/self.Fm_calib)
            
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
                
            F, A, _ = tools.FFT(self.bode_times[i], self.signal[i], padding, padding_value)
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
            
            err = tools.my_err(self.ffit_RC, self.popt_RC, self.pcov_RC, mf.RC_transfer)
            ax.fill_between(self.ffit_RC, self.afit_RC - err, self.afit_RC + err, alpha=0.05, color = 'k')
        
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
            ax[0].plot(self.bode_times[i], self.bode_data[i], "o-", markersize=1, linewidth=0.5)
            ax[1].plot(self.freqs[i], self.amps[i], "o-", markersize=1, linewidth=0.5)
            ax[1].plot(self.fund_freq[i], self.fund_amp[i], "x", markersize=3)            
            
            ax[0].set_xlabel("Time (s)", fontsize = 14)
            ax[0].set_ylabel("Fluorescence (r. u.)", fontsize = 12)
            ax[0].set_title("Time-domain signal", fontsize = 14)
            ax[1].set_xlabel("Frequency (Hz)", fontsize = 14)
            ax[1].set_ylabel("Amplitude (r. u.)", fontsize = 14)
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

        ax[0].plot(self.bode_times[i], self.bode_data[i], "o-", color = color, markersize=marker, linewidth=line, label = leg)
        ax[1].plot(self.freqs[i], self.amps[i], "o-",color = color, markersize=marker, linewidth=line, label = leg)
        ax[1].plot(self.fund_freq[i], self.fund_amp[i], "x", markersize=10)            
        
        ax[0].set_xlabel("Time (s)", fontsize = 14)
        ax[0].set_ylabel("Fluorescence (r. u.)", fontsize = 12)
        ax[0].set_title("Time-domain signal", fontsize = 14)
        ax[1].set_xlabel("Frequency (Hz)", fontsize = 14)
        ax[1].set_ylabel("Amplitude (r. u.)", fontsize = 14)
        ax[1].set_title("Fourier transform", fontsize = 14)
        
        return ax
    
            
            
            
        


                    
        