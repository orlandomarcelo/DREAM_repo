import numpy as np
import pandas as pd
import os
import sys
import re
import importlib
import glob
from scipy.signal import find_peaks
from scipy.signal import windows as wd
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
import tools

def band_pass_filter(bode_object, low_factor = 0.5, high_factor = 10):
    filtered_data = []
        
    for i, freq in enumerate(bode_object.frequency_list):
        sample_rate = 1/(bode_object.bode_times[i][1] - bode_object.bode_times[i][0])
        cut_off_low = freq * low_factor
        cut_off_high = freq * high_factor
        filtered_data.append(tools.band_pass_filter(bode_object.bode_data[i] - np.mean(bode_object.bode_data[i]), cut_off_low, cut_off_high, sample_rate))
        
    return filtered_data
    

def spline_detrending(ydata, order = 2, dspline = 30):
    data = ydata.copy()
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)

    x = np.arange(len(data))
    splknots = np.arange(dspline / 2.0, len(data) - dspline / 2.0 + 2, dspline)

    spl = LSQUnivariateSpline(x=x, y=data, t=splknots, k=order)
    fit = spl(x)

    data -= fit
    return data, fit

def get_bode_diagram(bode_object):
        
    signal = []
    detrend = []    
    fft_freq = []
    fft_amp = []
    fft_phase = []
    harmonics = pd.DataFrame()

    for i, k in enumerate(bode_object.bode_records):
        if bode_object.detrend:
            ydata, fit = spline_detrending(bode_object.bode_data[i])
            signal.append(ydata)
            detrend.append(fit)
            
        else:
            signal.append(bode_object.bode_data[i])
            
        if bode_object.windowing == "flat-top":
            signal[i] = signal[i] * wd.flattop(len(signal[i]))
            
        F, A, P = tools.FFT(bode_object.bode_times[i], signal[i], bode_object.padding, bode_object.padding_value)
        
        fft_freq.append(F)
        fft_amp.append(A)
        if bode_object.windowing == "flat-top":
            fft_amp[i] = fft_amp[i] * 4.18
        fft_phase.append(P)
            
        harmonics = pd.concat([harmonics, get_harmonics(bode_object, bode_object.frequency_list[i], fft_freq[i], fft_amp[i], fft_phase[i])])
        bode_object.phase_threshold
        #fft_amp[i] = [amplitude if frequency >= bode_object.frequency_list[i]/2 else 0.0 for frequency, amplitude in zip(fft_freq[i], fft_amp[i])]
        fft_phase[i] = [phase if amplitude >= max(fft_amp[i])/ bode_object.phase_threshold else 0.0 for amplitude, phase in zip(fft_amp[i], fft_phase[i])]
    
    return signal, detrend, fft_freq, fft_amp, fft_phase, harmonics
        
            
def get_harmonics(bode_object, input_freq, F, A, P):
    index_fund = tools.closest_index(F, input_freq)
    harmonics = {'f_input': input_freq}
    for i in range(bode_object.number_of_harmonics):
        index = index_fund*(i+1)
        search_window = [index - bode_object.pic_search_window, index + bode_object.pic_search_window]
        index_max = np.argmax(A[search_window[0]:search_window[1]]) + search_window[0]
        harmonics[f'A_{i}'] = A[index_max]
        harmonics[f'f_{i}'] = F[index_max]
        harmonics[f'P_{i}'] = P[index_max]

    return pd.DataFrame(harmonics, index=[0])



def plot_record_TF(bode_object, record, color = None, leg = None, fig = None, ax = None, fmt = '0-', line = 0.5, marker = 1, log = False):
    if ax is None:
        fig, ax = plt.subplots(1,3, figsize = (13,4))
    if leg is None:
        leg = record
    if color is None:
        color = "C0"
        
    i = np.where(np.array(bode_object.bode_records) == record)[0][0]

    ax[0].plot(bode_object.bode_times[i], bode_object.bode_data[i], "o-", color = color, markersize=marker, linewidth=line, label = leg)
    ax[1].plot(bode_object.fft_freq[i], bode_object.fft_amp[i], "o-",color = color, markersize=marker, linewidth=line, label = leg)
    ax[2].plot(bode_object.fft_freq[i], bode_object.fft_phase[i], "o",color = color, markersize=marker, linewidth=line, label = leg)
    
    for j in range(bode_object.number_of_harmonics):
        ax[1].plot(np.array(bode_object.harmonics[f'f_{j}'])[i], np.array(bode_object.harmonics[f'A_{j}'])[i], "x", markersize=8, markeredgewidth=3, color = color)
        ax[2].plot(np.array(bode_object.harmonics[f'f_{j}'])[i], np.array(bode_object.harmonics[f'P_{j}'])[i], "x", markersize=8, markeredgewidth=3, color = color)
        
    if log:
        ax[1].set_yscale('log')

    ax[0].set_xlabel("Time (s)", fontsize = 14)
    ax[0].set_ylabel("Fluorescence (r. u.)", fontsize = 12)
    ax[0].set_title("Time-domain signal", fontsize = 14)
    ax[1].set_xlabel("Frequency (Hz)", fontsize = 14)
    ax[1].set_ylabel("Amplitude (r. u.)", fontsize = 14)
    ax[1].set_title("FT - Magnitude", fontsize = 14)
    ax[2].set_xlabel("Frequency (Hz)", fontsize = 14)
    ax[2].set_ylabel("Phase (°)", fontsize = 14)
    ax[2].set_title("FT - Phase", fontsize = 14)
    
    return ax

def plot_record_steps(bode_object, record, color = None, leg = None, fig = None, ax = None, fmt = '0-', line = 0.5, marker = 1, log = False):
    if ax is None:
        fig, ax = plt.subplots(2,2, figsize = (10,10))
    if leg is None:
        leg = record
    if color is None:
        color = "C0"
        
    i = bode_object.bode_records.index(record)

    ax[0, 0].plot(bode_object.bode_times[i], bode_object.bode_data[i], "o-", color = color, markersize=marker, linewidth=line, label = leg)
    ax[0, 0].plot(bode_object.bode_times[i], bode_object.detrend_fit[i], "-", color = "k", linewidth= 0.75)
    ax[0, 1].plot(bode_object.bode_times[i], bode_object.signal[i], "o-", color = color, markersize=marker, linewidth=line, label = leg)
    ax[1, 0].plot(bode_object.fft_freq[i], bode_object.fft_amp[i], "o-",color = color, markersize=marker, linewidth=line, label = leg)
    ax[1, 1].plot(bode_object.fft_freq[i], np.deg2rad(bode_object.fft_phase[i]), "o",color = color, markersize=marker, linewidth=line, label = leg)
    ax[1, 1].set_ylim(-1.1*np.pi, 1.1*np.pi)
    
    for j in range(bode_object.number_of_harmonics):
        ax[1, 0].plot(np.array(bode_object.harmonics[f'f_{j}'])[i], np.array(bode_object.harmonics[f'A_{j}'])[i], "x", markersize=8, markeredgewidth=3, color = color)
        ax[1, 1].plot(np.array(bode_object.harmonics[f'f_{j}'])[i], np.deg2rad(np.array(bode_object.harmonics[f'P_{j}'])[i]), "x", markersize=8, markeredgewidth=3, color = color)
        
    if log:
        ax[1, 0].set_yscale('log')

    ax[0, 0].set_xlabel("Time (s)", fontsize = 14)
    ax[0, 0].set_xlabel("Time (s)", fontsize = 14)
    ax[0, 1].set_ylabel("Fluorescence (r. u.)", fontsize = 12)
    ax[0, 1].set_ylabel("Fluorescence (r. u.)", fontsize = 12)
    ax[0, 0 ].set_title("Raw signal", fontsize = 14)
    ax[0, 1].set_title("Detrended signal", fontsize = 14)
    ax[1, 0].set_xlabel("Frequency (Hz)", fontsize = 14)
    ax[1, 0].set_ylabel("Amplitude (r. u.)", fontsize = 14)
    ax[1, 0].set_title("FT - Magnitude", fontsize = 14)
    ax[1, 1].set_xlabel("Frequency (Hz)", fontsize = 14)
    ax[1, 1].set_ylabel("Phase (rad)", fontsize = 14)
    ax[1, 1].set_title("FT - Phase", fontsize = 14)
    
    return ax

def compare_bode(frequency_list, manips, frequency_to_plot = None, min = 0.5, max = 1.5, autoscale = True, leg = None, figsize = (8,6), log = False):
    
    if frequency_to_plot is None:
        frequency_to_plot = frequency_list
    
    if leg is None:
        leg = [manip.name for manip in manips]
    
    for i, k in enumerate(frequency_to_plot):
        fig , ax = plt.subplots(2,2, figsize = figsize)
        if k < 1:
            fig_title = f"P = {1/frequency_to_plot[i]:n} s"
        elif k == 1:
            fig_title = f"P = {1/frequency_to_plot[i]:n} s, F = {frequency_list[i]} Hz "
        elif k > 1:
            fig_title = f"F = {frequency_to_plot[i]} Hz "
        
        fig.suptitle(fig_title, fontsize = 16)
        
        for j, manip in enumerate(manips):
            ax = plot_record_steps(manip, manip.bode_records[frequency_list.index(k)], fig = fig, ax = ax, leg = leg[j], color = f"C{j}", log = log)

                                              

        if autoscale:
            xlim = ax[1,0].get_xlim()
            ax[1,0].set_xlim([k*min, k*max])
            tools.autoscale_y(ax[1,0])
            ax[1,0].set_xlim(0, k*max)
            ax[1,1].set_xlim(0, k*max)
            
        ax[1,0].legend()
                
        fig.tight_layout()
         
        fig.savefig(f"{manips[-1].fig_folder}/{fig_title}_compare.png")
        
        #return fig, ax


def spectral_content_fit(bode_object):
        nb_of_harmonics = 1
        harmonics_fit = pd.DataFrame()
        params_fit = []
        
        for i, freq in enumerate(bode_object.frequency_list):
            
            if i == 0:
                popt, _, _, = tools.sinus_fit(bode_object.bode_times[i], bode_object.filtered_data[i], 0, bode_object.bode_times[i][-1], len(bode_object.bode_data[i]), [100, freq, 0, 0], freq = freq)
            else:
                popt, _, _, = tools.sinus_fit(bode_object.bode_times[i], bode_object.filtered_data[i], 0, bode_object.bode_times[i][-1], len(bode_object.bode_data[i]), [1000, freq, popt[2], 0], freq = freq)
            
            params_fit.append(popt)

            aux_harmonics = {'f_input': freq}
            for j in range(nb_of_harmonics):
                aux_harmonics[f'A_{j}'] = popt[0]
                aux_harmonics[f'P_{j}'] = popt[2]
                
            harmonics_fit = pd.concat([harmonics_fit, pd.DataFrame(aux_harmonics, index=[i])], axis=0)
        

        return harmonics_fit, params_fit