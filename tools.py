import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math_functions as mf

def clean_spaces(vector):
    clean_vector = []
    for k in vector:
        try:
            clean_vector.append(float(k.strip()))
        except ValueError:
            clean_vector.append(float('nan'))
    return clean_vector

def exp_decay_fit(xdata, ydata, start, stop, num, p0 = None):
    def exp_decay(x, A, B):
        return A * (1- np.exp(-((x)/B)))
    popt, pcov = curve_fit(exp_decay, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = exp_decay(xfit, popt[0], popt[1])
    return popt, xfit, yfit
    
def lin_fit(xdata, ydata, start, stop, num):
    def lin(x, A, B):
        return A * x + B
    popt, pcov = curve_fit(lin, xdata, ydata)
    xfit = np.linspace(start, stop, num)
    yfit = lin(xfit, popt[0], popt[1])
    return popt, xfit, yfit

def Ek_fit(xdata, ydata, start, stop, num, p0 = None):
    def Ek(x, A, B):
        return A * (1 - np.exp(-(x/B)))
    popt, pcov = curve_fit(Ek, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = Ek(xfit, popt[0], popt[1])
    return popt, pcov, xfit, yfit

def sat_overshoot_fit(xdata, ydata, start, stop, num, p0 = None):
    def sat_overshoot(x, A, B, C, D):
        return A * (1 - np.exp(-(x/B))) + C * np.exp(-x/D)
    popt, pcov = curve_fit(sat_overshoot, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = sat_overshoot(xfit, popt[0], popt[1], popt[2], popt[3])
    return popt, xfit, yfit

def sigmoid_fit(xdata, ydata, start, stop, num):
    def sigmoid(x, A, B, C, D):
        return A / (1 + np.exp(-B * (x - C))) + D

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    xfit = np.linspace(start, stop, num)
    yfit = sigmoid(xfit, popt[0], popt[1], popt[2], popt[3])
    return popt, xfit, yfit

def sinus_fit(xdata, ydata, start, stop, num, p0 = None):
    def sinus(x, A, B, C, D):
        return A * np.sin(2* np.pi*(B * x + C)) + D

    popt, pcov = curve_fit(sinus, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = sinus(xfit, popt[0], popt[1], popt[2], popt[3])
    return popt, xfit, yfit

def RLC_transf_fit(xdata, ydata, start, stop, num, p0 = None, sigma = None):
    popt, pcov = curve_fit(mf.RLC_transfer, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = mf.RLC_transfer(xfit, popt[0], popt[1], popt[2])
    return popt, pcov, xfit, yfit

def RC_transf_fit(xdata, ydata, start, stop, num, p0 = None, sigma = None):
    popt, pcov = curve_fit(mf.RC_transfer, xdata, ydata, p0 = p0, sigma = sigma)
    xfit = np.linspace(start, stop, num)
    yfit = mf.RC_transfer(xfit, popt[0], popt[1])
    return popt, pcov, xfit, yfit

def sec_ord_fit(xdata, ydata, start, stop, num, p0 = None):
    popt, pcov = curve_fit(mf.second_order, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = mf.RC_transfer(xfit, popt[0], popt[1])
    return popt, xfit, yfit

def my_err_vec(x, pcov):
    gradvec=np.array([x,1])
    norm=np.matmul(gradvec.T,np.matmul(pcov,gradvec))
    lambd=np.sqrt(1/norm)
    return np.matmul(pcov,lambd*gradvec)


def FFT(Time, Signal, pad = False, length = None):
    if pad == True:
        Time = zero_padding(Time, length)
        Signal = zero_padding(Signal, length)
    freq = np.fft.fftfreq(len(Time), (Time[1] - Time[0]))
    F = freq[1:int(len(freq)/2)]
    ft = np.fft.fft(Signal)
    A = np.abs(ft[1:int(len(freq)/2)])
    P = np.angle(ft[1:int(len(freq)/2)])
    
    return F, A, P

def median_filter(data, window_size):
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        window = data[max(0, i-window_size):min(len(data), i+window_size+1)]
        filtered_data[i] = np.median(window)
    return filtered_data


def get_spectrum(Time, Signal, threshold = 100):
    F, A, P = FFT(Time, Signal)
    peaks, _ = find_peaks(A, threshold= max(A)/threshold)
    P = P * 180 / np.pi
    P = P - P[0]
    P = np.where(P < -180, P + 360, P)
    P = np.where(P > 180, P - 360, P)
    
    return F[peaks], A[peaks], P[peaks]

def get_bode_diagram(Freq_list, Time_list, Signal_list, threshold = 100):
    Freq = []
    Amp = []
    Phase = []

    for i in range(len(Freq_list)):
        F, A, P = get_spectrum(Time_list[i], Signal_list[i], threshold)
        Freq.append(np.asarray(F))
        Amp.append(np.asarray(A))
        Phase.append(np.asarray(P))
   
    
    return Freq, Amp, Phase

def create_record_list(input_str):
    nums = []
    for item in input_str.split(','):
        if '-' in item:
            start, end = item.split('-')
            nums.extend(range(int(start), int(end) + 1))
        else:
            nums.append(int(item))

    string_numbers = ["E" + str(num) for num in nums]

    return string_numbers



def zero_padding(data, pad_lenght = None):
   N = len(data)
   if pad_lenght == None:
    pad_lenght = int(2**np.ceil(np.log2(N))) #computes the closest power of 2 bigger than N
   zp_data = np.pad(data, (0, pad_lenght - N), 'constant', constant_values = 0)

   return zp_data


def closest_index(vector, y):
    vector = np.array(vector)
    diff = np.abs(vector - y)
    closest_index = np.argmin(diff)
    
    return closest_index

def bode_plot_axes(ax):
    labelsize = 15
    legendfontsize = 10
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Frequency (Hz)", fontsize = labelsize)
    ax.set_ylabel("Magnitude (a.u.)", fontsize = labelsize)
    ax.grid(which = "both", alpha = 0.4, linewidth = 0.5)
    ax.set_ylim(2e3, 3e5)

    ax.legend(fontsize = legendfontsize)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', width=2)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(labelsize)
        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(labelsize)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    return ax

def moving_average(signal, window_size):
    cumsum = np.cumsum(signal, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

    