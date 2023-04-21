import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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


def exp_decay_fit(xdata, ydata, start, stop, num):
    def exp_decay(x, A, B, C):
        return A * np.exp(-(x/B)) + C
    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    xfit = np.linspace(start, stop, num)
    yfit = exp_decay(xfit, popt[0], popt[1], popt[2])
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
    return popt, xfit, yfit

def sigmoid_fit(xdata, ydata, start, stop, num):
    def sigmoid(x, A, B, C, D):
        return A / (1 + np.exp(-B * (x - C))) + D

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    xfit = np.linspace(start, stop, num)
    yfit = sigmoid(xfit, popt[0], popt[1], popt[2])
    return popt, xfit, yfit


def FFT(Time, Signal):

    fs = 1 / (Time[1] - Time[0])
    freq = np.fft.fftfreq(len(Time), 1/fs) 
    F = freq[1:int(len(freq)/2)]
    ft = np.fft.fft(Signal)
    A = np.abs(ft[1:int(len(freq)/2)])
    P = np.angle(ft[1:int(len(freq)/2)])
    
    return F, A, P

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
