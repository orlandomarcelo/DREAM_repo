import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import math
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import approx_fprime
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
    def exp_decay(x, A, B, C):
        return A * (1- np.exp(-((x)/B))) + C
    popt, pcov = curve_fit(exp_decay, xdata, ydata, p0 = p0)
    xfit = np.linspace(start, stop, num)
    yfit = exp_decay(xfit, popt[0], popt[1], popt[2])
    return popt, xfit, yfit
    
def lin_fit(xdata, ydata, start, stop, num):
    def lin(x, A, B):
        return A * x + B
    popt, pcov = curve_fit(lin, xdata, ydata)
    xfit = np.linspace(start, stop, num)
    yfit = lin(xfit, popt[0], popt[1])
    y_pred = lin(np.array(xdata), popt[0], popt[1])
    residuals = ydata - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt, xfit, yfit, r_squared

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

def sinus_fit(xdata, ydata, start, stop, num, p0 = None, freq = None):
    def sinus(x, A, B, C, D):
        return A * np.sin(2* np.pi*(B * x + C)) + D
    if freq is None:
        bounds = ([0, -np.inf, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
    else:
        bounds = ([0, freq*0.99, -np.pi, -np.inf], [np.inf, freq*1.01, np.pi, np.inf])

    popt, pcov = curve_fit(sinus, xdata, ydata, p0=p0, bounds=bounds, maxfev=10000000)
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

def sec_ord_fit(xdata, ydata, start, stop, num, p0 = None, sigma = None):
    popt, pcov = curve_fit(mf.sec_ord_transfer, xdata, ydata, p0 = p0, sigma =sigma)
    xfit = np.linspace(start, stop, num)
    yfit = mf.sec_ord_transfer(xfit, popt[0], popt[1], popt[2])
    return popt, pcov, xfit, yfit


def gradient_at_point(f, point, *args, **kwargs):
    def wrapped_function(params):
        combined_args = tuple(params) + args
        return f(*combined_args, **kwargs)
    
    gradient = approx_fprime(point, wrapped_function, epsilon=np.sqrt(np.finfo(float).eps))
    return gradient


def my_err_vec_old(x, popt, pcov, funct):
    point = [x] + list(popt)
    gradvec=gradient_at_point(funct, point)
    gradvec = gradvec[1:]
    norm=np.matmul(gradvec.T,np.matmul(pcov,gradvec))
    lambd=np.sqrt(1/norm)
    return np.matmul(pcov,lambd*gradvec)

def my_err(x_vect, popt, pcov, funct):
    error_vec = []
    for x in x_vect:
        point = [x] + list(popt)
        gradvec=gradient_at_point(funct, point)
        gradvec = gradvec[1:]
        variance =np.matmul(gradvec.T,np.matmul(pcov,gradvec))
        error_vec.append(np.sqrt(variance))
    return np.asarray(error_vec)


def FFT(Time, Signal, pad = False, length = None):
    if pad == True:
        Time = zero_padding(Time, length)
        Signal = zero_padding(Signal, length)
    F = np.fft.rfftfreq(len(Time), (Time[1] - Time[0]))
    ft = np.fft.rfft(Signal)
    A = 2*np.abs(ft)/len(Time)
    P = np.angle(ft, deg=True)
    
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

    #ax.legend(fontsize = legendfontsize)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', width=2)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(labelsize)
        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(labelsize)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    return ax

def bode_plot_axes_phase(ax):
    labelsize = 15
    legendfontsize = 10
    ax.set_xscale('log')

    ax.set_xlabel("Frequency (Hz)", fontsize = labelsize)
    ax.set_ylabel("Phase (π rad)", fontsize = labelsize)
    ax.grid(which = "both", alpha = 0.4, linewidth = 0.5)

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

def poster_axes(ax, title, xlabel, ylabel, titlesize = 15, labelsize = 15, legendfontsize = 10, legend = True):

    if legend:
        ax.legend(fontsize = legendfontsize)
    ax.set_xlabel(xlabel, fontsize = labelsize)
    ax.set_ylabel(ylabel, fontsize = labelsize)
    ax.set_title(title, fontsize = titlesize)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(labelsize)
        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(labelsize)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    return ax
        
    

def moving_average(signal, window_size):
    cumsum = np.cumsum(signal)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def TF_plot_axes(ax):
    labelsize = 15
    legendfontsize = 10
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Frequency (Hz)", fontsize = labelsize)
    ax.set_ylabel("Magnitude (a.u.)", fontsize = labelsize)
    ax.grid(which = "both", alpha = 0.4, linewidth = 0.5)

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

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)
    
def plot_model(ax,  model, freq, amp, sigma = None, p0 =  None, line = 2.5, color = None, label = True, alpha = 0.2, Return_params = False):
    orange = [250/255, 116/255, 79/255]
    green = [7/255, 171/255, 152/255]
    blue = [24/255, 47/255, 74/255]
   
    if model == "RC":
        if p0 is None:
            p0 = [10, 0.1]
        if color is None:
            color = orange
        popt, pcov, ffit, afit = RC_transf_fit(freq, amp, 0.007, 130, 1000, p0 =  p0, sigma = sigma)
        err = my_err(ffit, popt, pcov, mf.RC_transfer)

        
    if model == "RLC":
        if p0 is None:
            p0 = [ 1e+10, -3e-06,  5e+05]
        if color is None:
            color = green
        popt, pcov, ffit, afit = RLC_transf_fit(freq, amp, 0.007, 130, 1000, p0 =  p0, sigma = sigma)
        try:
            err = my_err(ffit, popt, pcov, mf.RLC_transfer)
        except: pass
        
    if model == "sec_ord":
        if p0 is None:
            p0 = [ 20, 1,  0.1]
        if color is None:
            color = green
            
        popt, pcov, ffit, afit = sec_ord_fit(freq, amp, 0.007, 130, 1000, p0 =  p0, sigma = sigma)
        
        err = my_err(ffit, popt, pcov, mf.sec_ord_transfer)
    
    if label: 
        label = f"{model} model"
    else: label = None
    
    ax.plot(ffit, afit, linewidth=line, color = color, label = label)
    ax.fill_between(np.linspace(0.007, 130, 1000), afit - 1.94*err, afit + 1.94*err, alpha=alpha, color = color)
    
    if Return_params:
        return ax, popt
    else:
        return ax

    
def merge_dict(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
                dict_3[key] = [value , dict_1[key]]
    return dict_3

########## Filtering functions ##########
    
from scipy.signal import lfilter, butter

def high_pass_filter(signal, cutoff_frequency, sample_rate, order=5):    
    # Design a Butterworth high-pass filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def band_pass_filter(signal, lowcut, highcut, sample_rate, order=5):
    # Design a Butterworth band-pass filter
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = min(highcut / nyquist , 0.99)
    b, a = butter(order, [low, high], btype='band', analog=False)

    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def low_pass_filter(signal, cutoff_frequency, sample_rate, order=5):
    # Design a Butterworth low-pass filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = min(cutoff_frequency / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def subplot_grid(num_subplots):
    # Calculate the number of rows and columns for the subplot grid
    rows = int(math.sqrt(num_subplots))
    cols = int(math.ceil(num_subplots / rows))

    # Adjust the number of rows and columns to minimize empty subplots
    while rows * cols < num_subplots:
        rows += 1

    # Create the subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))

    # Flatten the 2D array of subplots to a 1D array
    axes = axes.flatten()

    # Remove empty subplots
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])

    return fig, axes