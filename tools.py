import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
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
    freq = np.fft.fftfreq(len(Time), (Time[1] - Time[0]))
    F = freq[1:int(len(freq)/2)]
    ft = np.fft.fft(Signal)
    A = np.abs(ft[1:int(len(freq)/2)])
    P = np.angle(ft[1:int(len(freq)/2)])
    P = P * 180 / np.pi
    
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
    cumsum = np.cumsum(signal, dtype=float)
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
    
def plot_model(ax,  model, freq, amp, sigma = None, p0 =  None, line = 2.5, color = None, label = True, alpha = 0.2):
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

    return ax

def compare_bode(frequency_list, manips, frequency_to_plot = None, min = 0.5, max = 1.5, autoscale = True, leg = None, figsize = (10,5)):
    if frequency_to_plot is None:
        frequency_to_plot = frequency_list
    
    if leg is None:
        leg = [manip.name for manip in manips]
    
    for i, k in enumerate(frequency_to_plot):
        fig , ax = plt.subplots(1,2, figsize = figsize)
        if k < 1:
            fig_title = f"P = {1/frequency_to_plot[i]:n} s"
        elif k == 1:
            fig_title = f"P = {1/frequency_to_plot[i]:n} s, F = {frequency_list[i]} Hz "
        elif k > 1:
            fig_title = f"F = {frequency_to_plot[i]} Hz "
        
        fig.suptitle(fig_title, fontsize = 16)
        
        for j, manip in enumerate(manips):
            ax = manip.plot_record_TF(manip.bode_records[frequency_list.index(k)], fig = fig, ax = ax, leg = leg[j], color = f"C{j}")
                                              

        if autoscale:
            xlim = ax[1].get_xlim()
            ax[1].set_xlim([k*min, k*max])
            autoscale_y(ax[1])
            ax[1].set_xlim(0, k*max)
            
        ax[1].legend()
                
        #fig.tight_layout()
         
        #fig.savefig(f"{manips[-1].fig_folder}/{fig_title}_compare.png")
        
        return fig, ax
    
def merge_dict(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
                dict_3[key] = [value , dict_1[key]]
    return dict_3