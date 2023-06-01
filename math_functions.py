import pandas as pd
import numpy as np

def exp_decay(x, A, B):
    return A * (1- np.exp(-((x)/B)))
    
def lin(x, A, B):
    return A * x + B
    
def exp_saturation(x, A, B, C):
    return A * (1 - np.exp(-(x/B))) + C

def exp_sat_overshoot(x, A, B, C, D):
    return A * (1 - np.exp(-(x/B))) + C * np.exp(-x/D)

def sigmoid(x, A, B, C, D):
    return A / (1 + np.exp(-B * (x - C))) + D

def sinusoid(x, A, B, C, D):
    return A * np.sin(2* np.pi*(B * x + C)) + D

def RC_transfer(w, R, C):
    return R / np.sqrt(1 + np.square(w * R * C))

def RLC_transfer(w, R, L, C):
    return w * C / np.sqrt(np.square(1 - w * w * L * C) - np.square((w * L)/R))

def second_order(w, K, w_0, csi):
    return  K / np.sqrt(np.square(1 - np.square(w/w_0)) + np.square((2* csi * (w/w_0))))





