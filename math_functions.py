import pandas as pd
import numpy as np

def exp_decay(x, A, B):
    return A * np.exp(-((x)/B))
    
def lin(x, A, B):
    return A * x + B

def Ek(x, A, B):
        return A * (1 - np.exp(-(x/B)))
    
def exp_saturation(x, A, B, C):
    return A * (1 - np.exp(-(x/B))) + C

def exp_sat_overshoot(x, A, B, C, D):
    return A * (1 - np.exp(-(x/B))) + C * np.exp(-x/D)

def sigmoid(x, A, B, C, D):
    return A / (1 + np.exp(-B * (x - C))) + D

def sinusoid(x, A, B, C, D):
    return A * np.sin(2* np.pi*(B * x + C)) + D

def RC_transfer(freq, R, C):
    return R / np.sqrt(1 + np.square(freq * R * C))

def RLC_transfer(freq, R, L, C):
    return freq * (C) / np.sqrt(np.square(1 - freq * freq* L * C) - np.square((freq* L)/R))

def sec_ord_transfer(freq, K, wn, zeta):
    s = 2j * np.pi * freq
    return np.abs(K / (s**2 + 2 * zeta * wn * s + wn**2))





