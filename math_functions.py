import pandas as pd
import numpy as np

def exp_decay(x, A, B):
    return A * (1- np.exp(-((x)/B)))
    
def lin(x, A, B):
    return A * x + B
    
def exp_saturation(x, A, B):
    return A * (1 - np.exp(-(x/B)))

def exp_sat_overshoot(x, A, B, C, D):
    return A * (1 - np.exp(-(x/B))) + C * np.exp(-x/D)

def sigmoid(x, A, B, C, D):
    return A / (1 + np.exp(-B * (x - C))) + D

def sinusoid(x, A, B, C, D):
    return A * np.sin(2* np.pi*(B * x + C)) + D


