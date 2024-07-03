import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Define the parameters of the RC circuit
R = 1000 # Resistance in ohms
C = 0.001 # Capacitance in farads

# Define the time interval and step size for the simulation
t_start = 0 # Start time in seconds
t_stop = 10 # Stop time in seconds
dt = 0.01 # Time step size in seconds
t = np.arange(t_start, t_stop, dt)

# Define the initial voltage across the capacitor
v_c0 = 0

# Define the input and output waveforms
v_c = np.zeros_like(t)
v_c[0] = v_c0

# Define the function that simulates the response of the RC circuit
def RC_response(frequency, fig, ax1, plot=True):
    w = 2 * np.pi * frequency # Angular frequency in rad/s
    i_in = np.sin(w*t)
    for i in range(1, len(t)):
        v_c[i] = v_c[i-1] + (i_in[i] / C - v_c[i-1] / (R * C)) * dt

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current (A)', color=color)
    ax1.plot(t, i_in, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # create a secondary y-axis that shares the same x-axis with ax1

    color = 'tab:blue'
    ax2.set_ylabel('Voltage (V)', color=color)
    ax2.plot(t, v_c, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()
        

# Create the interactive slider
frequency_slider = widgets.FloatSlider(
    value=1,
    min=0.01,
    max=10,
    step=0.01,
    description='Frequency (Hz):',
    readout=True,
    readout_format='.2f'
)

# Display the slider and plot the initial waveform
display(frequency_slider)
fig, ax = plt.subplots()
RC_response(frequency_slider.value, fig, ax)

# Define the function that updates the plot when the slider is moved
def on_value_change(change):
    frequency = change['new']
    RC_response(frequency, fig, ax)

# Attach the on_value_change function to the slider's value attribute
frequency_slider.observe(on_value_change, names='value')
