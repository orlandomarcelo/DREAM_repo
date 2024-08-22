####### Functions for the skeleton model v1 #######

# Import packages
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.integrate import odeint
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import tools



########## Function to define the model ##########

def model(y, t, I, parameters):
    # Unpack state variables (degree of reduction)
    X_PQ_red = y[0]
    X_PC_red = y[1]
    
    # Unpack parameters
    sigma_PSII = parameters["sigma_PSII"]
    sigma_PSI = parameters["sigma_PSI"]
    k_p = parameters["k_p"]
    k_fh = parameters["k_fh"]
    k_b6f = parameters["k_b6f"]
    k_PSI = parameters["k_PSI"]
    PQ_tot = parameters["PQ_tot"]
    PC_tot = parameters["PC_tot"]
    
    # Calculate concentrations from degrees of reduction
    C_PQ = PQ_tot * (1 - X_PQ_red)
    C_PQH2 = PQ_tot * X_PQ_red
    C_PC_plus = PC_tot * (1 - X_PC_red)
    C_PC = PC_tot * X_PC_red
    
    # Calculate fluxes using concentrations
    J_PSII = (I * sigma_PSII * k_p * C_PQ) / (k_p * C_PQ + k_fh)
    J_b6f = k_b6f * C_PC_plus * C_PQH2
    J_PSI = k_PSI * I * sigma_PSI * C_PC
    
    # Calculate derivatives of the degree of reduction
    dX_PQ_red_dt = (J_PSII - J_b6f) / PQ_tot
    dX_PC_red_dt = (2 * J_b6f - J_PSI) / PC_tot
    
    return [dX_PQ_red_dt, dX_PC_red_dt]


########## Function to obtain the steady state light curve ##########

def light_curve(min_light, max_light, light_steps, parameters):
    light_range = np.linspace(min_light, max_light, light_steps)[::-1]

    X_PQ_red_steady = np.zeros(light_steps)
    X_PC_red_steady = np.zeros(light_steps)

    ### initial guess for the max light ###
    X_PQ_red_guess = 1
    X_PC_red_guess = 0

    for i, light in enumerate(light_range):
        if i == 0:
            y0 = [X_PQ_red_guess, X_PC_red_guess]
        else:
            y0 = [X_PQ_red_steady[i-1], X_PC_red_steady[i-1]]

        sol = root(model, y0, args=(0, light, parameters))
        X_PQ_red_steady[i] = sol.x[0]
        X_PC_red_steady[i] = sol.x[1]

    return light_range, X_PQ_red_steady, X_PC_red_steady

########## Function to obtain the dynamic response to a step change in light ##########

def response_step(light_1, light_2, time_light, t_start, t_end, n_points, parameters):
    time = np.linspace(t_start, t_end, n_points)
    light = np.zeros(n_points)
    light[time < time_light] = light_1
    light[time >= time_light] = light_2

    X_PQ_red = np.zeros(n_points)
    X_PC_red = np.zeros(n_points)
    
    ### compute the initial conditions from steady state at light 1 ###
    steady_state_sol = root(model, [0.5, 0.5], args=(0, light_1, parameters))

    X_PQ_red[0] = steady_state_sol.x[0]
    X_PC_red[0] = steady_state_sol.x[1]
    
    ### integrate the model one time step at a time ###

    for i in range(1, n_points):
        time_interval = [time[i-1], time[i]]
        y = odeint(model, [X_PQ_red[i-1], X_PC_red[i-1]], time_interval, args=(light[i], parameters))
        X_PQ_red[i] = y[-1, 0]
        X_PC_red[i] = y[-1, 1]
        
    return time, light, X_PQ_red, X_PC_red


########## Function to obtain the dynamic response to a sinusoidal light ##########

def response_sinusoid(freq, nb_periods, points_per_period, offset, amplitude, parameters):
    time = np.linspace(0, nb_periods/freq, nb_periods * points_per_period)
    light = offset + amplitude * np.sin(2 * np.pi * freq * time)

    X_PQ_red = np.zeros(nb_periods * points_per_period)
    X_PC_red = np.zeros(nb_periods * points_per_period)
    
    ### compute the initial conditions from steady state at light 1 ###
    steady_state_sol = root(model, [0.5, 0.5], args=(0, offset, parameters))

    X_PQ_red[0] = steady_state_sol.x[0]
    X_PC_red[0] = steady_state_sol.x[1]
    
    ### integrate the model one time step at a time ###

    for i in range(1, nb_periods * points_per_period):
        time_interval = [time[i-1], time[i]]
        y = odeint(model, [X_PQ_red[i-1], X_PC_red[i-1]], time_interval, args=(light[i], parameters))
        X_PQ_red[i] = y[-1, 0]
        X_PC_red[i] = y[-1, 1]
        
    return time, light, X_PQ_red, X_PC_red


########## Functions to obtain the bode plot ##########

def get_harmonics(input_freq, F, A, P, pic_search_window):
    index_fund = tools.closest_index(F, input_freq)
    harmonics = {'f_input': input_freq}
    for i in range(5):
        index = index_fund*(i+1)
        search_window = [index - pic_search_window, index + pic_search_window]
        index_max = np.argmax(A[search_window[0]:search_window[1]]) + search_window[0]
        harmonics[f'A_{i}'] = A[index_max]
        harmonics[f'f_{i}'] = F[index_max]
        harmonics[f'P_{i}'] = P[index_max]

    return pd.DataFrame(harmonics, index=[0])

def simulate_bode_plot(freq_min, freq_max, nb_freqs, nb_periods, points_per_period, period_start_fft, offset, amplitude, parameters):
    
    # Create a dictionary to store the results
    X_PQ_red = np.zeros((nb_freqs, nb_periods * points_per_period))
    X_PC_red = np.zeros((nb_freqs, nb_periods * points_per_period))
    times = np.zeros((nb_freqs, nb_periods * points_per_period))

    # Run the dynamic response for each frequency
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), nb_freqs)

    harmonics_PQ = pd.DataFrame()
    harmonics_PC = pd.DataFrame()

    for i in range(nb_freqs):
        freq = frequencies[i]
        times[i], _, X_PQ_red[i], X_PC_red[i] = response_sinusoid(freq, nb_periods, points_per_period, offset, amplitude, parameters)
        F, A, P = tools.FFT(times[i][int(points_per_period*period_start_fft):], X_PQ_red[i][int(points_per_period*period_start_fft):])
        harmonics_PQ = pd.concat([harmonics_PQ, get_harmonics(freq, F, A, P, 5)])
        
        F, A, P = tools.FFT(times[i][int(points_per_period*period_start_fft):], X_PC_red[i][int(points_per_period*period_start_fft):])
        harmonics_PC = pd.concat([harmonics_PC, get_harmonics(freq, F, A, P, 5)])

    return harmonics_PQ, harmonics_PC, frequencies, times, X_PQ_red, X_PC_red 


########## Function to run the sensitivity of the bode plot in parallel ##########

def run_simulation_with_params(i, parameters, parameter_to_vary, parameter_values, freq_min, freq_max, nb_freqs, nb_periods, points_per_period, period_start_fft, offset, amplitude):
    parameters[parameter_to_vary] = parameter_values[i]
    harmonics_PQ, harmonics_PC, frequencies, times, X_PQ_red, X_PC_red = simulate_bode_plot(freq_min, freq_max, nb_freqs, nb_periods, points_per_period, period_start_fft, offset, amplitude, parameters)
    return harmonics_PQ, harmonics_PC, X_PQ_red, X_PC_red, frequencies, times