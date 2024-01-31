import NIControl.AcquisitionClass
from NIControl.RoutinesClass import Routines

import alienlab
from alienlab.init_logger import logger
from alienlab.regression_func import *

import numpy as np
import json
import matplotlib.pyplot as plt
from mvgavg import mvgavg


import ipdb

from   scipy import optimize
# 1. Simulate some data
# In the real worls we would collect some data, but we can use simulated data
# to make sure that our fitting routine works and recovers the parameters used
# to simulate the data


class DronpaIntensity(NIControl.AcquisitionClass.Acquisition):

    def __init__(self):
        super().__init__()
        video_array = []
    def exp_decay(self, parameters, xdata):
        '''
        Calculate an exponential decay of the form:
        S= a * exp(-xdata/b)
        '''
        A = parameters[0]
        tau = parameters[1]
        y0 = parameters[2]
        return A * np.exp(-xdata/tau) + y0

    def residuals(self, parameters, x_data, y_observed, func):
        '''
        Compute residuals of y_predicted - y_observed
        where:
        y_predicted = func(parameters,x_data)
        '''
        return func(parameters,x_data) - y_observed

    def generate_signal(self, volt, filter_LED):
        #480

        self.excitation_frequency = 0.1 * volt / filter_LED
        self.update_rates()
        self.offset =  volt * 2
        self.amplitude = 0
        self.phase_shift = 0
        signal_480 = self.square_pattern()

        #405
        self.update_rates()
        self.offset = volt
        self.amplitude = volt
        self.phase_shift = np.pi
        signal_405 = self.square_pattern()
    
     
        signal_12 = np.stack((signal_480, signal_405), axis = 0)
        return signal_12

    def get_tau(self, trig, fluo, time_array, volt, threshold_ticks = 1e0, averaging = False, N_avg = 0, plot = True):
        trig_shift = np.roll(trig, 1)
        diff = trig-trig_shift
        tau_list = []
        
        trig_diff = np.abs(diff) > threshold_ticks #PARAMETER HERE

        ticks = np.nonzero(trig_diff)[0].tolist()
        ticks.append(-1)
        first = ticks[0]
        #logger.critical(ticks)    
        for second in ticks[1:]:
            if second - first < (self.points_per_period // 2):  #PARAMETER HERE
                ticks.remove(second)
            else: 
                first = second

        #logger.info(ticks)
        first = ticks[0]
        for second in ticks[1:]:
            fluo_transition = fluo[first:second]
            #logger.critical(fluo_transition)
            time_transition = np.linspace(0, second - first - 1, second - first)
            x0 = [1e5, (second-first)/8, 1]
            OptimizeResult  = optimize.least_squares(self.residuals,  x0, bounds = (-1e9,1e9),
                                                args = (time_transition, fluo_transition, self.exp_decay))
            #logger.critical(OptimizeResult.x)
            parameters_estimated = OptimizeResult.x
            tau = parameters_estimated[1]
           
            #conditions on tau too low or too high for the second, more accurate, fit, because we will fit on a signal that lasts 5*tau
            if tau >  (second-first)//10: #if too high
                tau =  (second-first)//10
            if tau < 3: #if too low, increase it
                tau = 5
            x0 = parameters_estimated #initial guess: parameters from previous fit
            #second fit
            OptimizeResult  = optimize.least_squares(self.residuals,  x0, bounds = (-1e9,1e9),
                                                args = (time_transition[0:int(tau*5)], fluo_transition[0: int(tau*5)], self.exp_decay))
            parameters_estimated = OptimizeResult.x
        
            if plot:
                # Estimate data based on the solution found
                y_data_predicted = self.exp_decay(parameters_estimated, time_transition)

                
                # Plot all together
                self.cop.title = "Voltage = %0.3f"%volt
                self.cop.xlabel = 'time (s)'
                self.cop.ylabel = 'voltage (V)'
                self.cop.y2label = 'residuals (V)'
                self.cop.label_list = ['Real Data','Voltage = %0.3f, tau = %0.2f'%(volt, tau)]
                self.cop.label2_list = ['Residuals']
                if averaging:
                    self.cop.xval = mvgavg(time_transition, N_avg)
                    self.cop.yval = [mvgavg(fluo_transition, N_avg), mvgavg(y_data_predicted, N_avg)]
                    self.cop.x2val = mvgavg(time_transition, N_avg)
                    self.cop.y2val = mvgavg(y_data_predicted - fluo_transition, N_avg)
                else:
                    self.cop.xval = time_transition
                    self.cop.yval = [fluo_transition, y_data_predicted]
                    self.cop.x2val = time_transition
                    self.cop.y2val = y_data_predicted - fluo_transition

                f = self.cop.coplotting()
                self.cop.save_name = 'exponential_fit_%d'%first
                self.cop.saving(f)
    
            # How good are the parameters I estimated?
            #logger.info('Predicted: A, tau, y0 ' + str( parameters_estimated))
            tau_list.append(parameters_estimated[1]/self.sample_rate)
            first = second
            plt.close(f[0])

        tau_1 = tau_list[0::2]
        tau_2 = tau_list[1::2]
        
        return [np.asarray(tau_1), np.asarray(tau_2)]

    def intensity_range(self, voltage, N_points, filter_LED):
        tau_480_tot = {}
        tau_480_405_tot = {}
        
        val_480 = {}
        val_405 = {}

        for i, volt in enumerate(voltage):
            signal_12 = self.generate_signal(voltage[i], filter_LED)
            #print(signal_12)
            #logger.critical(self.generator_digital_channels)
            
            
            output, average_output, time_array, x_time = self.quick_test(signal_12, None)
            print("HERE is the voltage", volt)
            #ipdb.set_trace()
            #logger.critical(self.generator_digital_channels)

            output = np.vstack((output, time_array))
            np.savetxt("measure_times_%d.csv"%self.offset, output.T, delimiter = ',')


            trig = output[3]
            fluo = output[1]

            tau_480, tau_480_405 = self.get_tau(trig, fluo, time_array, voltage[i], threshold_ticks = 0.1)#1e1/filter_LED)
            tau_480_mean = np.mean(tau_480)
            tau_480_405_mean = np.mean(tau_480_405)

            if tau_480_mean < tau_480_405_mean:
                eph = tau_480
                tau_480 = tau_480_405
                tau_480_405 = eph

            tau_480_tot[str(volt)] = {'output': output, 'tau': tau_480, 'tau_mean': tau_480.mean(), 'tau_std': tau_480.std()}
            tau_480_405_tot[str(volt)] = {'output': output, 'tau': tau_480_405, 'tau_mean': tau_480_405.mean(), 'tau_std': tau_480_405.std()}   
        
            logger.critical(self.generator_digital_channels)

        routines = Routines()
        routines.p.save_folder = self.save_folder
        routines.generator_analog_channels = ["ao0"]
        routines.generator_digital_channels = []
        #480
        offset_min = (voltage).min() * 2
        offset_max =  (voltage).max() * 2
        amplitude = 0
        routines.excitation_frequency = 10
        routines.num_period = 10
        routines.points_per_period = 10000
        routines.update_rates()
        logger.critical("channels")
        logger.critical(routines.num_channels_analog_gene())
        offset_range_480, val_480['intensity'], fluo_range_480, full_output = routines.detector_response_routine(offset_min,
                                                                                        offset_max, amplitude, N_points, color = 'blue')

        probe_low_480, val_low_480, fluo_low_480, full_output = routines.detector_response_routine(0,
                                                                                        offset_min, amplitude, 20, color = 'blue')


        routines.generator_analog_channels = ["ao1"]
        routines.generator_digital_channels = []

        #405
        offset_min = (voltage).min()*2
        offset_max =  (voltage).max()*2
        amplitude = 0
        routines.update_rates()
        offset_range_405, val_405['intensity'], fluo_range_405, full_output = routines.detector_response_routine(offset_min, 
                                                                                        offset_max, amplitude, N_points, color = 'purple')

        probe_low_405, val_low_405, fluo_low_405, full_output = routines.detector_response_routine(0, 
                                                                                        offset_min, amplitude, 20, color = 'purple')

        return tau_480_tot, tau_480_405_tot, val_480, val_405, probe_low_480, val_low_480,probe_low_405, val_low_405, offset_range_480, offset_range_405, fluo_low_480, fluo_low_405


    def analyse_results(self, voltage, tau_480_tot, tau_480_405_tot, val_480, val_405):

        sigma_480 = 198
        sigma_405 = 415 #m²/mol
        kdelta = 0.014
        val_480['tau'] = np.asarray([tau_480_tot[key]['tau_mean'] for key in tau_480_tot.keys()])
        val_405['tau'] = np.asarray([tau_480_405_tot[key]['tau_mean'] for key in tau_480_405_tot.keys()])
        index  =  (val_480['tau'] == val_480['tau']) * (val_405['tau'] == val_405['tau'])
        self.p.xlabel = 'LED 405 voltage'
        self.p.ylabel = '1/tau'
        self.p.title = "Dronpa2 intensity"
        self.p.label_list = ['tau_480', 'tau_480_405']
        f = self.p.plotting(voltage[index], [1/val_480['tau'][index], 1/val_405['tau'][index]])
        self.p.save_name = 'tau_curve'
        self.p.saving(f)

        Yreg = {}
        a = {}
        b = {}
        val_480['Yreg'], val_480['a'], val_480['b'], val_480['results_summary'] = regression_affine(val_480['intensity'], 1/val_480['tau'])
        val_405['Yreg'], val_405['a'], val_405['b'], val_405['results_summary'] = regression_affine(val_405['intensity'], 1/val_405['tau'])

        self.p.xlabel = 'voltage on MPPC (V)'
        self.p.ylabel = '1/tau (s-1)'
        self.p.label_list = 'raw 405',  'regression 405'
        self.p.save_name = "Regression_curves_405"
        f = self.p.plotting(val_405['intensity'], [1/val_405['tau'], val_405['Yreg']])
        self.p.saving(f)

        self.p.xlabel = 'voltage on MPPC (V)'
        self.p.ylabel = '1/tau (s-1)'
        self.p.label_list = 'raw 480',  'regression 480'
        self.p.save_name = "Regression_curves_480"
        f = self.p.plotting(val_480['intensity'], [1/val_480['tau'], val_480['Yreg']])
        self.p.saving(f)


        val_480['I_480'] = (1/val_480['tau'] - kdelta)/sigma_480 
        self.p.xlabel = 'MPPC voltage'
        self.p.ylabel = "light intensity 480 nm eins/m²/s"
        self.p.save_name = "light_intensity_480"
        f = self.p.plotting(val_480['intensity'], val_480['I_480'])
        self.p.saving(f)

        val_405['I_405'] = (1/val_405['tau'] - 1/val_480['tau']) / sigma_405
        self.p.xlabel = 'MPPC voltage'
        self.p.ylabel = "light intensity 405 nm eins/m²/s"
        self.p.save_name = "light_intensity_405"
        f = self.p.plotting(val_405['intensity'], val_405['I_405'])
        self.p.saving(f)
        return val_480, val_405

if __name__ == "__main__":


    test = NIControl.DronpaIntensity()
    test.experiment_folder("Dronpa2_intensity")
    test.p.save_folder = test.save_folder
    p = test.p
    test.experiment_folder("Dronpa2_intensity")
    test.cop.save_folder = test.save_folder
    cop = test.cop
    logger.name = "Dronpa2_intensity"
 

    test.generator_analog_channel = "Dev1/ao0:1"
    test.num_channels_analog_gene = 2

    test.excitation_frequency = 1
    test.trig_chan_detector_edge = 'FALLING'



    test.num_period = 10
    test.points_per_period = 10000

    sigma_480 = 198
    sigma_405 = 415

    filter_LED = 1


    N_points = 25
    voltage = np.linspace(0.1, 5, N_points)

    tau_480_tot, tau_480_405_tot, val_480, val_405 = test.intensity_range(voltage, N_points, filter_LED)
    high_480, high_405 = test.analyse_results(voltage, tau_480_tot, tau_480_405_tot, val_480, val_405)


    input('Remove the filter')

    N_points = 15
    voltage = np.linspace(0.04, 0.1, N_points)



    tau_480_tot, tau_480_405_tot, val_480, val_405 = test.intensity_range(voltage, N_points, filter_LED)
    low_480, low_405 = test.analyse_results(voltage, tau_480_tot, tau_480_405_tot, val_480, val_405)

    plt.close('all')
    plt.plot(high_480['intensity'], 1/high_480['tau'])
    plt.plot(low_480['intensity'], 1/low_480['tau'])
    plt.show()

