    
import NIControl.AcquisitionClass
import NIControl.DronpaClass
import numpy as np
import alienlab
import matplotlib.pyplot as plt
import time
from alienlab.init_logger import logger
from scipy import optimize
import alienlab.regression_func
from config_DAQ import *




class BodeDiagram(NIControl.AcquisitionClass.Acquisition):

    def __init__(self):
        super().__init__()
    
    def bode_diagram(self, frequencies, signal, sleep_time):
        N = len(frequencies)
        radius = np.zeros((N, self.num_channels_analog() + self.num_channels_digital()))
        phase = np.zeros((N, self.num_channels_analog() + self.num_channels_digital()))
        sin_lo = np.zeros((N, self.num_channels_analog() + self.num_channels_digital()))
        cos_lo = np.zeros((N, self.num_channels_analog() + self.num_channels_digital()))
        #radius2 = np.zeros((N, self.num_channels))
        #phase2 = np.zeros((N, self.num_channels))
        all_outputs = []
        all_times = []

        self.cop.title = 'Bode plot'
        self.cop.xlabel = 'Frequency'
        self.cop.ylabel = 'Amplitude'
        self.cop.y2label = 'Phase'
        self.cop.label_list = ['Intensity', 'fluo', 'blue', 'purple'] +['Intensity', 'fluo', 'blue']
        self.cop.label2_list = ['Phase_Intensity', 'Phase_fluo', 'Phase_blue'] +['Intensity', 'fluo', 'blue']
        self.cop.ylog = 'semilogx'
        self.cop.y2log = 'semilogx'


        for i, freq in enumerate(frequencies):
            self.excitation_frequency = freq
            self.update_rates() 
            self.generator_analog_init(signal)
            self.detector_init()
            self.trigger_frequency = freq
            self.trigger_init()

            self.task_generator_analog.start()
            self.detector_start()
            self.task_trigger.start()

            output, time_array = self.read_detector()

            phase_shift = self.get_phase_shift(output[0], time_array, freq)
            sin_lock, cos_lock, radius_lock, phase_lock  = self.lock_in(output, time_array, freq, phase_shift = phase_shift)
            #sin_lock2, cos_lock2, radius_lock2, phase_lock2  = self.lock_in(output, time_array, 2*freq)

            radius[i] = radius_lock
            phase[i] = phase_lock
            sin_lo[i] = sin_lock.mean(axis = 1)
            cos_lo[i] = cos_lock.mean(axis = 1)

            #radius2[i] = radius_lock2
            #phase2[i] = phase_lock2
            all_outputs.append(output)
            all_times.append(time_array)
            self.end_tasks()
            self.set_level_LED(signal.mean(axis = 1), None, sleep_time)

            
        print(radius.shape, sin_lo.shape, cos_lo.shape)

        categories =  ['Intensity', 'fluo', 'blue', 'purple']
        for j in range(2):
            self.cop.label_list = ['amplitude', 'sin', 'cos']
            self.cop.label2_list = ['phase']

            self.cop.xval = frequencies
            self.cop.yval = [radius[:,j], sin_lo[:, j], cos_lo[:,j]]
            self.cop.x2val = frequencies
            self.cop.y2val = phase[:,j]# - phase[:,1]
            print(j)
            f = self.cop.coplotting()
            #f = p.plotting(p.xval, p.yval)
            self.cop.save_name = "bode_plot_%s"%categories[j]
            self.cop.saving(f)
            plt.close('all')
            """
            self.cop.label_list = ['amplitude', 'sin', 'cos']
            self.cop.label2_list = ['cos']

            self.cop.xval = frequencies
            self.cop.yval = [radius[:,j]]
            self.cop.x2val = frequencies
            self.cop.y2val = cos_lo[:,j]# - phase[:,1]

            f = self.cop.coplotting()
            #f = p.plotting(p.xval, p.yval)
            self.cop.save_name = "bode_plot_%s_amplitude"%categories[j]
            self.cop.saving(f)
            plt.close('all')
            f = self.cop.coplotting()
            #f = p.plotting(p.xval, p.yval)
            self.cop.save_name = "bode_plot_%s"%categories[j]
            self.cop.saving(f)
            """
            self.cop.label_list = ['amplitude', 'sin', 'cos']
            self.cop.label2_list = ['phase']
            #if j in [0, 2]:
            #    self.cop.xval = frequencies
            #    self.cop.yval = radius[:,j]/radius[:,0]
            #    self.cop.x2val = frequencies
            #    self.cop.y2val = phase[:,j]# - phase[:,1]

            #    f = self.cop.coplotting()
            #    #f = p.plotting(p.xval, p.yval)
            #    self.cop.save_name = "bode_plot_%s_amplitude_corrected"%categories[j]
            #    self.cop.saving(f)
            
        all_outputs_2D = np.concatenate(all_outputs, axis = 1 )
        all_outputs_3D = np.array(all_outputs)
        np.save(self.save_folder + "/bode_3D_output.npy", all_outputs_3D)
        np.save(self.save_folder + "/bode_3D_times.npy", all_times)
        np.savetxt(self.save_folder + "/bode_full_response.csv", all_outputs_2D.T, delimiter = ',')
        np.savetxt("/bode_full_response.csv", all_outputs_2D.T, delimiter = ',')
        self.copy_py_file("Notebooks/bode_full_trace.ipynb")
        self.set_level_LED(0*signal.mean(axis = 1), None, sleep_time)       
        return radius, phase, all_outputs_2D, sin_lo, cos_lo, all_outputs_3D
