import NIControl.AcquisitionClass
import numpy as np
import time
import pandas as pd
import logging
from scipy.interpolate import InterpolatedUnivariateSpline
from mvgavg import mvgavg

import nidaqmx
from nidaqmx.constants import AcquisitionType 
import nidaqmx.stream_writers
import nidaqmx.stream_readers
from nidaqmx.constants import Edge

import alienlab
import alienlab.plot
import matplotlib.pyplot as plt

from alienlab.init_logger import logger
logger.name = 'Routines'

class Routines(NIControl.AcquisitionClass.Acquisition):
    def __init__(self):
        super().__init__()


    def detector_response_routine(self, offset_min, offset_max, amplitude, N_points, color = 'blue'):
        logger.info("Running the detector response routine")
        offset_range = np.linspace(offset_min, offset_max, N_points)

        self.amplitude = amplitude
        intensity_range = offset_range *  0
        fluo_range = offset_range *  0

        full_output = []

        for i, ofst in enumerate(offset_range):
            logger.critical(ofst)
            self.offset = ofst
            signal = self.sine_pattern()
            time.sleep(1)
            self.generator_analog_init(signal)
            self.detector_init()
            self.trigger_init()
            self.task_generator_analog.start()
            self.task_trigger.start()
            self.detector_start()


            blank_output, blank_time = self.read_detector()
            intensity, fluo = [out.mean() for out in blank_output[:2]]
            intensity_range[i]= intensity
            fluo_range[i]= fluo
            full_output.append(blank_output[0])
            self.end_tasks()
            self.shut_down_LED(1)



        self.p.xlabel = 'Offset (V)'
        self.p.title = "Detectors Response"
        self.p.color_list = [(1, 0, 0), (0,0,1), (0, 0, 0)]
        self.p.label_list = ['intensity', 'fluo']
        self.p.xval = offset_range
        self.p.ylabel = 'Amplitude (V)'
        self.p.yval = [intensity_range, fluo_range]
        self.p.majorFormatterx = "%f"
        self.p.majorFormattery = "%f"

        fig = self.p.plotting(self.p.xval, self.p.yval)
        self.p.save_name = "Detector_response_curve_" + color
        self.p.saving(fig)
        self.set_level_LED(0, 0, 1)
        print("LEVEL set to 0")

        return offset_range, intensity_range, fluo_range, full_output

    def absorbance_routine(self):
        logger.info("Running the absorbance routine")
        self.amplitude = 0.5
        self.offset = 1
        self.update_rates()
        signal = self.sine_pattern()

        self.generator_init(signal)
        self.detector_init()
        self.trigger_init()

        self.generator_start()
        self.detector_start()

        input("Blank measurement")
        self.task_trigger.start()
        blank, blank_time = self.read_analog()
        sin_blank, cos_blank, radius_blank, phase_blank = self.lock_in(blank, blank_time, self.excitation_frequency, phase_shift = 0)
        
        self.detector_stop()
        self.task_trigger.stop()

        self.generator_to_zero("Switch to sample")

        self.generator_stop()
        self.generator_close()
        self.generator_init(signal)
        self.generator_start()
        self.detector_start()
        self.task_trigger.start()
        dense, dense_time = self.read_analog()
        sin_dense, cos_dense, radius_dense, phase_dense = self.lock_in(dense, dense_time, self.excitation_frequency, phase_shift = 0)

        self.end_tasks()
        self.set_level_LED(0, 1)

        logger.debug(radius_dense)
        logger.debug(radius_blank)
        A_locked = np.log10(radius_blank[1]/radius_dense[1]) #1 - radius_dense[1] / radius_blank[1]
        A_mean = np.log10(blank[1].mean()/dense[1].mean()) #1 - dense[1].mean()/blank[1].mean()
        logger.debug("absorbance locked = %0.6f, absorbance mean = %0.6f"%(A_locked, A_mean)) #5 is the log level
        #print("radius: ", radius_sample.mean(), radius_blank.mean(), "mean", sample.mean(), blank.mean())

        self.p.xlabel = 'time (s)'
        self.p.ylabel = 'voltage'
        self.p.title = "Absorbance"
        self.p.color_list = [(1, 0, 0), (0,0,1), (0, 0, 0)]
        self.p.label_list = ['blank','sample']
        fig = self.p.plotting(blank_time, [blank[1], dense[1]])
        self.p.save_name = "absorbance_routine"
        self.p.saving(fig)
        
        return A_locked, A_mean


if __name__ == "__main__":


    routines = Routines()


    if True: 
        routines.offset = 2
        routines.amplitude = 2
        routines.excitation_frequency = 10
        routines.num_period = 10
        routines.points_per_period = 10000
        routines.update_rates()
        signal = routines.square_pattern()
        output = routines.quick_test(np.tile(signal, routines.num_channels_analog_gene() + routines.num_channels_digital_gene() ), None)

    if False: 
        routines.generator_channel = "Dev1/ao0"

        offset_min = 0.01
        offset_max = 5
        amplitude = 0.005
        routines.excitation_frequency = 10
        routines.num_period = 10
        routines.points_per_period = 1000
        routines.update_rates()
        offset_range_405, intensity_range_405, photodiode_range_405, fluo_range_405 = routines.detector_response_routine(offset_min, offset_max, amplitude, 20)



        routines.generator_channel = "Dev1/ao1"

        offset_min = 0.01
        offset_max = 5
        amplitude = 0.005
        routines.excitation_frequency = 10
        routines.num_period = 10
        routines.points_per_period = 1000
        routines.update_rates()
        offset_range_480, intensity_range_480, photodiode_range_480, fluo_range_480 = routines.detector_response_routine(offset_min, offset_max, amplitude, 20)
    '''
        input('switch cables')

        routines.generator_channel = "Dev1/ao1"
        offset_min = 0.01
        offset_max = 5
        amplitude = 0.005
        routines.update_rates(excitation_frequency = 10, num_period = 10, points_per_period = 1000)
        offset_range_405_s, intensity_range_405_s, photodiode_range_405_s, fluo_range_405_s = routines.detector_response_routine(offset_min, offset_max, amplitude, 20)



        routines.generator_channel = "Dev1/ao0"

        offset_min = 0.01
        offset_max = 5
        amplitude = 0.005
        routines.update_rates(excitation_frequency = 10, num_period = 10, points_per_period = 1000)
        offset_range_480_s, intensity_range_480_s, photodiode_range_480_s, fluo_range_480_s = routines.detector_response_routine(offset_min, offset_max, amplitude, 20)

        plt.plot(intensity_range_405, label = '405 ao0')
    '''

    if True: 
        routines.generator_channel = "Dev1/ao0"

        routines.offset = 0.5
        routines.amplitude = 0.2
        routines.excitation_frequency = 1
        routines.num_period = 10
        routines.points_per_period = 10000
        routines.update_rates()
        routines.absorbance_routine()

