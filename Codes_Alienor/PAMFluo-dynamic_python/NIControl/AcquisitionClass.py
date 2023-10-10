import numpy as np
import time
import sys 
import os

import pandas as pd
from mvgavg import mvgavg

import nidaqmx
from nidaqmx.constants import AcquisitionType 
import nidaqmx.stream_writers
import nidaqmx.stream_readers
from nidaqmx.constants import Edge

import alienlab
import alienlab.plot
import matplotlib.pyplot as plt
import time
import shutil
import json

from alienlab.init_logger import logger, move_log
import datetime

from alienlab.regression_func import get_func


#intensity, photodiode, fluo, generator, trigger = [out for out in average_output]


class Acquisition:
    """
    """
    
    
    def __init__(self):

        self.all_connected_channels()

        self.trig_chan_generator = "/Dev1/PFI12"
        self.trig_chan_generator_edge = "RISING"
        self.trig_chan_detector = "/Dev1/PFI12"
        self.trig_chan_detector_edge = "RISING"
        self.trig_chan = "Dev1/Ctr0"



        self.excitation_frequency = 10
        self.amplitude = 0.2
        self.offset = 1
        self.phase_shift = 0
        self.num_period = 10
        self.points_per_period = 1000
        self.window = 5
        self.number_of_cycles = 100000

        self.writing_samples = 10000

        self.p = alienlab.plot.PlotFigure()
        self.p.save_folder = "save_figures"
        self.p.extension = ".png"

        self.cop = alienlab.plot.PlotFigure()
        self.cop.save_folder = "save_figures"
        self.cop.extension = ".png"
        
        self.g = alienlab.plot.ShowFigure()
        self.g.save_folder = "save_figures"
        self.g.extension = ".png"

        self.update_rates()
        #self.photodiode_LED1 = light_intensity.Photodiode("../graphs/photodiode_1226_44BQ1E.csv", 480)


        #self.sample_rate_factor = self.num_period * 100
        #self.acq_time = self.num_period/self.excitation_frequency

        #self.writing_rate = self.excitation_frequency * 1000
        #self.writing_samples = self.writing_rate
        #self.sample_rate = self.excitation_frequency * self.sample_rate_factor
        #self.reading_samples = int(self.num_period * self.sample_rate_factor)

        #self.task_generator = nidaqmx.Task()
        #self.task_detector = nidaqmx.Task()

    def all_connected_channels(self):
        self.generator_analog_channels = ["ao0", "ao1"] #"Dev1/ao0:1"
        self.generator_digital_channels = ["port0/line2"] #"Dev1/port0/line2"
        self.detector_analog_channels = ["ai0", "ai1", "ai2", "ai3"] #"Dev1/ai0:3"
        self.detector_digital_channels = ["port0/line5", "port0/line4", "port0/line0"] #"Dev1/port0/line0"


    def num_channels_analog(self):
        return len(self.detector_analog_channels)

    def num_channels_digital(self):
        return len(self.detector_digital_channels)
        
    def num_channels_analog_gene(self):
        return len(self.generator_analog_channels)
        
    def num_channels_digital_gene(self):
        return len(self.generator_digital_channels)

    def add_channels(self, operator, channels_list):
        for i in range(len(channels_list)):
            operator("Dev1/" + channels_list[i])


    def print_prop(self):
        attrs = vars(self)
        logger.debug('\n'.join("%s: %s" % item for item in attrs.items()))

    def update_rates(self):
        self.reading_samples = self.num_period * self.points_per_period
        self.writing_samples = self.num_period * self.points_per_period
        self.acq_time = self.num_period/self.excitation_frequency

        self.writing_rate = self.excitation_frequency * self.writing_samples//self.num_period #self.writing_samples/self.acq_time
        self.sample_rate = self.excitation_frequency * self.points_per_period
        
        #self.averaging_win = 1e6//self.reading_samples
        self.trigger_frequency = self.excitation_frequency
        logger.debug(self.points_per_period)
        logger.debug(self.acq_time)
        logger.debug(self.writing_rate)
        logger.debug(self.excitation_frequency)
        logger.debug(self.sample_rate)
        logger.debug(self.writing_samples)
        logger.debug(self.reading_samples)

    def update_offset(self, requested_intensity):
        self.offset = self.photodiode_LED1.update_level(requested_intensity)

    def update_amplitude(self, requested_intensity):
        self.amplitude = self.photodiode_LED1.update_level(requested_intensity)

    def averaging(self, measured_signal, 
                    time_array, 
                    do_binning = True):

        average_output = mvgavg(measured_signal, self.window, axis = 1, binning = do_binning)
        downscaled_time = time_array[self.window//2 :: self.window] 
        #todo: if binning == False, time array
        return average_output, downscaled_time    

    def lock_in(self, measured_signal, 
                time_array, 
                frequency, phase_shift):
        cos_ref = np.cos(2*np.pi * frequency * time_array - phase_shift)
        cos_ref = np.stack([cos_ref] * measured_signal.shape[0])
        sin_ref = np.sin(2*np.pi * frequency * time_array - phase_shift)
        sin_ref = np.stack([sin_ref] * measured_signal.shape[0])
        cos_lock = 2 * np.multiply(measured_signal, cos_ref)
        sin_lock =  2 * np.multiply(measured_signal, sin_ref)
        radius_lock = np.sqrt(sin_lock.mean(axis = 1)**2 + cos_lock.mean(axis = 1)**2)
        phase_lock = np.arctan(sin_lock.mean(axis = 1)/cos_lock.mean(axis = 1))
        return sin_lock, cos_lock, radius_lock, phase_lock

    def get_phase_shift(self, intensity_signal, 
                time_array, 
                frequency):
        cos_ref = np.cos(2*np.pi * frequency * time_array)
        sin_ref = np.sin(2*np.pi * frequency * time_array)
        cos_lock = 2 * np.multiply(intensity_signal, cos_ref)
        sin_lock =  2 * np.multiply(intensity_signal, sin_ref)
        phase_shift = np.arctan2(sin_lock.mean(), cos_lock.mean())
        logger.critical(phase_shift)
        return phase_shift

    
    def lock_in_generator(self, measured_signal, generator_signal):
        generator_signal = generator_signal / generator_signal.mean()
        generator_signal = generator_signal - 1
        generator_signal = np.stack([generator_signal] * measured_signal.shape[0])
        cos_lock = np.multiply(measured_signal, generator_signal) * np.sqrt(2)
        sin_lock =  np.multiply(measured_signal , -generator_signal) * np.sqrt(2)
        radius_lock = np.sqrt(sin_lock.mean(axis = 1)**2 + cos_lock.mean(axis = 1)**2)
        phase_lock = np.arctan2(sin_lock.mean(axis = 1), cos_lock.mean(axis = 1))
        return sin_lock, cos_lock, radius_lock, phase_lock


    def sine_pattern(self):

        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal = np.sin(sample*w + self.phase_shift)*self.amplitude + self.offset
        
        return signal      
    
    def sine_pattern_dw(self, dw, A1, A2):
        
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal1 = np.sin(sample*w + self.phase_shift)*self.amplitude + self.offset
        f = self.excitation_frequency - dw
        w = 2*np.pi*f
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal2 = np.sin(sample*w + self.phase_shift)*self.amplitude + self.offset
        return A1*signal1 + A2*signal2      


    def sine_pattern_modified_dw(self, dw):
        
        f = self.excitation_frequency - dw
        w = 2*np.pi*f
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal = np.sin(sample*w + self.phase_shift)*self.amplitude + self.offset
        return signal   


    def square_pattern_phase(self):
    
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal = np.sin(sample*w)*self.amplitude + self.offset
        signal[signal >= self.offset] = self.amplitude + self.offset
        signal[signal < self.offset] = self.offset - self.amplitude
        signal = np.roll(signal, int(self.phase_shift*test.writing_rate/(2*np.pi * self.excitation_frequency)) )
        return signal


    def triangle_pattern(self):
    
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.writing_samples, num = self.writing_samples, endpoint=False)
        signal = np.zeros(self.writing_samples)
        inf = sample%(self.writing_samples//self.num_period)<(self.writing_samples//(2*self.num_period))
        sup = sample%(self.writing_samples//self.num_period)>=(self.writing_samples//(2*self.num_period))
        signal[sup] = (sample[sup]%(self.writing_samples//self.num_period)) - (self.writing_samples//(2*self.num_period))
        signal[inf] = (self.writing_samples//(2*self.num_period))-(sample[inf]%(self.writing_samples//self.num_period))

        signal = np.roll(signal, int(self.phase_shift * self.writing_rate/w) )
        signal = signal-signal.mean()
        signal = signal*self.amplitude/signal.max()
        signal = signal + self.offset
        return signal

    def square_pattern2(self):
        
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal = sample * 0
        for k in np.linspace(1, 19, 10):
            signal += np.sin(sample*w*k)*self.amplitude/k
        #signal[signal > self.amplitude] = self.amplitude
        #signal[signal < -self.amplitude] = -self.amplitude
        #signal += self.offset
 
        return signal  

    def square_pattern3(self):
        
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.acq_time, num = self.writing_samples, endpoint=False)
        signal = sample * 0
        signal += np.sin(sample*w)*self.amplitude * 2
        signal[signal > self.amplitude] = self.amplitude
        signal[signal < -self.amplitude] = -self.amplitude
        signal += self.offset
 
        return signal  

    def square_pattern(self): 
    
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.writing_samples, num = self.writing_samples, endpoint=False) ##Coucou 1 endpoint=False
        signal = np.zeros(self.writing_samples)
        signal[sample%(self.writing_samples//self.num_period)<(self.writing_samples//(2*self.num_period))] = self.amplitude + self.offset
        signal[sample%(self.writing_samples//self.num_period)>=(self.writing_samples//(2*self.num_period))] = self.offset - self.amplitude
        signal = np.roll(signal, int(self.phase_shift * self.writing_rate/w) )


        return signal	

    def pulse(self, ratio = 1): 
    
        w = 2*np.pi*self.excitation_frequency
        sample = np.linspace(0, self.writing_samples, num = self.writing_samples, endpoint=False) ##Coucou 1 endpoint=False
        signal = np.ones(self.writing_samples) * (self.offset - self.amplitude)
        ind_on = sample%(self.writing_samples//self.num_period)<(self.writing_samples//(2*self.num_period))
        shift = np.roll(sample, -int(2 * np.pi * (1 - ratio) * self.writing_rate/w) )
        ind_off = shift%(self.writing_samples//self.num_period)<(self.writing_samples//(2*self.num_period))
        signal[ind_on * ind_off] = self.amplitude + self.offset
    


        return signal	



    def select_trigger(self, task, selected_edge, reference_trigger_channel, delay = 0):
        if selected_edge == "FALLING":
            edge = Edge.FALLING
        elif selected_edge == "RISING":
            edge = Edge.RISING
        else:
            logger.critical('Choose either FALLING or RISING as string for the edge detection')

        task.triggers.start_trigger.cfg_dig_edge_start_trig(reference_trigger_channel,
                                                        trigger_edge = edge)      
        
        if delay > 0: 
            task.triggers.start_trigger.delay = 19
            task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS


    def generator_analog_init(self, signal):

        task_generator_analog = nidaqmx.Task()
        self.add_channels(task_generator_analog.ao_channels.add_ao_voltage_chan, self.generator_analog_channels)
        # task_generator_analog.ao_channels.add_ao_voltage_chan(self.generator_analog_channel)
        task_generator_analog.timing.ref_clk_src = "/Dev1/10MHzRefClock"
        #task_generator_analog.timing.ref_clk_rate = 20e6
        task_generator_analog.timing.cfg_samp_clk_timing(rate = self.writing_rate, 
                                            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                            samps_per_chan = self.writing_samples * self.number_of_cycles)

        self.select_trigger(task_generator_analog, self.trig_chan_generator_edge, self.trig_chan_generator)

        if len(signal.shape) == 1:
            SignalStreamer = nidaqmx.stream_writers.AnalogSingleChannelWriter(task_generator_analog.out_stream, auto_start=False)
        elif len(signal.shape) > 1: 
            SignalStreamer = nidaqmx.stream_writers.AnalogMultiChannelWriter(task_generator_analog.out_stream, auto_start=False)
        else: 
            logger.error("Error in the signal array")
        SignalStreamer.write_many_sample(signal)
        self.task_generator_analog = task_generator_analog

    def generator_digital_init(self, signal):

        task_generator_digital = nidaqmx.Task()
        self.add_channels(task_generator_digital.do_channels.add_do_chan, self.generator_digital_channels)
#        task_generator_digital.do_channels.add_do_chan(self.generator_digital_channel)
        task_generator_digital.do_overcurrent_reenable_period = 1
        task_generator_digital.timing.ref_clk_src = "/Dev1/10MHzRefClock"
        task_generator_digital.timing.ref_clk_rate = 10.0e6
        task_generator_digital.timing.cfg_samp_clk_timing(rate = self.writing_rate, 
                                            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                            samps_per_chan = self.writing_samples * self.number_of_cycles)

        self.select_trigger(task_generator_digital, self.trig_chan_generator_edge, self.trig_chan_generator)
        signal = signal.astype(np.uint8)
        if len(signal.shape) == 1:
            SignalStreamer = nidaqmx.stream_writers.DigitalSingleChannelWriter(task_generator_digital.out_stream, auto_start=False)
        elif len(signal.shape) > 1: 
            SignalStreamer = nidaqmx.stream_writers.DigitalMultiChannelWriter(task_generator_digital.out_stream, auto_start=False)
        else: 
            logger.error("Error in the signal array")
        SignalStreamer.write_many_sample_port_byte(signal)
        self.signal_streamer = SignalStreamer
        self.task_generator_digital = task_generator_digital

    def generator_init(self, signal):
        #split digital and analog signal based on num_channel_analog_gene/digital_gene
        self.generator_analog_init(signal[:self.num_channels_analog_gene()])
        self.generator_digital_init(signal[self.num_channels_digital_gene():])
 
    def generator_init_and_start(self, signal_analog, signal_digital):
        #split digital and analog signal based on num_channel_analog_gene/digital_gene
        if self.num_channels_analog_gene() > 0:
            self.generator_analog_init(signal_analog)
            self.task_generator_analog.start()
        else: 
            logger.info("Did not start analog generator, check generator_analog_channels")
        if self.num_channels_digital_gene() > 0:
            self.generator_digital_init(signal_digital) 
            self.task_generator_digital.start()
        else: 
            logger.info("Did not start digital generator, check generator_analog_channels")


    def generator_start(self):
        try:
            self.task_generator_analog.start()
        except:
            pass
        try:
            self.task_generator_digital.start()
        except:
            pass

    def generator_stop(self):
        try:
            self.task_generator_analog.stop()
        except:
            pass
        try:
            self.task_generator_digital.stop()
        except:
            pass

    def generator_close(self):
        try:
            self.task_generator_analog.close()
        except:
            pass
        try:
            self.task_generator_digital.close()
        except:
            pass

    def detector_analog_init(self):

        task_detector_analog = nidaqmx.Task()
        self.add_channels(task_detector_analog.ai_channels.add_ai_voltage_chan, self.detector_analog_channels)
        #task_detector_analog.ai_channels.add_ai_voltage_chan(self.detector_analog_channel)
        #task_detector_analog.ai_channels.ai_averaging_win_size = self.averaging_win
        #task_detector.timing.ref_clk_src = "/Dev1/ai/SampleClock"
        task_detector_analog.timing.ref_clk_src = "/Dev1/10MHzRefClock"

        #task_detector_analog.timing.ref_clk_rate = 1e8

        task_detector_analog.timing.cfg_samp_clk_timing(self.sample_rate, 
                        sample_mode=nidaqmx.constants.AcquisitionType.FINITE, 
                        samps_per_chan = self.reading_samples)
        self.select_trigger(task_detector_analog, self.trig_chan_detector_edge, self.trig_chan_detector)

        reader_analog = nidaqmx.stream_readers.AnalogMultiChannelReader(task_detector_analog.in_stream)
        self.reader_analog = reader_analog
        self.task_detector_analog = task_detector_analog

    def detector_digital_init(self):
        task_detector_digital = nidaqmx.Task()
        self.add_channels(task_detector_digital.di_channels.add_di_chan, self.detector_digital_channels)
        #task_detector_digital.di_channels.add_di_chan(self.detector_digital_channel)
        #task_detector.timing.ref_clk_src = "/Dev1/ai/SampleClock"
        task_detector_digital.timing.ref_clk_src = "/Dev1/10MHzRefClock"

        #task_detector_digital.timing.ref_clk_rate = 1e8

        task_detector_digital.timing.cfg_samp_clk_timing(self.sample_rate, 
                        sample_mode=nidaqmx.constants.AcquisitionType.FINITE, 
                        samps_per_chan = self.reading_samples)

        self.select_trigger(task_detector_digital, self.trig_chan_detector_edge, self.trig_chan_detector)

        reader_digital = nidaqmx.stream_readers.DigitalMultiChannelReader(task_detector_digital.in_stream)
        self.reader_digital = reader_digital
        self.task_detector_digital = task_detector_digital

    def detector_init(self):
        self.detector_analog_init()
        self.detector_digital_init()

    def detector_start(self):
        try:
            self.task_detector_analog.start()
        except:
            pass
        try:
            self.task_detector_digital.start()
        except:
            pass

    def detector_stop(self):
        try:
            self.task_detector_analog.stop()
        except:
            pass
        try:
            self.task_detector_digital.stop()
        except:
            pass

    def detector_close(self):
        try:
            self.task_detector_analog.close()
        except:
            pass
        try:
            self.task_detector_digital.close()
        except:
            pass



    def trigger_init(self):
        
        task_trigger = nidaqmx.Task()
        
        task_trigger.co_ctr_timebase_rate = 1e9# self.sample_rate
        
        task_trigger.co_channels.add_co_pulse_chan_freq(counter = self.trig_chan, 
                                    freq = self.trigger_frequency, units = nidaqmx.constants.FrequencyUnits.HZ, 
                                initial_delay = 0.0,
                                duty_cycle = 0.5)
        task_trigger.timing.ref_clk_src ="/Dev1/10MHzRefClock"
        task_trigger.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS)
        
        #task_trigger.co_channels.add_co_pulse_chan_time(counter = self.trig_chan, low_time = 0.01, high_time = 0.01)
        self.task_trigger = task_trigger
    
    def read_analog(self):
        output_analog = np.zeros([self.num_channels_analog(), self.reading_samples])
        self.reader_analog.read_many_sample(data = output_analog, number_of_samples_per_channel = -1, #self.reading_samples, 
                                    timeout = max(5 * self.acq_time, 5/self.trigger_frequency, 100))
        output_analog = np.around(output_analog, 6) #Round all values to 6 decimals to avoid overflow
        time_array = np.linspace(0, self.reading_samples/self.sample_rate, self.reading_samples, endpoint=False)

        return output_analog, time_array

    def read_digital(self):
        output_digital = np.zeros([self.num_channels_digital(), self.reading_samples], dtype = np.uint16)
        self.reader_digital.read_many_sample_port_uint16(data = output_digital, number_of_samples_per_channel = -1, #self.reading_samples, 
                                    timeout = max(10 * self.acq_time, 3/self.trigger_frequency))
        output_digital = output_digital.astype(float) #Round all values to 6 decimals to avoid overflow
        #output_digital = np.around(output_digital, 6) #Round all values to 6 decimals to avoid overflow
        time_array = np.linspace(0, self.reading_samples/self.sample_rate, self.reading_samples, endpoint=False)

        return output_digital, time_array



    def read_detector_no_correct(self):
        logger.info("frequency: %0.3f \n amplitude %0.3f \n offset %0.3f  \n acquisiton time %0.3f \n points per period %0.3f \n number of periods %0.3f"%(self.excitation_frequency, self.amplitude, 
                        self.offset, self.acq_time, self.points_per_period, self.num_period))

        output_analog, time_array = self.read_analog()
        output_digital, time_array = self.read_digital()

        output = np.vstack((output_analog, output_digital))
    
        if np.sum(output[0]> 1.4) > 100:
            logger.error("Voltage saturation of the MPPC intensity reader")
        
        try:
            if np.sum(output[2]> 3.8) > 100:
                logger.error("Voltage saturation of the MPPC fluorescence reader")     
        except: 
            pass  
        return output, time_array

    
    def read_black(self):
        self.detector_init()
        self.trigger_init()

        self.task_trigger.start()

        output, time_array = self.read_detector_no_correct()

        self.end_tasks()     
        all_channels = self.detector_analog_channels + self.detector_digital_channels   
        with open("C:/Users/Lab/Desktop/DREAM_repo/Codes_Alienor/PAMFluo-dynamic_python/specs/config_black.json", 'r') as file:
            black_level = json.load(file)
            
            for i, channel in enumerate(all_channels):
                black_level[channel][0] = output[i].mean()
                black_level[channel][1] = output[i].std()

        with open("C:/Users/Lab/Desktop/DREAM_repo/Codes_Alienor/PAMFluo-dynamic_python/specs/config_black.json", 'w') as file:
            json.dump(black_level, file)


    def read_detector(self):
        output, time_array = self.read_detector_no_correct()
        all_channels = self.detector_analog_channels + self.detector_digital_channels   

        with open("C:/Users/Lab/Desktop/DREAM_repo/Codes_Alienor/PAMFluo-dynamic_python/specs/config_black.json", 'r') as file:
            black_level = json.load(file)
            
            for i, channel in enumerate(all_channels):
                output[i] -= black_level[channel][0] 
        return output, time_array
    
    def quick_test(self, signal_analog, signal_digital):

        self.generator_init_and_start(signal_analog, signal_digital)
        self.detector_init()
        self.trigger_init()

        self.task_trigger.start()

        output, time_array = self.read_detector()

        self.end_tasks()

        average_output, x_time = self.averaging(output, time_array)

        self.p.xlabel = 'time (s)'
        self.p.ylabel = 'voltage (V)'
        self.p.title = "Detectors responses"
        self.p.label_list = ["intensity", "None", "fluorescence", "None", "arduino_blue", "arduino_purple", "arduino_green"]

        fig = self.p.plotting(x_time, [out for out in average_output])
        self.p.save_name = "quick_test"
        self.p.saving(fig)
        self.shut_down_LED(1)
        return output, average_output, time_array, x_time



    def quick_acquisition(self):

        self.detector_init()

        self.select_trigger(self.task_detector_analog, self.trig_chan_detector_edge, "/Dev1/10MHzRefClock")
        self.select_trigger(self.task_detector_digital, self.trig_chan_detector_edge, "/Dev1/10MHzRefClock")

        self.detector_start()
        output, time_array = self.read_analog()

        self.task_detector_analog.stop()
        self.task_detector_digital.stop()

        self.task_detector_analog.close()
        self.task_detector_digital.close()

        average_output, x_time = self.averaging(output, time_array)

        self.p.xlabel = 'time (s)'
        self.p.ylabel = 'voltage (V)'
        self.p.title = "Detectors responses"
        self.p.color_list = [(1, 0, 0), (0,0,1), (0, 0, 0), (0,1,1), (1,1,0)]
        self.p.label_list = ['MPCC Intensité','Photodiode', 'MPPC Fluo', 'LED input', 'Trigger']
        fig = self.p.plotting(x_time, [out for out in average_output])
        self.p.save_name = "quick_acquisition"
        self.p.saving(fig)

        return output, average_output, time_array, x_time



    def shut_down_LED(self, sleep_time, all = False):
        if all == True: 
            self.all_connected_channels()

        logger.info('Shutting down all LEDs for %d seconds'%sleep_time)
        end_signal_analog = np.zeros((self.num_channels_analog_gene(), 100))
        end_signal_digital =np.zeros((self.num_channels_digital_gene(),100))
 
        self.generator_init_and_start(end_signal_analog, end_signal_digital)

        self.trigger_init()
        self.task_trigger.start()
        time.sleep(sleep_time)
        self.generator_stop()
        self.generator_close()
        self.task_trigger.stop()
        self.task_trigger.close()

    def set_level_LED(self, voltage_analog, voltage_digital, sleep_time):

        if self.num_channels_analog_gene() > 1:
            V = np.array(voltage_analog)
            V = np.repeat(np.expand_dims(V, axis = 1), 100, axis = 1)
            self.generator_analog_init(V)
            self.task_generator_analog.start()
        elif self.num_channels_analog_gene() == 1: 
            self.generator_analog_init(np.array([voltage_analog]*100, dtype = float))
            self.task_generator_analog.start()
        else: 
            pass
        if self.num_channels_digital_gene() > 1:
            V = np.array(voltage_digital)
            V = np.repeat(np.expand_dims(V, axis = 1), 100, axis = 1)
            self.generator_digital_init(V)
            self.task_generator_digital.start()
        elif self.num_channels_digital_gene() == 1: 
            self.generator_digital_init(np.array([voltage_digital]*100, dtype = float))
            self.task_generator_digital.start()
        else:
            pass

        self.trigger_init()
        
        self.task_trigger.start()
        time.sleep(sleep_time)
        self.task_trigger.stop()
        self.task_trigger.close()

        self.generator_stop()
        self.generator_close()

    def generator_to_zero(self, wait_for_user_message, mean_voltage = 0):
        self.generator_stop()
        self.generator_close()
        if self.num_channels_analog_gene() > 1:
            self.generator_init(np.vstack([self.sine_pattern()]*self.num_channels_analog_gene() * 0))
        else: 
            self.generator_init(self.sine_pattern() * mean_voltage)
        self.task_trigger.start()
        self.generator_start()
        input(wait_for_user_message)
        self.task_trigger.stop()

    def end_tasks(self):
        self.generator_stop()
        self.generator_close()
        self.detector_stop()
        self.detector_close()

        try: 
            self.task_trigger.stop()
        except: 
            pass

        try: 
            self.task_trigger.close()
        except: 
            pass


    def experiment_folder(self, experiment_name):
        """creates the folder that will store locally the experiment"""
        self.experiment_name = experiment_name
        self.save_folder = "E:/Experimental_data/DREAM_microscope/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_')) + experiment_name
        if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
    
    def copy_py_file(self, filename):
        source = os.path.realpath(filename)
        file_name = os.path.split(source)[1]
        target = self.save_folder + '/' + file_name
        shutil.copy(source, target)

    def move_logger(self):
        target = self.save_folder
        move_log(logger, target)

    def end_exp(self, filename):
        self.shut_down_LED(1)
        self.copy_py_file(filename)
        self.move_logger()


if __name__ == "__main__":

    test = Acquisition()
    
    logger.name = "AcquisitionClass"


    test.print_prop()    


    test.amplitude = 0.2
    test.offset = 1

    test.shut_down_LED(3)
