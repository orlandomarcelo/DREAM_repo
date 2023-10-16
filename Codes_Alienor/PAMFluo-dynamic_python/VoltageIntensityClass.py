import numpy as np
from alienlab.regression_func import get_func, regression_affine, get_affine_func, get_polyfit_func
from tkinter.filedialog import askopenfilename, askdirectory
import glob
from alienlab.utils import pandas_to_arrays
import pandas as pd
import numpy as np
from NIControl.RoutinesClass import Routines
import sys
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from config_DAQ import *
from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from ThorlabsControl.FW102 import FW102C
from time import time 
import os

class VoltageIntensity():
    def __init__(self, folder="None"):


        if  folder == "None":
            list_of_dir = glob.glob("G:/DREAM/from_github/PAMFluo/Experiments/*bode_diagram") # * means all if need specific format then *.csv
            self.experiment_folder = max(list_of_dir, key=os.path.getctime)
        else:
            self.experiment_folder = folder        
        #self.experiment_folder = askdirectory(initialdir = "G:/DREAM/from_github/PAMFluo/Experiments/")
        #print(self.experiment_folder)
        
        #"G:\DREAM/from_github\PAMFluo\Experiments/2021-05-19_18_30_bode_diagram"#askdirectory()
        headers, I480 = pandas_to_arrays(glob.glob(self.experiment_folder + "/*light_intensity_480.csv")[0])
        headers, I405 = pandas_to_arrays(glob.glob(self.experiment_folder + "/*light_intensity_405.csv")[0])
        self.voltage = {}
        self.voltage['blue'] = I480[1]
        self.voltage['purple'] =  I405[1]
        self.intensity = {}
        self.intensity['blue'] = I480[2]
        self.intensity['purple'] = I405[2]
        self.DO_spectrum = pd.read_csv("G:/DREAM/from_github/PAMFluo/specs/DO_wheel.csv", sep = ';', decimal = ",")
        self.detector_response = {}
        self.detector_response["blue"] = pandas_to_arrays(glob.glob(self.experiment_folder + "/*Detector_response_curve_blue.csv")[0])[1]
        self.detector_response["purple"] = pandas_to_arrays(glob.glob(self.experiment_folder + "/*Detector_response_curve_purple.csv")[0])[1]


    def get_DO_val(self, LED_color, DO):
        if LED_color == 'blue':
            wlgh = 480
        if LED_color == 'purple':
            wlgh = 405
        if DO != 0:
            func = get_func(self.DO_spectrum['wavelength'], self.DO_spectrum[str(DO)])
            density = func(wlgh)
        else:
            density = 0
        return np.float(10**(-density))


    def get_MPPC_voltage(self, LED_color, DO, voltage_input):
        voltage = self.detector_response[LED_color][1]
        MPPC_voltage = self.detector_response[LED_color][2]
        actinic_density = self.get_DO_val(LED_color, "do_mppc")
        MPPC_voltage = MPPC_voltage/actinic_density
        func = get_polyfit_func(voltage, MPPC_voltage, 2)
        density = self.get_DO_val(LED_color, DO)
        #print("density:", density)
        return func(voltage_input)*density



 #   def get_intensity_voltage(self):
 #       
 #       func = get_func(intensity, voltage)
 #       include DO

    def get_intensity_MPPC(self, LED_color, DO, MPPC_input):
        density = self.get_DO_val(LED_color, DO)
        MPPC_voltage = self.voltage[LED_color]
        actinic_density = self.get_DO_val(LED_color, "do_mppc")
        MPPC_voltage = MPPC_voltage/actinic_density
        intensity = self.intensity[LED_color]
        func = get_polyfit_func(MPPC_voltage, intensity, 2)
        return func(MPPC_input/density)*density

    def get_intensity_voltage(self, LED_color, DO, voltage_input):
        MPPC_input = self.get_MPPC_voltage(LED_color, DO, voltage_input)
        return self.get_intensity_MPPC(LED_color, DO, MPPC_input)


    def assert_calibration(self, logger, filter):
        port_DC4100 = "COM5"
        port_filter_wheel = "COM3"
        ctrlLED = ThorlabsDC4100(logger, port = port_DC4100)
        ctrlLED.initialise_fluo()
        ctrlLED.set_user_limit(LED_blue, 1000)
        ctrlLED.set_user_limit(LED_green, 1000)
        ctrlLED.set_user_limit(LED_purple, 1000)
   


        fwl = FW102C(port=port_filter_wheel)



 #       send 3 intensity, given voltage, measure MPPC, check predicted voltage 
        routines = Routines()
        routines.experiment_folder("Check_calibration")
        routines.generator_analog_channels = ["ao0"]
        routines.generator_digital_channels = []
        #480
        offset_min = 0.25
        offset_max =  2
        N_points = 5
        amplitude = 0
        routines.excitation_frequency = 10
        routines.num_period = 10
        routines.points_per_period = 10000
        routines.update_rates()

        fwl.move_to_filter(filters[filter])
        offset_range_480, val_480, fluo_range_480, full_output = routines.detector_response_routine(offset_min,
                                                                                    offset_max, amplitude, N_points, color = 'blue')

        predicted_MPPC = self.get_MPPC_voltage('blue', filter, offset_range_480)
        r2 = r2_score(predicted_MPPC, val_480)
        print(r2)
        plt.figure()
        plt.plot(predicted_MPPC, val_480)
        if abs(r2-1) > 0.2:
            print("YOU NEED TO CALIBRATE THIS SET-UP")
            sys.exit()
        else:
            print("CALIBRATION OK")
        return offset_range_480, val_480, fluo_range_480, full_output

    def visualise_conversion(self, LED_color):
        voltage = self.detector_response[LED_color][1]
        MPPC_voltage = self.detector_response[LED_color][2]
        actinic_density = self.get_DO_val(LED_color, "do_mppc")
        MPPC_voltage = MPPC_voltage/actinic_density        
        func1 = get_polyfit_func(voltage, MPPC_voltage, 2)

        plt.plot(voltage, MPPC_voltage,'o', color = 'r', label = "MPPC_voltage")
        plt.plot(voltage, func1(voltage),'--', color = 'r')
        plt.xlabel("voltage input")
        plt.legend()
        intensity = self.intensity[LED_color]
        func2 = get_polyfit_func(MPPC_voltage, intensity, 2)
        plt.figure()
        plt.plot(MPPC_voltage, intensity,'o', color = 'r', label = "intensity")
        plt.plot(MPPC_voltage, func2(MPPC_voltage),'--', color = 'r')
        plt.plot(MPPC_voltage, self.get_intensity_MPPC(LED_color, 0, MPPC_voltage), '*', color = 'k')
        plt.xlabel("MPPC voltage")
        plt.legend()     

        plt.figure()
        plt.plot(voltage, intensity, 'o', color = 'r', label = 'intensity')
        plt.plot(voltage, func2(func1(voltage)), '--', color = 'r')
        plt.plot(voltage, self.get_intensity_voltage(LED_color, 0, voltage), '*', color = 'k')
        plt.xlabel("voltage input")
        plt.legend()     

        plt.figure()
        x = []
        y = []
        for f in [0,1,2,3]:
            xi = self.get_MPPC_voltage(LED_color, f, voltage)
            yi = self.get_intensity_voltage(LED_color, f, voltage) 
            x.append(xi)
            y.append(yi)
            plt.loglog(xi, yi, '*', color = 'k', label = f)
        plt.xlabel("MPPC_voltage")
        plt.legend()  
        return x, y

if __name__ == "__main__":

    from alienlab.init_logger import logger

    calib = {}
    V = VoltageIntensity()
    for i in [0, 1, 2, 3]:
        calib[i] = V.assert_calibration(logger, i)

    plt.figure()
    for i in range(0,4):
        plt.loglog(calib[i][0]*V.get_DO_val('blue', i), calib[i][1])
        plt.loglog(calib[i][0]*V.get_DO_val('blue', i), V.get_MPPC_voltage('blue',i ,calib[i][0]))
    plt.show()

