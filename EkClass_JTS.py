import numpy as np
import pandas as pd

import ExperimentClass
import tools
import math_functions as mf


class EkClass_JTS(ExperimentClass.Experiment):
    def __init__(self, name, Ek_records_str = "16-21", PWM_list = [0.01, 0.04, 0.1, 0.25, 0.5, 0.8], Flash_record_str= "7", 
                 start = -5, stop = 25, num = 50, index_stop_fit = 26):
        super().__init__(name, equipment = "JTS", local="IBPC", DataType=".dat", is_sub_experiment=False, parent_experiment_name=None)
        self.Ek_records = tools.create_record_list(Ek_records_str)
        self.PWM_list = PWM_list
        self.Flash_records = tools.create_record_list(Flash_record_str)
        self.start = start
        self.stop = stop 
        self.num = num
        self.index_stop_fit = index_stop_fit
        
        self.Flash_Time = self.clean_times[self.records.index(self.Flash_records[0])]
        self.Flash_Data = self.clean_data[self.records.index(self.Flash_records[0])]
        
        self.calib = self.Flash_Data[20]-self.Flash_Data[19]
        
        self.xfit_lin = []
        self.yfit_lin = []
        self.param = []
        self.vitesse = []


        self.Ek_time = self.clean_times[self.records.index(self.Ek_records[0])][21:self.index_stop_fit]


        for i, k in enumerate(self.Ek_records):
            ydata = self.clean_data[self.records.index(k)][21:index_stop_fit]
            popt, x, y =  tools.lin_fit(self.Ek_time, ydata, start, stop, num)
            self.xfit_lin.append(x)
            self.yfit_lin.append(y)
            self.param.append(popt)
            self.vitesse.append(-1000*self.param[i][0]/self.calib)
            
            
        self.intensity_rel = np.insert(self.PWM_list, 0, 0)
        self.intensity = 1326 * np.asarray(self.PWM_list) + 11.9
        self.intensity = np.insert(self.intensity, 0, 0)
        self.vitesse.insert(0,0)
        
        popt, pcov, self.xfit_Ek_rel, self.yfit_Ek_rel = tools.Ek_fit(self.intensity_rel, self.vitesse, 0, 1, 50, p0 = [200, 0.2])
        self.Ek_rel = popt[1]
        self.Ek_rel_err = np.sqrt(np.diag(pcov))[1]
     
        popt, pcov,  self.xfit_Ek, self.yfit_Ek = tools.Ek_fit(self.intensity, self.vitesse, 0, 1200, 50, p0 = [200, 100])
        self.Ek = popt[1]
        self.Ek_err = np.sqrt(np.diag(pcov))[1]

                
                
