import numpy as np
import pandas as pd
import os
import sys
import importlib
import glob

import ExperimentClass
import tools
import math_functions as mf

class EkClass_PSI(ExperimentClass.Experiment):
    
    def __init__(self, name, intensity_list,  t_F0 = [0, 15], t_Fstat = [22, 35], t_Fmax = [35.004, 35.007]):
        super().__init__(name, "PSI", DataType = ".csv", sep = ';')
        self.intensity_list = intensity_list
        
        F_max = []
        F_stat = []
        F_0 = []
        phi_stat = []
        phi_0 = []
        NPQ = []
        ETR = []

        for i, k in enumerate(self.records[::-1]):
            F_max.append(np.mean(self.Data[k][np.where(self.Time>= t_Fmax[0])[0][0] :np.where(self.Time>= t_Fmax[1])[0][0]]))
            F_0.append(np.mean(self.Data[k][np.where(self.Time>= t_F0[0])[0][0] :np.where(self.Time>= t_F0[1])[0][0]]))
            F_stat.append(np.mean(self.Data[k][np.where(self.Time>= t_Fstat[0])[0][0] :np.where(self.Time>= t_Fstat[1])[0][0]]))
            phi_stat.append((F_max[i] - F_stat[i]) / F_max[i])
            phi_0.append((F_max[i] - F_0[i]) / F_max[i])
            NPQ.append((F_max[0] - F_max[i])/F_max[i])
            ETR.append(phi_stat[i] * self.intensity_list[i])
            
        self.Fm_norm = F_max[0]
        
        F_max = list(np.array(F_max)/self.Fm_norm)
        F_stat = list(np.array(F_stat)/self.Fm_norm)
        F_0 = list(np.array(F_0)/self.Fm_norm)
        
        self.params = pd.DataFrame({'Record': self.records[::-1], 'Actinic': self.intensity_list, 'F_max': F_max, 'F_stat': F_stat, 'F_0': F_0, 'phi_stat': phi_stat, 'phi_0': phi_0, 'NPQ': NPQ, 'ETR': ETR})
        
        self.light = [0] + self.params.Actinic.tolist()
        self.ETR = [0] + self.params.ETR.tolist()
        
        self.popt, self.pcov, self.xfit, self.yfit = tools.Ek_fit(self.light, self.ETR, 0, 100, 100, p0 = [200, 200])

        self.fit_err = tools.my_err(self.xfit, self.popt, self.pcov, mf.Ek)
        
        self.Ek = self.popt[1]
        self.Ek_err =  np.sqrt(np.diag(self.pcov))[1]
        
        
        