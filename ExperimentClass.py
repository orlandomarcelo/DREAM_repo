import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re

from striprtf.striprtf import rtf_to_text


class Experiment:
    def __init__(self, name, equipment, local="IBPC", DataType=".dat", sep = ',', is_sub_experiment=False, parent_experiment_name=None):

        self.name = name
        self.equipment = equipment
        self.DataType = DataType
        self.local = local
        self.is_sub_experiment = is_sub_experiment
        self.parent_experiment_name = parent_experiment_name

        if self.is_sub_experiment:
            if self.local == "IBPC":
                self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{parent_experiment_name}"
            elif self.local == "ENS":
                self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{parent_experiment_name}"
            elif self.local == "home":
                self.path = f"C:/Users/marce/ownCloud_ORLANDO/Doutorado/Dados experimentais/{self.equipment}/{parent_experiment_name}"
            else:
                print("Dont know the path for this place")
        else:
            if self.local == "IBPC":
                self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
            elif self.local == "ENS":
                self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
            elif self.local == "home":
                self.path = f"C:/Users/marce/ownCloud_ORLANDO/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
            else:
                print("Dont know the path for this place")
                
        self.fig_folder = self.path + "/Figures"
        if not os.path.isdir(self.fig_folder):
            os.mkdir(self.fig_folder)


        self.AllData = pd.read_csv(f"{self.path}/{self.name}{self.DataType}", index_col=False, sep = sep)
        self.Data = self.AllData.iloc[:,1:]
        self.Time = self.AllData.iloc[:,0]
        self.records = []

        for i, k in enumerate(self.Data.keys()):
            self.records.append(k.replace(" ",""))
            
        self.Data.columns = self.records


        self.clean_times = []
        self.clean_data = []
        aux_time = np.array(self.Time)
        for i, k in enumerate(self.records):
            a = np.array(self.Data.iloc[:, i])
            indices = np.invert(np.isnan(a))
            self.clean_times.append(aux_time[indices])
            self.clean_data.append((a[indices]))
        
        if not os.path.exists(f"{self.path}/pre_annotation.csv"):
            rtf_file_path = f"{self.path}/{self.name}.rtf"
            try:
                with open(rtf_file_path) as infile:
                    content = infile.read()
                    text = rtf_to_text(content)

                pattern = r"Enregistrement\s(\d{1,3})\s{6}(\S{8}).+\s(\d{1,3})\sµE.+\s(.+)"
                pre_info = re.findall(pattern, text)
                record_str = ["E" + str(num) for num in np.asarray(pre_info)[:,0]]
                df = pd.DataFrame(np.asarray(pre_info), columns=["Record", "Real_time", "Light_intensity", "Comment"])
                df.insert(1, "Record_str", record_str)
                df.to_csv(f"{self.path}/pre_annotation.csv", index=False)
            except:
                pass

        try:
            self.annotations = pd.read_csv(f"{self.path}/annotation.csv", index_col=False, sep = ";")
        except:
            pass



    def sub_experiments(self, sub_experiment_name, keys):
        sub_experiment_data = []
        count = 0
        aux = list(keys)
        for i , k in enumerate(self.records):
            for l, m in enumerate(aux):
                if k == m:
                    if count == 0:
                        sub_experiment_data.append(self.clean_times[i])
                        count = 1
                    sub_experiment_data.append(self.clean_data[i])
        aux.insert(0,"Time")
        try:
            df = pd.DataFrame(data = np.transpose(np.array(sub_experiment_data)), columns = aux)
        except:
            df = pd.DataFrame(data = np.array(sub_experiment_data), columns = aux)

        df.to_csv(f"{self.path}/{sub_experiment_name}.dat", index = False)



    def get_all_sub_experiments(self, names_list, keys_list):
        for i, k in enumerate(names_list):
            self.sub_experiments(k, list(keys_list[i]))
    
    def get_annotations(self):
        return self.annotations
    
    def get_keys_list(self, record_array):
        return ["E" + str(x) for x in record_array.tolist()]


    def extract_recording(self, record_name):
        for i , k in enumerate(self.records):
            if k == record_name:
                sub_experiment_time = self.clean_times[i]
                sub_experiment_data = self.clean_data[i]
        
        aux = ["Time", record_name]
        
        df = pd.DataFrame(data = np.transpose([sub_experiment_time,sub_experiment_data]) , columns = aux)

        df.to_csv(f"{self.path}/{record_name}.dat", index = False)
        
    def average_recordings(self, record_list):
        selected_cols = self.Data[record_list]
        row_means  = selected_cols.mean(axis=1)
        rows_std = selected_cols.std(axis=1)
        
        return np.array(row_means), np.array(rows_std)







