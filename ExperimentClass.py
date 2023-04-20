import pandas as pd
import numpy as np


class Experiment:
    def __init__(self, name, equipment, local="IBPC", DataType=".dat", diff_xaxis=False, is_sub_experiment=False, parent_experiment_name=None):

        self.name = name
        self.equipment = equipment
        self.DataType = DataType
        self.local = local
        self.diff_xaxis = diff_xaxis
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

        AllData = pd.read_csv(f"{self.path}/{self.name}{self.DataType}", index_col=False)
        self.Data = AllData.iloc[:,1:]
        self.Time = AllData.iloc[:,0]
        self.keys = []

        for i, k in enumerate(self.Data.keys()):
            self.keys.append(k.replace(" ",""))

        if self.diff_xaxis:
            self.clean_times = []
            self.clean_data = []
            aux_time = np.array(self.Time)
            for i, k in enumerate(self.keys):
                a = np.array(self.Data.iloc[:, i])
                indices = np.invert(np.isnan(a))
                self.clean_times.append(aux_time[indices])
                self.clean_data.append((a[indices]))



    def sub_experiments(self, sub_experiment_name, keys):
        sub_experiment_data = []
        count = 0
        aux = list(keys)
        for i , k in enumerate(self.keys):
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
        self.annotations = pd.read_csv(f"{self.path}/annotation.csv", index_col=False, sep = ";")
        return self.annotations
    
    def get_keys_list(self, record_array):
        return ["E" + str(x) for x in record_array.tolist()]


    def extract_recording(self, record_name):
        for i , k in enumerate(self.keys):
            if k == record_name:
                sub_experiment_time = self.clean_times[i]
                sub_experiment_data = self.clean_data[i]
        
        aux = ["Time", record_name]
        
        df = pd.DataFrame(data = np.transpose([sub_experiment_time,sub_experiment_data]) , columns = aux)

        df.to_csv(f"{self.path}/{record_name}.dat", index = False)



