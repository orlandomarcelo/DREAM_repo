import pandas as pd
import numpy as np


class Experiment:
    def __init__(self, name, equipment, local="IBPC", DataType=".dat", diff_xaxis=False):

        self.name = name
        self.equipment = equipment
        self.DataType = DataType
        self.local = local
        self.diffx_axis = diff_xaxis

        if self.local == "IBPC":
            self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
        elif self.local == "ENS":
            self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
        elif self.local == "home":
            self.path = f"C:/Users/Orlando/ownCloud - ORLANDO Marcelo@mycore.cnrs.fr/Doutorado/Dados experimentais/{self.equipment}/{self.name}"
        else:
            print("Dont know the path for this place")

        self.RawData = pd.read_csv(f"{self.path}/{self.name}{self.DataType}", index_col=False)

        if self.diffx_axis:
            self.clean_times = []
            self.clean_data = []
            aux_time = np.array(self.RawData.Time)
            for i, k in enumerate(self.RawData.keys()):
                if i == 0:
                    continue
                else:
                    a = np.array(self.RawData.iloc[:, i])
                    indices = np.invert(np.isnan(a))
                    self.clean_times.append(aux_time[indices])
                    self.clean_data.append((a[indices]))
