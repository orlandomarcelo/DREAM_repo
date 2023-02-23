import numpy as np
import glob
import skimage
from joblib import Parallel, delayed


class activation_experiment():
    def __init__(self, folder):
        self.experiment_folder = folder
        self.activation = glob.glob(folder + "/*constant_light_*")
        #self.relaxation_LL = glob.glob(folder + "/*constant_light_LL*")
        self.measure = []
        for i in range(30):
            self.measure.append(glob.glob(folder + "/*qE_calib_%d"%i))
        
        self.measures = glob.glob(folder + "/*qE_calib*")

        self.ratio_set = []
        self.dark_relaxation = glob.glob(folder + "/*dark_relaxation*")
        self.ana=False
    
    def get_exp_dicts(self, exp_set):
        exp_dicts = []
        for exp in exp_set:
            if self.ana==False:
                exp_dicts.append(np.load(exp + '/items_dict.npy', allow_pickle=True).item())
            if self.ana==True:
                exp_dicts.append(np.load(exp + '/analysis/items_dict.npy', allow_pickle=True).item())
        return exp_dicts
            
    def times_absolute(self, exp_set): 
        times = []
        
        for exp in exp_set:
            times.append(np.load(exp + "/video_timing.npy"))
        return np.array(times)
    
    def times(self, exp_set): 
        times = self.times_absolute(exp_set)
        return times - np.expand_dims(times[:,0], 1)

    def means(self, exp_set):
        means = []
        exp_dicts = self.get_exp_dicts(exp_set)
        for exp_dict in exp_dicts:
            means.append(exp_dict["total_mean"])        
        return np.array(means)
    
    def read_pulses(self, exp_set):
        pulse_data = []
        for exp in exp_set:
            frames_full = skimage.io.imread(exp + "/video.tiff")
            value = np.sum(frames_full, axis=(1,2))
            frameset = np.mean(frames_full, axis = 0)
            pulse_data.append(value/np.sum(frameset>2))
        return np.array(pulse_data)
        
        
    def get_traces(self, exp_set):
        traces = []
        labels = []
        exp_dicts = self.get_exp_dicts(exp_set)
        i = 0
        for exp_dict in exp_dicts:
 
            for algae in exp_dict["items_dict"].keys():
                s = exp_dict["items_dict"][algae]["surface"]
                if s>5 and s<60:
                    traces.append(exp_dict["items_dict"][algae]["mean"])
                    labels.append(i)
                    i += 1
        return np.array(traces), np.array(labels)
    
    
    def get_traces_and_times(self, exp_set):
        traces = []
        labels = []
        times = []
        exp_dicts = self.get_exp_dicts(exp_set)
        i = 0
        times_general = self.times(exp_set)
        for k, exp_dict in enumerate(exp_dicts):
 
            for algae in exp_dict["items_dict"].keys():
                s = exp_dict["items_dict"][algae]["surface"]
                if s>5 and s<10000:
                    traces.append(exp_dict["items_dict"][algae]["mean"])
                    labels.append(i)
                    times.append(times_general[k])
                    i += 1
        return np.array(traces), np.array(labels), np.array(times)
        
    def get_qE(self, exp_set):
        traces = []
        labels = []
        exp_dicts = self.get_exp_dicts(exp_set)
        i = 0
        for exp_dict in exp_dicts:
 
            for algae in exp_dict["items_dict"].keys():
                s = exp_dict["items_dict"][algae]["surface"]
                if s>5 and s<60:
                    trace = exp_dict["items_dict"][algae]["mean"]
                    traces.append((trace[250]-trace)/trace)
                    labels.append(i)
                    i += 1
        return np.array(traces), np.array(labels)
    
    
    def get_ratio(self, exp_set, F0, F1):
        traces, labels = self.get_traces(exp_set)

        if isinstance(F1, int):
            value = traces[:,F1]
        else:
            value = np.mean(traces[:,F1], axis = 1)

            
        q1 = (traces[:,F0] - value)/value

        return q1   

    def get_yield(self, exp_set, F0, F1):
        traces, labels = self.get_traces(exp_set)

        if isinstance(F1, int):
            value = traces[:,F1]/np.mean(traces[:,F1+2:F1+8])
        else:
            value = np.mean(traces[:,F1], axis = 1)/np.mean(traces[:,F1[1]+2:F1[1]+8])
        
        
        
    def get_ratio_array(self, exp_set, list_of_couples):
        ratios = []
        for couple in list_of_couples:
            ratios.append(self.get_ratio(exp_set, couple[0], couple[1]))
        ratios = np.array(ratios)
        return ratios
        
        
    def get_yield_array(self, exp_set, list_of_couples):
        ratios = []
        for couple in list_of_couples:
            ratios.append(self.get_ratio(exp_set, couple[0], couple[1]))
        ratios = np.array(ratios)
        return ratios
    
    def get_ratio_diff(self, exp_set, F0, F1):
        traces, labels = self.get_traces(exp_set)
        value_F = [[], []]
        for pos, F in enumerate([F0, F1]):
            if isinstance(F, int):
                value_F[pos] = (traces[:,250] - traces[:,F])/traces[:,F]
            else:
                value_F[pos] = np.mean((traces[:,250] - traces[:,F])/traces[:,F], axis = 1)


        q1 = value_F[1] - value_F[0]

        return q1      
        
        
    def get_diff_array(self, exp_set, list_of_couples):
        ratios = []
        for couple in list_of_couples:
            ratios.append(self.get_ratio_diff(exp_set, couple[0], couple[1]))
        ratios = np.array(ratios)
        return ratios
    
    
    def get_ratio_from_video(self, video, F0, F1, transform):
        #video = imageio.get_reader(start_vid)
        frame_250 = skimage.transform.warp(video.get_data(250), transform)
        frame_F0 =  skimage.transform.warp(video.get_data(F0), transform)
        frame_F1 = skimage.transform.warp(video.get_data(F1), transform)
        r0  = (frame_250 - frame_F0)/frame_F0
        r1 = (frame_250 - frame_F1)/frame_F1
        diff = r1-r0
        diff = np.nan_to_num(diff, neginf=0, nan=0, posinf=0) 

        
        return diff
    
    def get_ratio_video_array(self, video, list_of_couples, transform):
        ratios = []
        
        ratios = Parallel(n_jobs = -1 )(self.get_ratio_from_video(video, couple[0], couple[1], transform) for couple in list_of_couples)

        ratios = np.array(ratios)
        return ratios
