#image analysis
import sys
import os
sys.path.append(os.path.abspath('../../../'))

#image analysis
import skimage.io
import imageio
import alienlab.plot
from alienlab.improcessing import normalize, grey_to_rgb, make_binary

from alienlab.fo import FramesOperator
import alienlab.io
from scipy import optimize
import glob
from alienlab.regression_func import *
import copy
from VoltageIntensityClass import VoltageIntensity
from tqdm import tqdm

from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from skimage.transform import resize
from IPython.display import display



from PIL import Image

import scipy

import tifffile as tiff

#interactive widget packages
from ipywidgets import interact, interactive, fixed, interact_manual
from tkinter.filedialog import askopenfilename, askdirectory

from VoltageIntensityClass import VoltageIntensity


import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as wdg  # Using the ipython notebook widgets
from alienlab.regression_func import platt, residuals
from alienlab.utils import clip
from alienlab.segment import uniform_mask, label_to_data
from alienlab.widget import click_to_graph
#%matplotlib inline


from mvgavg import mvgavg

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = StarDist2D.from_pretrained('2D_demo')

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--folder', type = str)

args = parser.parse_args()


def get_fit(decay, time, give_y = False):    

    time_spread = time.max()-time.min()
    start = np.mean(decay[0])
    stop = np.mean(decay[-5:])
    if len(decay)<15:
        x0 = [start, time_spread/10, stop]
    else:
        x0 = [start, time_spread, stop]

    parameters_estimated = optimize.least_squares(residuals,  x0, bounds = (-1e8,1e8),
                                args = (time, decay, exp_decay))
    
    if give_y:
        return np.array(parameters_estimated.x), exp_decay(parameters_estimated.x, time)
    else:
        return np.array(parameters_estimated.x)

def make_fit(decay, time_array):
    plt.figure()
    params, ypred = get_fit(decay, time_array, give_y = True)
    plt.plot(time_array, ypred, label = params[1])
    plt.plot(time_array, decay, '.')
    plt.xlabel("time(s)")
    plt.ylabel("fluorescence")
    plt.legend()
    
def predict_labels(image):
    axis_norm = (0,1)   # normalize channels independently
    lbl_cmap = random_label_cmap()
    img = normalize(image, 1,99.9, axis=axis_norm)
    labels, details = model.predict_instances(img)
    fig = plt.figure(figsize=(12,12))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');
    return labels, img, fig

def dict_trajectories(segmented, video):
    # Item time trajectories with overlaps
    # create a dictionnary with one entry for each item:
    '''
    { '1.0': {'x_coords': np array, x coordinates in HQ}
                'y_coords': np array,  y coordinates in HQ
                'binned_coords': set, couples of (x,y) coordinates in binned video
                'surface': number of pixels in the item in HQ
                'pixel_values': array, size: (N, s) where N is number of frames and s surface
                'mean': array, size N, mean value of the item intensity for each frame
                'std':  array, size N, std value of the item intensity for each frame
                'remains' : True, the item is present in this segmentation step
                }
    '2.0': {'x_coords'...
                    }
        }
    '''
    items = np.unique(segmented) #returns the set of values in items, corresponds to the values of the markers of local_maxima

    items_dict = {}
    for k in items:
        key = str(k)
        items_dict[key] = {}
        x_coords, y_coords = np.nonzero(segmented == k)
        items_dict[key]['x_coords'] = x_coords
        items_dict[key]['y_coords'] = y_coords
        pixel_values = video[:,x_coords, y_coords]
        items_dict[key]['pixel_values'] = pixel_values
        items_dict[key]['surface'] = pixel_values.shape[1]
        items_dict[key]['mean'] = np.mean(pixel_values, axis = 1)
        items_dict[key]['std'] = np.std(pixel_values, axis = 1)
        items_dict[key]['remains'] = True

    return items_dict


if args.folder == "None":
    f = askdirectory(title = 'Select an experiment folder', initialdir="G:/DREAM/from_github/PAMFluo/Experiments")  # pops up a window to select your file 
else:
    f = args.folder
    
W = 100



show = True #option to output intermediary images in the segmentation process
# Initialize plotting tools
g = alienlab.plot.ShowFigure()
g.figsize = (15,7)
g.save_folder = f
g.date = False
p = alienlab.plot.PlotFigure()
p.figsize = (15,7)
p.save_folder = f
p.date = False


file_path = f + "/video.tiff"
video_file = tiff.imread(file_path)
v_high = np.mean(video_file, axis = (1,2))
file_path = f + "/video_timing.npy"
v_time_high = np.load(file_path)
t0 = v_time_high[0]
v_time_high = v_time_high -t0
v_time_high = v_time_high%(3600*24)

threshold = np.max(v_high)*0.5

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
axs[0].plot(v_time_high[v_high < threshold], v_high[v_high <threshold], label = f)
axs[0].set_xlabel("time(s)")
axs[0].legend()
axs[1].plot(v_time_high[v_high > threshold], v_high[v_high >threshold])
axs[1].set_ylabel("time(s)")
p.saving(fig)

im_ref = np.mean(video_file[100:200], axis = 0)

labels, img, fig= predict_labels(im_ref)
del model
p.save_name = "label_prediction"
p.saving(fig)
np.save(f + "/image_ref.npy", img)
np.save(f + "/labels.npy", labels)

################################################################################################

data_sequence = {}

start = 0

zone = []
time_zone = []
video_zone = []

indices_clean = 59, 216, len(v_high)
THRESHOLD = np.max(v_high)*0.5
N_mvg = 5

for i in range(len(indices_clean)-1):
        ind_range = list(range(indices_clean[i],indices_clean[i+1]))
        trace = v_high[ind_range]
        ind = trace > THRESHOLD
        nind = trace <= THRESHOLD
        nind[0]=0
        nind[-1]=0
        zone.append(trace[ind])
        zone.append(mvgavg(trace[nind], N_mvg))
        
        v_time = v_time_high[ind_range]
        time_zone.append(v_time[ind])
        time_zone.append(mvgavg(v_time[nind], N_mvg))
        
        vid = video_file[ind_range]
        video_zone.append(vid[ind])
        video_zone.append(mvgavg(vid[nind], N_mvg))

   
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

#time_zone[2] = np.append(time_zone[3][0], time_zone[2])
#zone[2] = np.append(zone[0][0], zone[2])
#video_zone[2] = np.concatenate([video_zone[0], video_zone[2]])


for k in range(len(zone)):
        print(k)
        axs[k%2][(k//2)%2].plot(time_zone[k], zone[k], label = k)
        axs[k%2][(k//2)%2].set_title("zone %d"%k)
        axs[k%2][(k//2)%2].set_xlabel("time (s)")
        axs[k%2][(k//2)%2].legend()
plt.savefig(f + "/phase_graph.pdf")

##################################################################################################

flat_mask = labels.flatten()
algae_list = np.unique(labels)

L = len(video_zone) + 1
col = 2
fig, axs = plt.subplots(col, 3, figsize=(12,12))

video_list_init = video_zone
time_list = time_zone
video_list = []
for video in video_list_init:
    if video.shape[0]!=0:
        video_list.append(video)

for i, video in enumerate(video_list): 
    video_list[i] = video.reshape(video.shape[0], -1)





video = video_list[1]
for i in range(0, 6):
    inds = random.sample(sorted(algae_list[1:]), 4)
    for ind in inds:
        if ind != 0:
            pos = flat_mask == ind

            y = np.mean(video[:, pos], axis = 1)    
            x = time_zone[1]
            params, ypred = get_fit(y, x, give_y = True)
            axs[(i)%col][(i)//col].plot(x, ypred, label = params[1])
            axs[(i)%col][(i)//col].plot(x, y, '.')
            axs[(i)%col][(i)//col].set_title("zone %d"%(1+1))
            axs[(i)%col][(i)//col].legend()

plt.savefig(f + "/example_algae.pdf")

###########################################################################################################
"""
z = 1
S = video_zone[z].shape[0]
i=50
x = np.mean(video_zone[1].reshape(S, -1)[:,flat_mask==i], axis = 1)
ref = np.mean(video_zone[1].reshape(S, -1)[:,flat_mask != 0], axis = 1)

def cor_short(x, ref):
    x = (x - np.mean(x)) / (np.std(x))
    ref = (ref - np.mean(ref)) / (np.std(ref))
    #plt.figure()
    #plt.plot(x)
    #plt.plot(ref)
    return np.corrcoef(x[0:8], ref[0:8])[1][0]

def return_cor(x, ref):
    return np.corrcoef(x[0:15], ref[0:15])[1,0]

def compare_ratio(x):
    return np.mean((x[0:1])-np.mean(x[15:25:]))/np.mean(x[15:25:])

fig, axs = plt.subplots(3, 1, figsize = (5, 8))

ref_curve_control =  np.load("specs/control.npy")
ref_curve_HL = np.load("specs/HL.npy")

params_high_HL = Parallel(n_jobs = -1 )(delayed(cor_short)(np.mean(video_zone[z].reshape(S, -1)[:,flat_mask==i], axis = 1), ref_curve_HL) for i in range(len(algae_list)))
params_high_control = Parallel(n_jobs = -1 )(delayed(cor_short)(np.mean(video_zone[z].reshape(S, -1)[:,flat_mask==i], axis = 1), ref_curve_control) for i in range(len(algae_list)))

#params_high = Parallel(n_jobs = -1 )(delayed(return_cor)(np.mean(video_zone[z].reshape(S, -1)[:,flat_mask==i], axis = 1), ref_curve) for i in range(len(algae_list)))
#params_high = Parallel(n_jobs = -1 )(delayed(compare_ratio)(np.mean(video_zone[z].reshape(S, -1)[:,flat_mask==i], axis = 1)) for i in range(len(algae_list)))

im_tau_high = np.zeros(labels.shape)
for j in range(len(params_high_HL)):
    if j==0:
        im_tau_high[labels==j]=0.5
    else:
        im_tau_high[labels==j]=params_high_HL[j]>0.9

params_high_HL = np.array(params_high_HL)
params_high_control = np.array(params_high_control)

axs[0].imshow(im_tau_high)

axs[1].hist(params_high_HL, 40,  density= False, alpha=0.5, facecolor = "red", edgecolor = "black")#color = (1,0,0,0.5))
axs[2].hist(params_high_control, 40,  density= False, alpha=0.5, facecolor = "white", edgecolor = "black")#color = (1,0,0,0.5))
axs[2].set_xlabel("correlation")
axs[2].set_title('zone %d'%z)
axs[1].set_xlabel("correlation")
axs[1].set_title('zone %d'%z)
plt.savefig(f + "/dispersion_zone%d.pdf"%z)

np.save(f + "/im_tau_high.npy", im_tau_high)

"""
###############################################

file_path = f + "/video.tiff"
video_file_high = tiff.imread(file_path)

items_dict = dict_trajectories(labels, video_file_high)
items_dict.pop("0")


exp_dict =  {}
exp_dict["folder"] = f
exp_dict["labels"] = labels
exp_dict["im_ref"] = im_ref
exp_dict["items_dict"] = items_dict
exp_dict["time"] = v_time_high
np.save(f + "/exp_dict.npy", exp_dict)




##################################################
from alienlab.widget import click_to_graph

file_path = f + "/video.tiff"
video_file_high = tiff.imread(file_path)

video_list =  video_zone
time_list =  time_zone
initial_condition_list = [[],[],[],[],[],[],[],[]]
click_to_graph(labels, im_ref ,video_list[0:4],time_list, get_fit, clipit=False, col = 2)

if args.folder != "None":
    plt.close("all")