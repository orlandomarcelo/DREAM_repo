import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import ipdb

from PIL import Image

from joblib import Parallel

from function_intensity_fit_map import threshold_convert, exp_decay, simple_tau, get_rate_from_timing, parallel_tau, save_npy_to_csv, plot_map, plot_hist


from skimage.transform import rescale, resize, downscale_local_mean
import skimage
from scipy.ndimage import binary_erosion

from sacred import Ingredient


import tifffile as tiff
from tqdm import tqdm

sigma_480 = 198 #m2/mol
sigma_405 = 415 #m2/mol
tau_relax = 0.014


process_video = Ingredient('process_video')

def dump_db(_run, save_name):
    _run.add_artifact(save_name, os.path.split(save_name)[1])


@process_video.config
def cfg():
    filter = "minimum"
    smoothing = 1
    threshold = 0.8
    extension = ".png"
    limits = (0, 10000)




@process_video.capture
def make_maps(_run, save_folder, video, timing, extension, filter, smoothing, limits):
    framerate = get_rate_from_timing(timing)
    timing = timing#/framerate
    fig = plt.figure()
    plt.xlabel('time (s)')
    plt.ylabel('mean signal')
    avg_video = np.mean(video[0:], axis = (1,2))
    plt.plot(timing, avg_video, "ko")
    save_name = save_folder + "/full_frames" + extension
    plt.savefig(save_name)
    dump_db(_run, save_name)

    fig = plt.figure()
    slices = np.mean(video, axis = (1,2))
    #frames = video[slices>np.quantile(slices, 0.8)]
    #frame_ref = np.mean(frames, axis =0)
    frame_ref = video[np.argmax(slices)]
    plt.imshow(frame_ref)
    save_name = save_folder + "/ref_image" 
    plt.savefig(save_name+ extension)
    dump_db(_run, save_name+ extension)
    image = Image.fromarray(frame_ref)

    image.save(save_name + ".tiff")    
    dump_db(_run, save_name + ".tiff")

    D_image = downscale_local_mean(frame_ref, (smoothing,smoothing))

    #downscaling 
    video_downscaled = []
    for i in range(0, video.shape[0]):
        video_downscaled.append(downscale_local_mean(video[i], (smoothing, smoothing)))
    video_downscaled = np.array(video_downscaled)
    init_shape = np.copy(video_downscaled.shape)
    L, H = init_shape[1:]


    #mask
    threshold_method = threshold_convert(filter)
    threshold = threshold_method(D_image)
    mask = D_image>threshold/2
    plt.imshow(mask)
    save_name = save_folder + "/mask" + extension
    plt.savefig(save_name)
    dump_db(_run, save_name)


    #downscale video mask
    video_2D = np.copy(video_downscaled)
    video_downscaled = video_downscaled.reshape(video_downscaled.shape[0], -1)

    to_evaluate = video_downscaled[:,mask.flatten()]

    plt.plot(np.mean(to_evaluate, axis = 1), 'ok')

    #focus on curve
    start = np.argmax(avg_video)
    stop = -1


    #fit on the mean value before performing the fit on each pixel
    params = simple_tau(np.mean(to_evaluate[start:stop], axis = 1), timing[start:stop], sample_rate = framerate)

    time = timing[start:stop]
    time -= time[0]
    plt.figure()
    plt.plot(time, np.mean(to_evaluate[start:stop], axis = 1), "ok", label = "raw data")

    plt.plot(time, exp_decay(params, time), "k", label = "fit")
    plt.xlabel('time(s)')
    plt.ylabel("fluorescence")
    plt.legend()

    save_name = save_folder + "/decay_mask" + extension
    plt.savefig(save_name)
    dump_db(_run, save_name)

    ## check saturation
    sat = np.sum(to_evaluate[0]==255)/len(to_evaluate[0].flatten())*100
    print("percent of saturated pixels in the mask: ", sat)

    if sat > 2:
        print("WARNING, more than 2% of the pixels are saturated on the first frame, try acquiring the movie with shorter exposure/smaller gain.")

    ### Parallel pixel computing

    tau_lists = Parallel(n_jobs = -1 )(parallel_tau(to_evaluate[start:stop,i], timing[start:stop], sample_rate = framerate) for i in tqdm(range(to_evaluate.shape[1])))

    

    tau_480 = np.array(tau_lists)
    tau_480_map = np.zeros(mask.shape)
    tau_480_map[mask] = tau_480
    csv_file = save_folder + "/tau_480_map.csv"
    save_npy_to_csv(tau_480_map, csv_file)
    dump_db(_run, csv_file)

    I_480 = 1e6*(1/tau_480 - tau_relax)/sigma_480
    I_480_map = np.zeros(mask.shape)
    I_480_map[mask] = I_480

    save_name = save_folder + "/I_480_map"
    save_npy_to_csv(I_480_map, save_name + ".csv")
    dump_db(_run, save_name + ".csv" )

    image = Image.fromarray(I_480_map)
    image.save(save_name + ".tiff")    
    dump_db(_run, save_name + ".tiff")

    plt.rcParams['image.cmap'] = 'viridis'

    save_name = save_folder + "/I_480_map_plot" 
    I_map = plot_map(I_480, I_480_map, save_name+ extension, limits = limits)
    dump_db(_run, save_name + extension)
    save_npy_to_csv(I_map, csv_file + ".csv")
    dump_db(_run, save_name + ".csv")

    save_name = save_folder + "/I_480_hist"
    I_distrib = plot_hist(I_map, save_name+ extension)
    dump_db(_run, save_name + extension)
    save_npy_to_csv(I_distrib, save_name + ".csv")
    dump_db(_run, save_name + ".csv")

    np.save(save_folder + "/video2D.npy", video_2D[start:stop])

    csv_file = save_folder + "/framerate.csv"
    save_npy_to_csv([framerate], csv_file)
    dump_db(_run, save_name + ".csv")
