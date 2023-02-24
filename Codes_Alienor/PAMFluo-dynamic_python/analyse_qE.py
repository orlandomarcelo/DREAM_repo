
#image analysis
import sys
import os

import cv2

import skimage.io
import alienlab.plot
from alienlab.improcessing import normalize, grey_to_rgb, make_binary
import alienlab.segment
from alienlab.fo import FramesOperator
import alienlab.io
import glob
from alienlab.regression_func import *
import time
import os
import numpy as np
import matplotlib.pyplot as plt

def init_image(file_path):
    frames_full = skimage.io.imread(file_path)

    #frames_full = np.stack([frames_full[:,:,1]]*10, 0) 
    #uncomment this line if you have a single RGB image. The [:,:,1] stands for selection of the green channel

    FO = FramesOperator(frames_full)
    im = normalize(FO.frames[0], 0, 1)
    im = grey_to_rgb(im)*255
    FO.compute_stats()

    # CROP
    #y, x = alienlab.io.select_roi(np.uint8(im)) #select area of interest

    FO.x = 100, 800
    FO.y = 100, 800
    #FO.crop() #crop image
    return FO


def segment_image(FO, contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, ni, interact = True, showit = False):
    
    start_time = time.time()
    inds_max = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*1.1) # Select only images with high intensity to increase contrast and lower computation time
    inds_med = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*0.1) # Select only images with high intensity to increase contrast and lower computation time
    FO.selected_inds = np.array(list(set(inds_med) - set(inds_max)))[250:300]
    
    def make_mask(contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, soft_hard_contrast, soft_hard_autolevel):
        #apply contrast filter to all frames
        frames_contrast = FO.apply(skimage.filters.rank.enhance_contrast,  selem = skimage.morphology.disk(contrast))
        #apply autolevel filter to all frames
        frames_autolevel = FO.apply(skimage.filters.rank.autolevel, selem = skimage.morphology.disk(autolevel))
        #sum the contrast images to get a reference grey-level contrast image
        frame_contrast = np.sum(frames_contrast, axis = 0)
        #sum the autolevel images to get a reference grey-level autolevel image
        frame_autolevel = np.sum(frames_autolevel, axis = 0)
        #obtain contrast mask from reference contrast image
        mask_contrast = make_binary(frame_contrast, soft_hard = soft_hard_contrast)
        #otbain autolevel mask from reference autolevel image
        mask_autolevel =  make_binary(frame_autolevel, soft_hard = soft_hard_autolevel)
        #intersection of contrast aud autolevel masks
        mask_intersect = mask_contrast * mask_autolevel
        #clean the masks with a binary opening
        mask_intersect = skimage.morphology.binary_opening(mask_intersect, selem = skimage.morphology.disk(disk_size))
        #mask_intersect = skimage.morphology.binary_erosion(mask_intersect, selem = skimage.morphology.disk(disk_size))

        #reference image of altitude for the watershed
        auto_contrast = normalize(mask_intersect * frame_autolevel)
        print("--- Computed binary mask in %04f seconds ---" % (time.time() - start_time))

        ni.g.cmap = "inferno"
        if showit:
            ni.g.figsize = (20,8)
            ni.g.title_list =  'contrast', 'contrast threshold', 'mask intersect','autolevel', 'autolevel threshold','segmentation image'
            ni.g.col_num = 3
            fig = ni.g.multi([frame_contrast, mask_contrast, mask_intersect, 
                           frame_autolevel, mask_autolevel,  auto_contrast])
            ni.g.save_name = 'Segmentation reference'
            ni.g.saving(fig)
            
        return auto_contrast, mask_intersect
    
    auto_contrast, mask_intersect = make_mask(contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, soft_hard_contrast = 1, soft_hard_autolevel = 1)
    ref, mask = make_mask(contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, soft_hard_contrast = 0.3, soft_hard_autolevel = 0.5)

    start_time = time.time()

    #locate the local maxima
    local_maxi = alienlab.segment.local_maxima(auto_contrast, max_contrast, ni.g,
                                                     ref_distance = dist_max, mask = mask_intersect, show = showit)
    #perform watershed segmentation
    watershed_im_mask = alienlab.segment.watershed(ref*mask_intersect, mask , local_maxi,
                                                         ni.g, ref_distance = dist_seg, show = True)
    segmented = watershed_im_mask
    print("--- Computed segmentation in %04f seconds ---" % (time.time() - start_time))

    if showit:
        alienlab.segment.show_segmentation(FO, segmented, ni.g)
        
    if interact == False:
       return watershed_im_mask, FO

def trajectories(segmented, FO):
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
        pixel_values = FO.frames[:,x_coords, y_coords]
        items_dict[key]['pixel_values'] = pixel_values
        items_dict[key]['surface'] = pixel_values.shape[1]
        items_dict[key]['mean'] = np.mean(pixel_values, axis = 1)
        items_dict[key]['std'] = np.std(pixel_values, axis = 1)
        items_dict[key]['remains'] = True
        items_dict[key]['algae_image'] = get_algae_im(x_coords, y_coords, np.mean(FO.frames[FO.selected_inds], axis = 0))

    return items_dict

   
def get_algae_im(xcoords, ycoords, imref):
    back = np.zeros((60,60))
    xmin = max(xcoords.min() - 5, 0)
    xmax = min(xcoords.max() + 5, imref.shape[0], xmin + 60)
    ymin = max(ycoords.min() - 5, 0)
    ymax = min(ycoords.max() + 5, imref.shape[1], ymin + 60)
    start = 0
    
    back[start:start + xmax-xmin, start: start+ymax-ymin] = imref[xmin:xmax, ymin:ymax]
    #back = v
    #print(back.shape)
    return back#, v, theta0


def save_video(video_name, folder):
    shape = cv2.imread(glob.glob(folder + "/*.jpg")[0]).shape
    out = cv2.VideoWriter(folder + "_" + video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 4, (shape[1], shape[0]))
    for i, file in enumerate(glob.glob(folder + "/*.jpg")):
        out.write(cv2.imread(file)) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()
    

def analyse_qE_data(folder):
    g = alienlab.plot.ShowFigure()
    g.figsize = (15,7)
    g.save_folder = folder + "/analysis"
    g.date = False
    p = alienlab.plot.PlotFigure()
    p.figsize = (15,7)
    p.save_folder = folder + "analysis"
    p.date = False
    FO = init_image(folder + "/video.tiff")
    mask, FO = segment_image(FO, contrast = 6, autolevel = 5, dist_max = True, dist_seg=True, disk_size = 1, max_contrast = 3, interact = False, showit= True)

    g.cmap = "tab20"
    g.figsize = (10,5)
    fig = g.multi(mask)
    plt.savefig(folder + "/segmented.pdf")
    L, H  = np.shape(mask)

    file_path = folder + "/video_timing.npy"
    v_time = np.load(file_path)
    items_dict = trajectories(mask, FO)
    items_dict.pop("0")

    im_ref = np.mean(FO.frames, axis = 0)
    exp_dict =  {}
    exp_dict["folder"] = folder
    exp_dict["labels"] = mask
    exp_dict["im_ref"] = im_ref
    k = FO.frames
    k = np.reshape(k, (k.shape[0], -1))
    exp_dict["total_mean"] = np.mean(k[:, (mask.flatten())!=0], axis = 1)
    exp_dict["items_dict"] = items_dict
    exp_dict["time"] = v_time
    np.save(folder + "/analysis/items_dict.npy", exp_dict)