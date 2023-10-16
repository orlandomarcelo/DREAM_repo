import numpy as np
from sacred import Ingredient
from config_DAQ import *
import matplotlib.pyplot as plt
import ipdb
import skimage
from alienlab.improcessing import normalize, grey_to_rgb, make_binary
import alienlab.segment
from alienlab.fo import FramesOperator
import alienlab.io
import time
import scipy

"""basic bricks for experiments"""


segmentation = Ingredient('segementation')

@segmentation.config
def cfg():
    contrast = 2
    autolevel = 3
    dist_max = True
    dist_seg = True
    disk_size = 2
    max_contrast = 3

    max_im_ref = 0.8



@segmentation.capture
def preprocess_movie(file_path, p, show = True):
    frames_full = skimage.io.imread(file_path)

    #frames_full = np.stack([frames_full[:,:,1]]*10, 0) 
    #uncomment this line if you have a single RGB image. The [:,:,1] stands for selection of the green channel

    FO = FramesOperator(frames_full)
    im = normalize(FO.frames[0], 0, 1)
    im = grey_to_rgb(im)*255

    # CROP
    #y, x = alienlab.io.select_roi(np.uint8(im)) #select area of interest
    #FO.x = x
    #FO.y = y
    #FO.crop() #crop image

    start_time = time.time()
    FO.compute_stats() #compute various statistical values on the frames and the pixels
    FO.normalize(0, 1)
    print("--- Computed frames statistics in %04f seconds ---" % (time.time() - start_time))

    #FO.global_stats: each array has size N, number of frames and represents the stats of each frame
    #FO.frames_stats: each array has size FO.x, FO.y and is an image representing the N frames stats overlayed

    if show:
        p.title = 'statistics'
        p.xlabel = 'frame number'
        p.ylabel = 'amplitude'
        p.label_list = ['max', 'min', 'mean', 'std']
        fig = p.plotting(np.asarray(FO.inds), [FO.global_stats['max'], 
                            FO.global_stats['min'], 
                            FO.global_stats['mean']])
        p.save_name = 'frames_stats'
        p.saving(fig)

    ''' IMAGE SEGMENTATION '''

    # selection of the frames with high dynamics that will be used for the image segmentation process.
    # Let M be the highest value taken by a pixel in all the frames of the video. The frame F is kept for processing only if at
    # least one pixel in the frame F has a value above 0.8*M. 
    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*0.6)
    return FO


@segmentation.capture
def segment_movie(FO, g, _run, contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, max_im_ref, showit = True):
        
    start_time = time.time()
    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*max_im_ref) # Select only images with high intensity to increase contrast and lower computation time

    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*0.98) # Select only images with high intensity to increase contrast and lower computation time
    imref = FO.frames[FO.selected_inds].sum(axis = 0)
    fig = plt.figure()
    plt.imshow(imref)
    g.save_name = "im_ref"
    g.saving(fig)


    #apply contrast filter to all frames
    frames_contrast = FO.apply(skimage.filters.rank.enhance_contrast,  selem = skimage.morphology.disk(contrast))
    #apply autolevel filter to all frames
    frames_autolevel = FO.apply(skimage.filters.rank.autolevel, selem = skimage.morphology.disk(autolevel))
    #sum the contrast images to get a reference grey-level contrast image
    frame_contrast = np.sum(frames_contrast, axis = 0)
    #sum the autolevel images to get a reference grey-level autolevel image
    frame_autolevel = np.sum(frames_autolevel, axis = 0)
    #obtain contrast mask from reference contrast image
    mask_contrast = make_binary(frame_contrast, soft_hard = 1)
    #otbain autolevel mask from reference autolevel image
    mask_autolevel =  make_binary(frame_autolevel, soft_hard = 1)
    #intersection of contrast aud autolevel masks
    mask_intersect = mask_contrast * mask_autolevel
    #clean the masks with a binary opening
    mask_intersect = skimage.morphology.binary_opening(mask_intersect, selem = skimage.morphology.disk(disk_size))
    #reference image of altitude for the watershed
    auto_contrast = normalize(mask_intersect * frame_autolevel)
    print("--- Computed binary mask in %04f seconds ---" % (time.time() - start_time))

    g.cmap = "inferno"
    if showit:
        g.figsize = (40,15)
        g.title_list =  'contrast', 'contrast threshold', 'mask intersect','autolevel', 'autolevel threshold','segmentation image'
        g.col_num = 3
        fig = g.multi([frame_contrast, mask_contrast, mask_intersect, 
                    frame_autolevel, mask_autolevel,  auto_contrast])
        g.save_name = "Segmentation_reference"
        g.saving(fig)


    start_time = time.time()
    ref = auto_contrast
    mask = mask_intersect
    g.col_num=2
    #locate the local maxima
    local_maxi = alienlab.segment.local_maxima(auto_contrast, max_contrast, g,
                                                    ref_distance = dist_max, mask = mask, show = showit)
    #perform watershed segmentation
    watershed_im_mask = alienlab.segment.watershed(ref, mask, local_maxi,
                                                        g, ref_distance = dist_seg, show = False)
    segmented = watershed_im_mask
    print("--- Computed segmentation in %04f seconds ---" % (time.time() - start_time))

    if showit:
        alienlab.segment.show_segmentation(FO, segmented, g)
        
    return watershed_im_mask, FO


def segment_algae(FO, g, _run, contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast, soft_hard_contrast = 0.3, soft_hard_autolevel = 0.6, showit = True):
    
    start_time = time.time()
    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max()*0.98) # Select only images with high intensity to increase contrast and lower computation time

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
    mask_intersect = scipy.ndimage.binary_fill_holes(mask_intersect).astype(int)
    #clean the masks with a binary opening
    mask_intersect = skimage.morphology.binary_opening(mask_intersect, selem = skimage.morphology.disk(disk_size))
    #reference image of altitude for the watershed
    auto_contrast = normalize(mask_intersect * frame_autolevel)
    print("--- Computed binary mask in %04f seconds ---" % (time.time() - start_time))

    g.cmap = "inferno"
    if showit:
        g.figsize = (20,15)
        g.title_list =  'contrast', 'contrast threshold', 'mask intersect','autolevel', 'autolevel threshold','segmentation image', 'bla'
        g.col_num = 3
        fig = g.multi([frame_contrast, mask_contrast, mask_intersect, 
                       frame_autolevel, mask_autolevel,  auto_contrast, frame_contrast*frame_autolevel])
        g.save_name = 'Segmentation reference'
        g.saving(fig)

    start_time = time.time()
    ref = auto_contrast
    mask = mask_intersect
    #locate the local maxima
    local_maxi = alienlab.segment.local_maxima(auto_contrast, max_contrast, g,
                                                     ref_distance = dist_max, mask = mask, show = showit)
    #perform watershed segmentation
    watershed_im_mask = alienlab.segment.watershed(ref, mask, local_maxi,
                                                         g, ref_distance = dist_seg, show = False)
    segmented = watershed_im_mask
    print("--- Computed segmentation in %04f seconds ---" % (time.time() - start_time))

    if showit:
        alienlab.segment.show_segmentation(FO, segmented, g)
        

    return watershed_im_mask, FO



@segmentation.capture
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

    return items_dict


