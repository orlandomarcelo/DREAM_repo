#image analysis
import skimage.io
import imageio
import alienlab.plot
from alienlab.improcessing import normalize, grey_to_rgb, make_binary
import alienlab.segment
from alienlab.fo import FramesOperator
import alienlab.io

import os
import numpy as np
import matplotlib.pyplot as plt
import importlib

importlib.reload(alienlab.fo)
importlib.reload(alienlab.segment)


def segment_image(contrast, autolevel, dist_max, dist_seg, disk_size, max_contrast):
    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats[
        'max'].max() * 0.8, g)  # Select only images with high intensity to increase contrast and lower computation time

    # apply contrast filter to all frames
    frames_contrast = FO.apply(skimage.filters.rank.enhance_contrast, footprint=skimage.morphology.disk(contrast))
    # apply autolevel filter to all frames
    frames_autolevel = FO.apply(skimage.filters.rank.autolevel, footprint=skimage.morphology.disk(autolevel))
    # sum the contrast images to get a reference grey-level contrast image
    frame_contrast = np.sum(frames_contrast, axis=0)
    # sum the autolevel images to get a reference grey-level autolevel image
    frame_autolevel = np.sum(frames_autolevel, axis=0)
    # obtain contrast mask from reference contrast image
    mask_contrast = make_binary(frame_contrast, soft_hard=1)
    # otbain autolevel mask from reference autolevel image
    mask_autolevel = make_binary(frame_autolevel, soft_hard=1)
    # intersection of contrast aud autolevel masks
    mask_intersect = mask_contrast * mask_autolevel
    # clean the masks with a binary opening
    mask_intersect = skimage.morphology.binary_opening(mask_intersect, footprint=skimage.morphology.disk(disk_size))
    # reference image of altitude for the watershed
    auto_contrast = normalize(mask_intersect * frame_autolevel)

    ref = auto_contrast
    mask = mask_intersect
    # locate the local maxima
    local_maxi = alienlab.segment.local_maxima(auto_contrast, max_contrast, g,
                                               ref_distance=dist_max, mask=mask, show=False)
    # perform watershed segmentation
    watershed_im_mask = alienlab.segment.watershed(ref, mask, local_maxi,
                                                   g, ref_distance=dist_seg, show=False)
    segmented = watershed_im_mask

    return watershed_im_mask, FO

def im_seg(file):
    # Pre-processing
    # read the stacked frame. dim = NxHxW (N images in the video, Heigt, Width)

    frames_full = file

    # frames_full = np.stack([frames_full[:,:,1]]*10, 0)
    # uncomment this line if you have a single RGB image. The [:,:,1] stands for selection of the green channel

    FO = FramesOperator(frames_full)
    im = normalize(FO.frames[0], 0, 1)
    im = grey_to_rgb(im) * 255

    # CROP
    # y, x = alienlab.io.select_roi(np.uint8(im)) #select area of interest
    # FO.x = x
    # FO.y = y
    # FO.crop() #crop image

    FO.compute_stats()  # compute various statistical values on the frames and the pixels
    FO.normalize(0, 1)

    # FO.global_stats: each array has size N, number of frames and represents the stats of each frame
    # FO.frames_stats: each array has size FO.x, FO.y and is an image representing the N frames stats overlayed

    ''' IMAGE SEGMENTATION '''

    # selection of the frames with high dynamics that will be used for the image segmentation process.
    # Let M be the highest value taken by a pixel in all the frames of the video. The frame F is kept for processing only if at
    # least one pixel in the frame F has a value above 0.8*M.

    FO.selected_inds = FO.select_frames(FO.global_stats['max'], FO.global_stats['max'].max() * 0.8)

    #Segmentation parameters

    contrast = 2
    autolevel = 5
    dist_max = False
    dist_seg = False
    disk_size = 2
    max_contrast = 3

    # apply contrast filter to all frames
    frames_contrast = FO.apply(skimage.filters.rank.enhance_contrast, footprint=skimage.morphology.disk(contrast))
    # apply autolevel filter to all frames
    frames_autolevel = FO.apply(skimage.filters.rank.autolevel, footprint=skimage.morphology.disk(autolevel))
    # sum the contrast images to get a reference grey-level contrast image
    frame_contrast = np.sum(frames_contrast, axis=0)
    # sum the autolevel images to get a reference grey-level autolevel image
    frame_autolevel = np.sum(frames_autolevel, axis=0)
    # obtain contrast mask from reference contrast image
    mask_contrast = make_binary(frame_contrast, soft_hard=1)
    # otbain autolevel mask from reference autolevel image
    mask_autolevel = make_binary(frame_autolevel, soft_hard=1)
    # intersection of contrast aud autolevel masks
    mask_intersect = mask_contrast * mask_autolevel
    # clean the masks with a binary opening
    mask_intersect = skimage.morphology.binary_opening(mask_intersect, footprint=skimage.morphology.disk(disk_size))
    # reference image of altitude for the watershed
    auto_contrast = normalize(mask_intersect * frame_autolevel)

    ref = auto_contrast
    mask = mask_intersect
    # locate the local maxima
    local_maxi = alienlab.segment.local_maxima(auto_contrast, max_contrast, ref_distance=False, mask=mask)
    # perform watershed segmentation
    watershed_im_mask = alienlab.segment.watershed(ref, mask, local_maxi, ref_distance=False)
    segmented = watershed_im_mask

    return watershed_im_mask, FO


