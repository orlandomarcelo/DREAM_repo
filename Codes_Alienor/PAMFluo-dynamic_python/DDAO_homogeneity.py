import pandas as pd
from alienlab.init_logger import logger
from tqdm import tqdm
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg
import datetime
import tifffile as tiff

import cv2

from ArduinoControl import python_comm

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera
import ipdb

from ThorlabsControl.KPZ101 import KPZ101

ex = Experiment('autofocus', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "autofocus"
    acq_time = 14*60
    actinic_filter = 0
    move_plate = False
    length_SP = 200
    period_SP= 10*sec
    limit_purple = 0
    limit_blue = 0
    limit_red=0
    limit_green=0
    trigger_color = "no_LED_trig"
    sample_rate=1e5
    exposure = 800
    gain  = 50


@ex.automain
def send_IBPC(_run, name, limit_blue, gain, exposure, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, period_SP, length_SP, period_ML, actinic_filter, move_plate):


    def grab_image(exposure, num_frames, gain):
        thread, cam = init_camera(exposure = exposure, num_frames = num_frames, gain =  gain)

        # Acquisition
        ni.detector_start()
        
        thread.start()
        time.sleep(1)
        thread.stop_thread = True
        thread.join()
        cam.exit()
        try: #color camera vs grey
            L, H = cam.video[0].shape
            video = np.array(cam.video)

        except:
            L, H, d = cam.video[0].shape
            video = np.array(cam.video, dtype = "uint16")[:,:,:, 2]
        return video
    
    
    
    def show_images(images, cols = 1, titles = None):
        """Display a list of images in a single figure with matplotlib.
        
        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.
        
        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).
        
        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        #plt.show()
    
    # LED control tool DC4104 Thorlabs
    logger.name = name


    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo = True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]

    fwl.move_to_filter(filters[actinic_filter])
    


    

    def focus(min, max, step, link_arduino, exposure, gain):

        add_master_digital_pulse(link_arduino, pins['no_LED_trig'], 0, 1000, 20, 0) #trigger
        add_digital_pulse(link_arduino, pins["green"], 0, 20*minute, 20*minute, 0)

    
        voltages =  np.linspace(min, max, step)
        start_measurement(link_arduino)
        images = []
        for voltage in voltages:
            ctrlLED.set_user_limit(LED_green, voltage)

            print(voltage)
            time.sleep(10)
            images.append(grab_image(exposure, 1, gain)[0])
        
        stop_measurement(link_arduino)
        
        show_images(images, cols = 3)
        #ipdb.set_trace()

        
        return  voltages, images
        
    min = 0
    max = 500
    steps = 10

    v, images = focus(min, max, steps, link_arduino, exposure, gain)
 
        #ipdb.set_trace()
        
        
    close_all_links(*all_links)
    

    np.save(ni.p.save_folder + "/voltage.npy", v)
    np.save(ni.p.save_folder + "/images.npy", images)
    _run.add_artifact(ni.g.save_folder + "/voltage.npy")
    _run.add_artifact(ni.g.save_folder + "/images.npy")

    #ipdb.set_trace()
    
    for i in range(len(images)):
        fig = plt.figure()
        plt.imshow(images[i])
        ni.p.saving(fig)
        _run.add_artifact(ni.p.save_path + ni.p.extension)
    ipdb.set_trace()


    
    
    
    
    #ref: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    #ref: https://github.com/antonio490/Autofocus
    #ref: https://sites.google.com/site/cuongvt101/research/Sharpness-measure
    
